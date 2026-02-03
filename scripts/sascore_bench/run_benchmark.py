#!/usr/bin/env python
"""
=============================================================================
SAScore Benchmark: Combined Policy (Filtering + Ranking)
=============================================================================

This script runs retrosynthesis benchmarks on SAScore-stratified datasets
using SynPlanner's combined policy (filtering + ranking networks).

QUICK START
-----------

1. Clone the repository and install dependencies:

   git clone https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner.git
   cd SynPlanner
   uv sync --extra cpu   # or --extra cuda for GPU support

2. Run the benchmark (data will be auto-downloaded from Hugging Face):

   uv run sascore-benchmark

   Or directly:
   uv run python scripts/sascore_bench/run_benchmark.py

Note: The script will automatically download required data (reaction rules, 
      policies, building blocks, and benchmark targets) from Hugging Face 
      on first run if they're not found locally.

CONFIGURATION
-------------

The script auto-discovers config.yaml in the same folder.
You can also specify a custom config:

   uv run sascore-benchmark --config /path/to/custom_config.yaml

See scripts/sascore_bench/config.yaml for all available options.

OUTPUT
------

Results are saved to ./benchmark_results_combined_policy/<timestamp>/ with:
  - config.json: Full configuration used
  - summary.json: Summary statistics
  - sascore_X.X_Y.Y/stats.csv: Per-target results for each SAScore range

After running, use 'uv run sascore-plot' to visualize results.

=============================================================================
"""

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from tqdm.auto import tqdm

from synplan.chem.utils import mol_from_smiles
from synplan.mcts.tree import Tree
from synplan.utils.config import TreeConfig, RolloutEvaluationConfig
from synplan.utils.loading import (
    load_building_blocks,
    load_combined_policy_function,
    load_evaluation_function,
    load_reaction_rules,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default config location (same folder as this script)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def ensure_data_downloaded(data_folder: Path, benchmark_folder: Path) -> None:
    """
    Download required data from Hugging Face if not present locally.
    
    This function checks if required files exist, and if not, downloads them
    from the SynPlanner Hugging Face repository.
    
    Args:
        data_folder: Root folder where data should be stored
        benchmark_folder: Folder where benchmark data should be stored
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError
    
    # Define required files
    required_files = [
        ("ranking_policy", "uspto/weights/ranking_policy_network.ckpt"),
        ("filtering_policy", "uspto/weights/filtering_policy_network.ckpt"),
        ("reaction_rules", "uspto/uspto_reaction_rules.pickle"),
        ("building_blocks", "building_blocks/building_blocks_em_sa_ln.smi"),
    ]
    
    # Check which files are missing
    missing_files = []
    for name, rel_path in required_files:
        full_path = data_folder / rel_path
        if not full_path.exists():
            missing_files.append((name, rel_path))
    
    # Check if benchmark files are missing
    benchmark_missing = False
    if not benchmark_folder.exists() or not any(benchmark_folder.glob("targets_with_sascore_*.smi")):
        benchmark_missing = True
        # Add benchmark files to download list
        missing_files.append(("benchmark_data", "benchmarks/sascore"))
    
    # If any files are missing, download them
    if missing_files:
        logger.info(f"Downloading {len(missing_files)} missing files from Hugging Face...")
        
        # Download files individually
        REPO_ID = "Laboratoire-De-Chemoinformatique/SynPlanner"
        
        for name, rel_path in missing_files:
            if name == "benchmark_data":
                # For benchmark data, we need to download the whole folder
                # List the known benchmark files
                logger.info(f"  - benchmark files: benchmarks/sascore/*.smi")
                for i in range(1, 6):  # sascore ranges 1.5_2.5, 2.5_3.5, 3.5_4.5, 4.5_5.5, 5.5_6.5
                    sascore_low = 1.5 + (i - 1)
                    sascore_high = sascore_low + 1.0
                    filename = f"targets_with_sascore_{sascore_low}_{sascore_high}.smi"
                    try:
                        hf_hub_download(
                            repo_id=REPO_ID,
                            subfolder="benchmarks/sascore",
                            filename=filename,
                            local_dir=str(data_folder),
                        )
                    except RemoteEntryNotFoundError:
                        logger.warning(f"    Benchmark file not found: {filename}")
            else:
                # Split path into subfolder and filename
                parts = rel_path.split("/")
                if len(parts) > 1:
                    subfolder = "/".join(parts[:-1])
                    filename = parts[-1]
                else:
                    subfolder = ""
                    filename = rel_path
                
                logger.info(f"  - {name}: {rel_path}")
                
                # Try to download the file, first try .zip version for .smi files
                downloaded = False
                if filename.endswith(".smi"):
                    try:
                        zip_path = Path(hf_hub_download(
                            repo_id=REPO_ID,
                            subfolder=subfolder,
                            filename=f"{filename}.zip",
                            local_dir=str(data_folder),
                        ))
                        # Extract the zip file
                        import zipfile
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            zf.extractall(zip_path.parent)
                        zip_path.unlink()  # Remove the zip file after extraction
                        downloaded = True
                    except RemoteEntryNotFoundError:
                        # If .zip doesn't exist, try the plain file
                        pass
                
                if not downloaded:
                    # Download the file directly (not zipped)
                    try:
                        hf_hub_download(
                            repo_id=REPO_ID,
                            subfolder=subfolder,
                            filename=filename,
                            local_dir=str(data_folder),
                        )
                    except RemoteEntryNotFoundError:
                        logger.error(f"    File not found on Hugging Face: {rel_path}")
                        raise
        
        logger.info("Download completed successfully!")
    else:
        logger.info("All required data files found locally.")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Auto-discovers config.yaml in the script's folder if no path specified.

    Args:
        config_path: Path to YAML config file. If None, uses default config.yaml

    Returns:
        Dictionary with configuration sections: paths, benchmark, policy, tree, evaluation

    Example:
        >>> config = load_config()
        >>> print(config["policy"]["top_rules"])
        100
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Expected at: {DEFAULT_CONFIG_PATH}"
        )

    logger.info(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_policy_from_config(config_path: Optional[Path] = None):
    """
    Load just the combined policy function from config.

    This is useful when you want to use the policy in your own code
    without running the full benchmark.

    Args:
        config_path: Path to YAML config file. If None, uses default config.yaml

    Returns:
        policy_function: The combined policy function ready to use

    Example:
        >>> from scripts.sascore_bench.run_benchmark import load_policy_from_config
        >>> policy = load_policy_from_config()
        >>> # Use policy in your own Tree or search
    """
    config = load_config(config_path)
    paths_cfg = config["paths"]
    policy_cfg = config["policy"]

    data_folder = Path(paths_cfg["data_folder"]).resolve()
    ranking_policy_path = data_folder / paths_cfg["ranking_policy"]
    filtering_policy_path = data_folder / paths_cfg["filtering_policy"]

    # Verify paths exist
    for path in [ranking_policy_path, filtering_policy_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    logger.info("Loading combined policy (filtering + ranking)...")
    policy_function = load_combined_policy_function(
        filtering_weights_path=str(filtering_policy_path),
        ranking_weights_path=str(ranking_policy_path),
        top_rules=policy_cfg["top_rules"],
        rule_prob_threshold=policy_cfg["rule_prob_threshold"],
    )

    return policy_function


def load_resources_from_config(config_path: Optional[Path] = None):
    """
    Load all resources (policy, reaction rules, building blocks) from config.

    This is useful when you want to set up your own planning pipeline
    using the same configuration as the benchmark.

    Args:
        config_path: Path to YAML config file. If None, uses default config.yaml

    Returns:
        dict with keys:
            - policy_function: Combined policy function
            - reaction_rules: Loaded reaction rules
            - building_blocks: Loaded building blocks set
            - tree_config: TreeConfig object
            - config: Raw config dictionary

    Example:
        >>> from scripts.sascore_bench.run_benchmark import load_resources_from_config
        >>> resources = load_resources_from_config()
        >>> tree = Tree(
        ...     target=my_mol,
        ...     config=resources["tree_config"],
        ...     reaction_rules=resources["reaction_rules"],
        ...     building_blocks=resources["building_blocks"],
        ...     expansion_function=resources["policy_function"],
        ... )
    """
    config = load_config(config_path)
    paths_cfg = config["paths"]
    policy_cfg = config["policy"]
    tree_cfg = config["tree"]

    data_folder = Path(paths_cfg["data_folder"]).resolve()
    ranking_policy_path = data_folder / paths_cfg["ranking_policy"]
    filtering_policy_path = data_folder / paths_cfg["filtering_policy"]
    reaction_rules_path = data_folder / paths_cfg["reaction_rules"]
    building_blocks_path = data_folder / paths_cfg["building_blocks"]

    # Verify paths exist
    for path in [
        ranking_policy_path,
        filtering_policy_path,
        reaction_rules_path,
        building_blocks_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load policy
    logger.info("Loading combined policy (filtering + ranking)...")
    policy_function = load_combined_policy_function(
        filtering_weights_path=str(filtering_policy_path),
        ranking_weights_path=str(ranking_policy_path),
        top_rules=policy_cfg["top_rules"],
        rule_prob_threshold=policy_cfg["rule_prob_threshold"],
    )

    # Load reaction rules
    logger.info("Loading reaction rules...")
    reaction_rules = load_reaction_rules(str(reaction_rules_path))

    # Load building blocks
    logger.info("Loading building blocks...")
    building_blocks = load_building_blocks(building_blocks_path, standardize=True)

    # Create tree config - build kwargs from config, using defaults for missing values
    tree_kwargs = {
        "algorithm": tree_cfg.get("algorithm", "nmcs"),
        "max_iterations": tree_cfg.get("max_iterations", 100),
        "max_time": tree_cfg.get("max_time", 120),
        "max_depth": tree_cfg.get("max_depth", 6),
        "max_tree_size": tree_cfg.get("max_tree_size", 1000000),
        "exclude_small": tree_cfg.get("exclude_small", True),
        "min_mol_size": tree_cfg.get("min_mol_size", 6),
        "silent": tree_cfg.get("silent", True),
    }
    
    # Add NMCS-specific parameters if present
    if "nmcs_level" in tree_cfg:
        tree_kwargs["nmcs_level"] = tree_cfg["nmcs_level"]
    if "nmcs_playout_mode" in tree_cfg:
        tree_kwargs["nmcs_playout_mode"] = tree_cfg["nmcs_playout_mode"]
    if "lnmcs_ratio" in tree_cfg:
        tree_kwargs["lnmcs_ratio"] = tree_cfg["lnmcs_ratio"]
    
    # Add UCT-specific parameters if present (for backwards compatibility)
    if "search_strategy" in tree_cfg:
        tree_kwargs["search_strategy"] = tree_cfg["search_strategy"]
    if "ucb_type" in tree_cfg:
        tree_kwargs["ucb_type"] = tree_cfg["ucb_type"]
    if "c_ucb" in tree_cfg:
        tree_kwargs["c_ucb"] = tree_cfg["c_ucb"]
    if "backprop_type" in tree_cfg:
        tree_kwargs["backprop_type"] = tree_cfg["backprop_type"]
    if "evaluation_agg" in tree_cfg:
        tree_kwargs["evaluation_agg"] = tree_cfg["evaluation_agg"]
    if "init_node_value" in tree_cfg:
        tree_kwargs["init_node_value"] = tree_cfg["init_node_value"]
    if "epsilon" in tree_cfg:
        tree_kwargs["epsilon"] = tree_cfg["epsilon"]
    if "enable_pruning" in tree_cfg:
        tree_kwargs["enable_pruning"] = tree_cfg["enable_pruning"]
    
    tree_config = TreeConfig(**tree_kwargs)

    return {
        "policy_function": policy_function,
        "reaction_rules": reaction_rules,
        "building_blocks": building_blocks,
        "tree_config": tree_config,
        "config": config,
    }


def extract_tree_stats(tree: Tree, target_smi: str) -> dict:
    """Extract statistics from a completed tree search."""
    return {
        "target_smiles": target_smi,
        "num_routes": len(tree.winning_nodes),
        "num_nodes": len(tree),
        "num_iter": tree.curr_iteration,
        "tree_depth": max(tree.nodes_depth.values()),
        "search_time": round(tree.curr_time, 1),
        "solved": len(tree.winning_nodes) > 0,
    }


def run_benchmark(
    targets_path: Path,
    policy_function,
    evaluation_function,
    tree_config: TreeConfig,
    reaction_rules,
    building_blocks,
    results_dir: Path,
) -> dict:
    """Run benchmark on a single targets file."""

    results_dir.mkdir(parents=True, exist_ok=True)
    stats_file = results_dir / "stats.csv"

    stats_header = [
        "target_smiles",
        "num_routes",
        "num_nodes",
        "num_iter",
        "tree_depth",
        "search_time",
        "solved",
        "error",
    ]

    n_total = 0
    n_solved = 0
    n_errors = 0

    with open(targets_path, "r", encoding="utf-8") as targets_file:
        targets = [line.strip() for line in targets_file if line.strip()]

    with open(stats_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stats_header)
        writer.writeheader()

        for target_smi in tqdm(targets, desc=f"Processing {targets_path.name}"):
            n_total += 1
            row = {"target_smiles": target_smi, "error": ""}

            try:
                target_mol = mol_from_smiles(target_smi, standardize=True)
                if target_mol is None:
                    row["error"] = "Failed to parse SMILES"
                    row["solved"] = False
                    writer.writerow(row)
                    n_errors += 1
                    continue

                tree = Tree(
                    target=target_mol,
                    config=tree_config,
                    reaction_rules=reaction_rules,
                    building_blocks=building_blocks,
                    expansion_function=policy_function,
                    evaluation_function=evaluation_function,
                )

                # Run tree search
                for solved, node_id in tree:
                    pass

                # Extract stats
                stats = extract_tree_stats(tree, target_smi)
                row.update(stats)

                if stats["solved"]:
                    n_solved += 1

            except Exception as e:
                row["error"] = str(e)
                row["solved"] = False
                n_errors += 1
                logger.warning(f"Error processing {target_smi}: {e}")

            writer.writerow(row)
            csvfile.flush()

    return {
        "file": targets_path.name,
        "total": n_total,
        "solved": n_solved,
        "errors": n_errors,
        "solve_rate": n_solved / n_total if n_total > 0 else 0,
    }


def main():
    """Main entry point for the benchmark script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run SAScore benchmark with combined policy"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help=f"Path to config YAML file (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    paths_cfg = config["paths"]
    benchmark_cfg = config["benchmark"]
    policy_cfg = config["policy"]
    tree_cfg = config["tree"]
    eval_cfg = config["evaluation"]

    # Resolve paths
    data_folder = Path(paths_cfg["data_folder"]).resolve()
    benchmark_folder = data_folder / paths_cfg["benchmark_folder"]
    results_root = Path(paths_cfg["results_folder"]).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = results_root / timestamp

    # Data paths
    ranking_policy_path = data_folder / paths_cfg["ranking_policy"]
    filtering_policy_path = data_folder / paths_cfg["filtering_policy"]
    reaction_rules_path = data_folder / paths_cfg["reaction_rules"]
    building_blocks_path = data_folder / paths_cfg["building_blocks"]

    # Ensure all required data is downloaded from Hugging Face if missing
    logger.info("Checking for required data files...")
    ensure_data_downloaded(data_folder, benchmark_folder)

    # Verify paths exist (should exist after download)
    for path in [
        ranking_policy_path,
        filtering_policy_path,
        reaction_rules_path,
        building_blocks_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    logger.info("Loading resources...")

    # Load combined policy
    logger.info("Loading combined policy (filtering + ranking)...")
    policy_function = load_combined_policy_function(
        filtering_weights_path=str(filtering_policy_path),
        ranking_weights_path=str(ranking_policy_path),
        top_rules=policy_cfg["top_rules"],
        rule_prob_threshold=policy_cfg["rule_prob_threshold"],
    )

    # Load reaction rules and building blocks
    logger.info("Loading reaction rules...")
    reaction_rules = load_reaction_rules(str(reaction_rules_path))

    logger.info("Loading building blocks...")
    building_blocks = load_building_blocks(building_blocks_path, standardize=True)

    # Tree configuration - build kwargs from config, using defaults for missing values
    tree_kwargs = {
        "algorithm": tree_cfg.get("algorithm", "nmcs"),
        "max_iterations": tree_cfg.get("max_iterations", 100),
        "max_time": tree_cfg.get("max_time", 120),
        "max_depth": tree_cfg.get("max_depth", 6),
        "max_tree_size": tree_cfg.get("max_tree_size", 1000000),
        "exclude_small": tree_cfg.get("exclude_small", True),
        "min_mol_size": tree_cfg.get("min_mol_size", 6),
        "silent": tree_cfg.get("silent", True),
    }
    
    # Add NMCS-specific parameters if present
    if "nmcs_level" in tree_cfg:
        tree_kwargs["nmcs_level"] = tree_cfg["nmcs_level"]
    if "nmcs_playout_mode" in tree_cfg:
        tree_kwargs["nmcs_playout_mode"] = tree_cfg["nmcs_playout_mode"]
    if "lnmcs_ratio" in tree_cfg:
        tree_kwargs["lnmcs_ratio"] = tree_cfg["lnmcs_ratio"]
    
    # Add UCT-specific parameters if present (for backwards compatibility)
    if "search_strategy" in tree_cfg:
        tree_kwargs["search_strategy"] = tree_cfg["search_strategy"]
    if "ucb_type" in tree_cfg:
        tree_kwargs["ucb_type"] = tree_cfg["ucb_type"]
    if "c_ucb" in tree_cfg:
        tree_kwargs["c_ucb"] = tree_cfg["c_ucb"]
    if "backprop_type" in tree_cfg:
        tree_kwargs["backprop_type"] = tree_cfg["backprop_type"]
    if "evaluation_agg" in tree_cfg:
        tree_kwargs["evaluation_agg"] = tree_cfg["evaluation_agg"]
    if "init_node_value" in tree_cfg:
        tree_kwargs["init_node_value"] = tree_cfg["init_node_value"]
    if "epsilon" in tree_cfg:
        tree_kwargs["epsilon"] = tree_cfg["epsilon"]
    if "enable_pruning" in tree_cfg:
        tree_kwargs["enable_pruning"] = tree_cfg["enable_pruning"]
    
    tree_config = TreeConfig(**tree_kwargs)

    # Evaluation configuration (rollout)
    eval_config = RolloutEvaluationConfig(
        policy_network=policy_function,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        min_mol_size=tree_config.min_mol_size,
        max_depth=tree_config.max_depth,
        normalize=eval_cfg["normalize"],
    )
    evaluation_function = load_evaluation_function(eval_config)

    # Prepare actual config for saving
    actual_config = {
        "tree_config": tree_config.to_dict(),
        "eval_config": {
            "min_mol_size": eval_config.min_mol_size,
            "max_depth": eval_config.max_depth,
            "normalize": eval_config.normalize,
        },
        "policy_config": {
            "top_rules": policy_cfg["top_rules"],
            "rule_prob_threshold": policy_cfg["rule_prob_threshold"],
        },
    }

    # Print actual config
    print("\n" + "=" * 60)
    print("ACTUAL CONFIGURATION")
    print("=" * 60)
    print(json.dumps(actual_config, indent=2))
    print("=" * 60 + "\n")

    # Save config to benchmark folder
    results_root.mkdir(parents=True, exist_ok=True)
    config_file = results_root / "config.json"
    with open(config_file, "w") as f:
        json.dump(actual_config, f, indent=2)
    logger.info(f"Config saved to: {config_file}")

    # Find all benchmark files
    benchmark_files = sorted(benchmark_folder.glob(benchmark_cfg["target_pattern"]))

    # Apply slice if specified
    slice_start = benchmark_cfg.get("file_slice_start")
    slice_end = benchmark_cfg.get("file_slice_end")
    if slice_start is not None or slice_end is not None:
        benchmark_files = benchmark_files[slice_start:slice_end]

    if not benchmark_files:
        raise FileNotFoundError(f"No benchmark files found in {benchmark_folder}")

    logger.info(f"Found {len(benchmark_files)} benchmark files")
    logger.info(
        f"Configuration: algorithm={tree_config.algorithm}, "
        f"max_time={tree_config.max_time}s, "
        f"max_depth={tree_config.max_depth}, "
        f"max_iterations={tree_config.max_iterations}"
    )
    logger.info(f"Results will be saved to: {results_root}")

    # Run benchmarks
    all_results = []
    for benchmark_file in benchmark_files:
        logger.info(f"\nProcessing: {benchmark_file.name}")

        # Create results directory for this benchmark
        sascore_range = benchmark_file.stem.replace("targets_with_sascore_", "")
        benchmark_results_dir = results_root / f"sascore_{sascore_range}"

        result = run_benchmark(
            targets_path=benchmark_file,
            policy_function=policy_function,
            evaluation_function=evaluation_function,
            tree_config=tree_config,
            reaction_rules=reaction_rules,
            building_blocks=building_blocks,
            results_dir=benchmark_results_dir,
        )
        all_results.append(result)

        logger.info(
            f"  Completed: {result['solved']}/{result['total']} solved "
            f"({result['solve_rate']:.1%}), {result['errors']} errors"
        )

    # Save summary
    summary_file = results_root / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "config": actual_config,
                "results": all_results,
            },
            f,
            indent=2,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Algorithm: {tree_config.algorithm}")
    print(f"Policy: Combined (filtering + ranking)")
    print(
        f"Config: max_time={tree_config.max_time}s, "
        f"max_depth={tree_config.max_depth}, "
        f"max_iterations={tree_config.max_iterations}"
    )
    print("-" * 60)
    print(f"{'SAScore Range':<20} {'Solved':<10} {'Total':<10} {'Rate':<10}")
    print("-" * 60)

    total_solved = 0
    total_molecules = 0
    for result in all_results:
        sascore_range = (
            result["file"].replace("targets_with_sascore_", "").replace(".smi", "")
        )
        print(
            f"{sascore_range:<20} {result['solved']:<10} {result['total']:<10} {result['solve_rate']:.1%}"
        )
        total_solved += result["solved"]
        total_molecules += result["total"]

    print("-" * 60)
    print(
        f"{'TOTAL':<20} {total_solved:<10} {total_molecules:<10} {total_solved/total_molecules:.1%}"
    )
    print("=" * 60)
    print(f"\nResults saved to: {results_root}")
    print("\nTo plot results, run:")
    print("  uv run sascore-plot")


if __name__ == "__main__":
    main()
