#!/usr/bin/env python3
"""Generate routes, compose RouteCGRs, and verify RouteCGR -> route JSON restore."""

from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from synplan.chem.reaction.routes.representation import (
    compose_route_cgr,
    prepare_route_cgr_reconstruction,
    route_json_from_route_cgrs,
    routes_dict_from_route_cgrs,
)
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.tree import Tree
from synplan.utils.config import RolloutEvaluationConfig, TreeConfig
from synplan.utils.loading import (
    download_preset,
    load_building_blocks,
    load_evaluation_function,
    load_policy_function,
    load_reaction_rules,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "synplan_data"
DEFAULT_TARGET_DIR = DEFAULT_DATA_DIR / "benchmarks/sascore/subset_100"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "route_cgr_roundtrip_results"


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preset", default="synplanner-gps")
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument(
        "--targets-per-bin",
        type=int,
        default=None,
        help="Take up to this many targets from each SAScore TSV before applying --max-targets.",
    )
    parser.add_argument(
        "--require-solved",
        action="store_true",
        help="Exit with failure when no target produced a solved route.",
    )
    parser.add_argument(
        "--required-solved-targets",
        type=int,
        default=None,
        help="Continue scanning until this many solved targets have verified routes, or --max-targets attempts are exhausted.",
    )
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--max-time", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-mol-size", type=int, default=1)
    parser.add_argument("--download-preset", action="store_true")
    return parser.parse_args()


def iter_targets(target_dir: Path, targets_per_bin: int | None = None):
    for path in sorted(target_dir.glob("*.tsv")):
        yielded_from_file = 0
        with path.open(encoding="utf-8", newline="") as f:
            for row_index, row in enumerate(csv.DictReader(f, delimiter="\t")):
                if targets_per_bin is not None and yielded_from_file >= targets_per_bin:
                    break
                smiles = row.get("smiles")
                if smiles:
                    yielded_from_file += 1
                    yield {
                        "target_file": str(path),
                        "target_row": row_index,
                        "smiles": smiles,
                        "id": row.get("id", f"{path.stem}:{row_index}"),
                        "sascore": row.get("sascore"),
                    }


def resolve_preset_paths(data_dir: Path, preset: str, *, download: bool) -> dict[str, Path]:
    if download:
        return download_preset(preset_name=preset, save_to=data_dir)

    preset_path = data_dir / "presets" / f"{preset}.yaml"
    if not preset_path.exists():
        return download_preset(preset_name=preset, save_to=data_dir)

    import yaml

    with preset_path.open(encoding="utf-8") as f:
        preset_data = yaml.safe_load(f)
    return {key: data_dir / repo_path for key, repo_path in preset_data["files"].items()}


def build_tree(target_smiles, reaction_rules, building_blocks, policy_function, config):
    target = mol_from_smiles(
        target_smiles, clean2d=True, standardize=True, clean_stereo=True
    )
    eval_config = RolloutEvaluationConfig(
        policy_network=policy_function,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        min_mol_size=config.min_mol_size,
        max_depth=config.max_depth,
        normalize=config.normalize_scores,
    )
    evaluator = load_evaluation_function(eval_config)
    tree = Tree(
        target=target,
        config=config,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        expansion_function=policy_function,
        evaluation_function=evaluator,
    )
    solved = False
    for step_solved, _node_id in tree:
        solved = solved or bool(step_solved)
    tree._log_final_stats("completed")
    return tree, solved


def canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def reaction_atom_maps(reaction):
    """Return side-wise atom-map numbers, ignoring molecule ordering."""

    return {
        "reactants": sorted(sorted(molecule._atoms) for molecule in reaction.reactants),
        "products": sorted(sorted(molecule._atoms) for molecule in reaction.products),
    }


def verify_tree_routes(tree: Tree) -> list[dict[str, Any]]:
    rows = []
    for route_id in sorted(set(tree.winning_nodes)):
        composed = compose_route_cgr(
            tree,
            route_id,
            preserve_transient_bonds=True,
        )
        if not composed:
            rows.append({"route_id": route_id, "status": "compose_failed"})
            continue

        route_cgr = prepare_route_cgr_reconstruction(
            composed["cgr"],
            composed["reactions_dict"],
            route_id,
            tree=tree,
        )
        restored_routes = routes_dict_from_route_cgrs({route_id: route_cgr})
        expected_json = {route_id: getattr(route_cgr, "route_json", None)}
        restored_json = route_json_from_route_cgrs({route_id: route_cgr})
        json_matches = canonical_json(expected_json) == canonical_json(restored_json)
        reaction_atom_maps_match = all(
            reaction_atom_maps(composed["reactions_dict"][step_id])
            == reaction_atom_maps(restored_routes[route_id][step_id])
            for step_id in composed["reactions_dict"]
        )
        matches = json_matches and reaction_atom_maps_match
        rows.append(
            {
                "route_id": route_id,
                "status": "ok" if matches else "mismatch",
                "n_steps": len(composed["reactions_dict"]),
                "schema": getattr(route_cgr, "route_reconstruction_schema", None),
                "json_matches": json_matches,
                "reaction_atom_maps_match": reaction_atom_maps_match,
                "expected_json": expected_json,
                "restored_json": restored_json,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_preset_paths(args.data_dir, args.preset, download=args.download_preset)
    reaction_rules = load_reaction_rules(paths["reaction_rules"])
    building_blocks = load_building_blocks(
        paths["building_blocks"], standardize=False, silent=True
    )
    policy_function = load_policy_function(weights_path=paths["ranking_policy"])
    config = TreeConfig(
        search_strategy="expansion_first",
        algorithm="UCT",
        enable_pruning=False,
        max_iterations=args.max_iterations,
        max_time=args.max_time,
        max_depth=args.max_depth,
        min_mol_size=args.min_mol_size,
        silent=True,
    )

    summary_rows = []
    checked_targets = 0
    solved_targets = 0
    for target in iter_targets(args.target_dir, targets_per_bin=args.targets_per_bin):
        if checked_targets >= args.max_targets:
            break
        if (
            args.required_solved_targets is not None
            and solved_targets >= args.required_solved_targets
        ):
            break
        checked_targets += 1
        print(
            f"[{checked_targets}/{args.max_targets}] "
            f"solved={solved_targets}/{args.required_solved_targets or 'any'} "
            f"{target['id']} {target['smiles']}",
            flush=True,
        )
        try:
            tree, solved = build_tree(
                target["smiles"], reaction_rules, building_blocks, policy_function, config
            )
            route_rows = verify_tree_routes(tree) if solved and tree.winning_nodes else []
            all_match = bool(route_rows) and all(row["status"] == "ok" for row in route_rows)
            if solved and all_match:
                solved_targets += 1
            summary_rows.append(
                {
                    **target,
                    "solved": solved,
                    "n_winning_routes": len(set(tree.winning_nodes)),
                    "n_checked_routes": len(route_rows),
                    "roundtrip_ok": all_match,
                }
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            route_out = args.output_dir / f"{checked_targets:03d}_{target['id']}_routes.json"
            route_out.write_text(
                json.dumps(route_rows, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            summary_rows.append(
                {
                    **target,
                    "solved": False,
                    "n_winning_routes": 0,
                    "n_checked_routes": 0,
                    "roundtrip_ok": False,
                    "error": repr(exc),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    fieldnames = sorted({key for row in summary_rows for key in row})
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    failures = [row for row in summary_rows if row.get("solved") and not row.get("roundtrip_ok")]
    solved_count = sum(bool(row.get("solved")) for row in summary_rows)
    print(f"Wrote {summary_path}")
    if args.require_solved and solved_count == 0:
        raise SystemExit("No solved routes were generated")
    if (
        args.required_solved_targets is not None
        and solved_count < args.required_solved_targets
    ):
        raise SystemExit(
            f"Only {solved_count}/{args.required_solved_targets} solved targets "
            f"were generated within {args.max_targets} attempts"
        )
    if failures:
        raise SystemExit(f"Round-trip failed for {len(failures)} solved target(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
