"""Module containing commands line scripts for training and planning steps."""

import warnings
from pathlib import Path

import click
import yaml

from synplan.chem.building_blocks import load_building_block_indexes
from synplan.chem.reaction.curation.filtering import (
    ReactionFilterConfig,
    filter_reactions_from_file,
)
from synplan.chem.reaction.curation.mapping import (
    MappingConfig,
    map_reactions_from_file,
)
from synplan.chem.reaction.curation.standardizing import (
    ReactionStandardizationConfig,
    standardize_reactions_from_file,
)
from synplan.chem.reaction.routes.cli import run_cluster_cli
from synplan.chem.reaction.rules.config import RuleExtractionConfig
from synplan.chem.reaction.rules.extraction import extract_rules_from_reactions
from synplan.chem.synthon.config import SynthonConfig
from synplan.chem.utils import standardize_building_blocks
from synplan.interfaces.synthon_commands import (
    classify_file,
    coverage_file,
    enumerate_file,
    fragment_file,
    scaffolds_file,
    synthonise_file,
)
from synplan.mcts.config import (
    CombinedPolicyConfig,
    PolicyEvaluationConfig,
    RandomEvaluationConfig,
    RDKitEvaluationConfig,
    RolloutEvaluationConfig,
    TreeConfig,
    ValueNetworkEvaluationConfig,
)
from synplan.mcts.search import run_search
from synplan.ml.config import (
    PolicyNetworkConfig,
    TuningConfig,
    ValueNetworkConfig,
)
from synplan.ml.training.reinforcement import run_updating
from synplan.ml.training.supervised import (
    create_policy_dataset,
    run_mhn_network_tuning,
    run_policy_training,
)
from synplan.utils.loading import (
    download_all_data,
    download_preset,
    load_building_blocks,
    load_policy_function,
    load_reaction_rules,
)

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version
except Exception:  # pragma: no cover
    _dist_version = None  # type: ignore[assignment]
    PackageNotFoundError = Exception  # type: ignore[assignment]


def _resolve_cli_version() -> str:
    # Prefer installed distribution version
    if _dist_version is not None:
        try:
            return _dist_version("SynPlanner")
        except PackageNotFoundError:
            pass
    # Fallback to package attribute in editable/dev mode
    try:
        from synplan import __version__ as _pkg_version

        return _pkg_version
    except Exception:
        return "0.0.0+unknown"


warnings.filterwarnings("ignore")


@click.group(name="synplan")
@click.version_option(version=_resolve_cli_version(), prog_name="synplan")
def synplan():
    """SynPlanner command line interface."""


@synplan.command(name="download_preset")
@click.option(
    "--preset",
    default="synplanner-gps",
    help="Preset name (e.g. 'synplanner-gps').",
)
@click.option(
    "--save_to", "save_to", default=".", help="Directory to save downloaded data."
)
def download_preset_cli(preset: str, save_to: str) -> None:
    """Download a ready-to-use data preset from HuggingFace."""
    paths = download_preset(preset_name=preset, save_to=save_to)
    click.echo(f"Preset '{preset}' downloaded:")
    for key, path in paths.items():
        if path is not None:
            click.echo(f"  {key}: {path}")


@synplan.command(name="download_all_data")
@click.option(
    "--save_to",
    "save_to",
    help="Path to the folder where downloaded data will be stored.",
)
def download_all_data_cli(save_to: str = ".") -> None:
    """Downloads all data from the legacy repo. Deprecated: use download_preset instead."""
    download_all_data(save_to=save_to)


@synplan.command(name="building_blocks_standardizing")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with building blocks to be standardized.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(),
    help="Path to the file where standardized building blocks will be stored.",
)
def building_blocks_standardizing_cli(input_file: str, output_file: str) -> None:
    """Standardize molecular files or prepare a vendor-aware JSON catalogue."""
    standardize_building_blocks(input_file=input_file, output_file=output_file)


@synplan.command(name="ord_convert")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the ORD .pb Dataset file.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(),
    help="Path to the output .smi file.",
)
def ord_convert_cli(input_file: str, output_file: str) -> None:
    """Convert an ORD .pb Dataset file to a SynPlanner-compatible SMILES file."""
    from synplan.utils.ord.reader import convert_ord_to_smiles

    n = convert_ord_to_smiles(input_path=input_file, output_path=output_file)
    click.echo(f"Converted {n} reactions → {output_file}")


@synplan.command(name="reaction_standardizing")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the configuration file for reactions standardizing.",
)
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with reactions to be standardized.",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(),
    help="Path to the file where standardized reactions will be stored.",
)
@click.option(
    "--num_cpus", default=4, type=int, help="The number of CPUs to use for processing."
)
@click.option(
    "--ignore-errors/--no-ignore-errors",
    default=True,
    help="Skip bad reactions instead of crashing (default: skip).",
)
@click.option(
    "--error-file",
    "error_file",
    default=None,
    type=click.Path(),
    help="Write failed reactions here. Default: <output>.errors.tsv",
)
@click.option(
    "--batch_size",
    default=100,
    type=int,
    help="Number of reactions per batch sent to each worker.",
)
@click.option(
    "--silent/--no-silent",
    default=False,
    help="Suppress the progress bar (default: show progress).",
)
def reaction_standardizing_cli(
    config_path: str,
    input_file: str,
    output_file: str,
    num_cpus: int,
    ignore_errors: bool,
    error_file: str | None,
    batch_size: int,
    silent: bool,
) -> None:
    """Standardizes reactions and remove duplicates."""
    stand_config = ReactionStandardizationConfig.from_yaml(config_path)
    standardize_reactions_from_file(
        config=stand_config,
        input_reaction_data_path=input_file,
        standardized_reaction_data_path=output_file,
        num_cpus=num_cpus,
        batch_size=batch_size,
        silent=silent,
        ignore_errors=ignore_errors,
        error_file_path=error_file,
    )


@synplan.command(name="reaction_filtering")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the configuration file for reactions filtering.",
)
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with reactions to be filtered.",
)
@click.option(
    "--output",
    "output_file",
    default=Path("./"),
    type=click.Path(),
    help="Path to the file where successfully filtered reactions will be stored.",
)
@click.option(
    "--num_cpus", default=4, type=int, help="The number of CPUs to use for processing."
)
@click.option(
    "--ignore-errors/--no-ignore-errors",
    default=True,
    help="Skip bad reactions instead of crashing (default: skip).",
)
@click.option(
    "--error-file",
    "error_file",
    default=None,
    type=click.Path(),
    help="Write failed/filtered reactions here. Default: <output>.errors.tsv",
)
@click.option(
    "--batch_size",
    default=100,
    type=int,
    help="Number of reactions per batch sent to each worker.",
)
def reaction_filtering_cli(
    config_path: str,
    input_file: str,
    output_file: str,
    num_cpus: int,
    ignore_errors: bool,
    error_file: str | None,
    batch_size: int,
):
    """Filters erroneous reactions."""
    reaction_check_config = ReactionFilterConfig().from_yaml(config_path)
    filter_reactions_from_file(
        config=reaction_check_config,
        input_reaction_data_path=input_file,
        filtered_reaction_data_path=output_file,
        num_cpus=num_cpus,
        batch_size=batch_size,
        ignore_errors=ignore_errors,
        error_file_path=error_file,
    )


@synplan.command(name="reaction_mapping")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="YAML configuration file (optional; defaults used if omitted).",
)
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with reactions to be mapped.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(),
    help="Path to the file where mapped reactions will be stored.",
)
@click.option(
    "--workers", "num_workers", default=0, type=int, help="CPU workers (0 = auto)."
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cuda", "mps", "cpu"], case_sensitive=False),
    help="Torch device (default: auto-detect).",
)
@click.option(
    "--no-amp", "no_amp", is_flag=True, help="Disable automatic mixed precision."
)
@click.option(
    "--batch-size", "batch_size", default=None, type=int, help="GPU batch size."
)
@click.option(
    "--ignore-errors/--no-ignore-errors",
    default=True,
    help="Skip bad reactions instead of crashing (default: skip).",
)
@click.option(
    "--error-file",
    "error_file",
    default=None,
    type=click.Path(),
    help="Write failed reactions here. Default: <output>.errors.tsv",
)
def reaction_mapping_cli(
    config_path,
    input_file,
    output_file,
    num_workers,
    device,
    no_amp,
    batch_size,
    ignore_errors,
    error_file,
):
    """Map reaction atoms using a neural attention model."""
    config = MappingConfig.from_yaml(config_path) if config_path else MappingConfig()
    if device is not None:
        config.device = device
    if no_amp:
        config.no_amp = True
    if batch_size is not None:
        config.batch_size = batch_size

    map_reactions_from_file(
        config=config,
        input_reaction_data_path=input_file,
        mapped_reaction_data_path=output_file,
        num_workers=num_workers,
        silent=False,
        ignore_errors=ignore_errors,
        error_file_path=error_file,
    )


@synplan.command(name="rule_extracting")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the configuration file for reaction rules extracting.",
)
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with reactions for reaction rules extraction.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(),
    help="Path to the file where extracted reaction rules will be stored.",
)
@click.option(
    "--num_cpus", default=4, type=int, help="The number of CPUs to use for processing."
)
@click.option(
    "--ignore-errors/--no-ignore-errors",
    default=True,
    help="Skip bad reactions instead of crashing (default: skip).",
)
@click.option(
    "--error-file",
    "error_file",
    default=None,
    type=click.Path(),
    help="Write failed reactions here. Default: <output>.errors.tsv",
)
@click.option(
    "--batch_size",
    default=100,
    type=int,
    help="Number of reactions per batch sent to each worker.",
)
def rule_extracting_cli(
    config_path: str,
    input_file: str,
    output_file: str,
    num_cpus: int,
    ignore_errors: bool,
    error_file: str | None,
    batch_size: int,
):
    """Reaction rules extraction."""
    reaction_rule_config = RuleExtractionConfig.from_yaml(config_path)
    extract_rules_from_reactions(
        config=reaction_rule_config,
        reaction_data_path=input_file,
        reaction_rules_path=output_file,
        num_cpus=num_cpus,
        batch_size=batch_size,
        ignore_errors=ignore_errors,
        error_file_path=error_file,
    )


@synplan.command(name="ranking_policy_training")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the configuration file for ranking policy training.",
)
@click.option(
    "--policy_data",
    required=True,
    type=click.Path(exists=True),
    help="Path to the policy training mapping file (*_policy_data.tsv) "
    "generated during rule extraction.",
)
@click.option(
    "--results_dir",
    default=Path("."),
    type=click.Path(),
    help="Path to the directory where the trained policy network will be stored.",
)
@click.option(
    "--workers",
    "num_workers",
    default=0,
    type=int,
    help="CPU workers for dataset preprocessing (0 = auto-detect).",
)
@click.option(
    "--no-cache",
    "no_cache",
    is_flag=True,
    default=False,
    help="Disable dataset caching (always reprocess from scratch).",
)
@click.option(
    "--logger",
    "logger_type",
    default=None,
    type=click.Choice(
        ["csv", "tensorboard", "mlflow", "wandb", "litlogger"],
        case_sensitive=False,
    ),
    help="Enable a training logger: csv, tensorboard, mlflow, wandb, or litlogger.",
)
def ranking_policy_training_cli(
    config_path: str,
    policy_data: str,
    results_dir: str,
    num_workers: int,
    no_cache: bool,
    logger_type: str | None,
) -> None:
    """Ranking policy network training."""
    policy_config = PolicyNetworkConfig.from_yaml(config_path)
    policy_config.policy_type = "ranking"
    if logger_type is not None:
        policy_config.logger = {"type": logger_type}

    datamodule = create_policy_dataset(
        policy_data_path=policy_data,
        results_dir=results_dir,
        dataset_type="ranking",
        batch_size=policy_config.batch_size,
        num_workers=num_workers,
        cache=not no_cache,
    )

    run_policy_training(datamodule, config=policy_config, results_path=results_dir)


@synplan.command(name="mhn_network_tuning")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the configuration file for MHN policy fine-tuning.",
)
@click.option(
    "--policy_network",
    "policy_network",
    required=True,
    type=click.Path(exists=True),
    help="Path to an already trained MHN ranking policy checkpoint.",
)
@click.option(
    "--new_policy_data",
    "new_policy_data",
    required=True,
    type=click.Path(exists=True),
    help="Path to the new ranking policy mapping file (*_policy_data.tsv).",
)
@click.option(
    "--results_dir",
    default=Path("."),
    type=click.Path(),
    help="Path to the directory where the tuned MHN checkpoint will be stored.",
)
@click.option(
    "--workers",
    "num_workers",
    default=0,
    type=int,
    help="CPU workers for dataset preprocessing (0 = auto-detect).",
)
@click.option(
    "--no-cache",
    "no_cache",
    is_flag=True,
    default=False,
    help="Disable dataset caching (always reprocess from scratch).",
)
def mhn_network_tuning_cli(
    config_path: str,
    policy_network: str,
    new_policy_data: str,
    results_dir: str,
    num_workers: int,
    no_cache: bool,
) -> None:
    """Fine-tune an existing MHN ranking checkpoint on new policy data."""
    policy_config = PolicyNetworkConfig.from_yaml(config_path)

    run_mhn_network_tuning(
        policy_network_path=policy_network,
        new_policy_data_path=new_policy_data,
        results_path=results_dir,
        config=policy_config,
        num_workers=num_workers,
        cache=not no_cache,
    )


@synplan.command(name="filtering_policy_training")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the configuration file for filtering policy training.",
)
@click.option(
    "--molecule_data",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with molecules for filtering policy training.",
)
@click.option(
    "--reaction_rules",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with extracted reaction rules.",
)
@click.option(
    "--results_dir",
    default=Path("."),
    type=click.Path(),
    help="Path to the directory where the trained policy network will be stored.",
)
@click.option(
    "--num_cpus",
    default=8,
    type=int,
    help="The number of CPUs to use for training set preparation.",
)
@click.option(
    "--no-cache",
    "no_cache",
    is_flag=True,
    default=False,
    help="Disable dataset caching (always reprocess from scratch).",
)
@click.option(
    "--logger",
    "logger_type",
    default=None,
    type=click.Choice(
        ["csv", "tensorboard", "mlflow", "wandb", "litlogger"],
        case_sensitive=False,
    ),
    help="Enable a training logger: csv, tensorboard, mlflow, wandb, or litlogger.",
)
def filtering_policy_training_cli(
    config_path: str,
    molecule_data: str,
    reaction_rules: str,
    results_dir: str,
    num_cpus: int,
    no_cache: bool,
    logger_type: str | None,
):
    """Filtering policy network training."""

    policy_config = PolicyNetworkConfig.from_yaml(config_path)
    policy_config.policy_type = "filtering"
    if policy_config.architecture == "mhn_ranking":
        raise click.UsageError(
            "architecture=mhn_ranking is only supported by ranking_policy_training"
        )
    if logger_type is not None:
        policy_config.logger = {"type": logger_type}

    datamodule = create_policy_dataset(
        reaction_rules_path=reaction_rules,
        molecules_or_reactions_path=molecule_data,
        results_dir=results_dir,
        dataset_type="filtering",
        batch_size=policy_config.batch_size,
        num_cpus=num_cpus,
        cache=not no_cache,
    )

    run_policy_training(
        datamodule,
        config=policy_config,
        results_path=results_dir,
    )


@synplan.command(name="value_network_tuning")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the configuration file for value network training.",
)
@click.option(
    "--targets",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with target molecules for planning simulations.",
)
@click.option(
    "--reaction_rules",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with extracted reaction rules. Needed for planning simulations.",
)
@click.option(
    "--building_blocks",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with building blocks. Needed for planning simulations.",
)
@click.option(
    "--policy_network",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with trained policy network. Needed for planning simulations.",
)
@click.option(
    "--value_network",
    default=None,
    type=click.Path(exists=True),
    help="Path to the file with trained value network. Needed in case of additional value network fine-tuning",
)
@click.option(
    "--results_dir",
    default=".",
    type=click.Path(exists=False),
    help="Path to the directory where the trained value network will be stored.",
)
def value_network_tuning_cli(
    config_path: str,
    targets: str,
    reaction_rules: str,
    building_blocks: str,
    policy_network: str,
    value_network: str,
    results_dir: str,
):
    """Value network tuning."""

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    policy_config = PolicyNetworkConfig.from_dict(config["node_expansion"])
    policy_config.weights_path = policy_network

    value_config = ValueNetworkConfig.from_dict(config["value_network"])
    if value_network is not None:
        value_config.weights_path = value_network

    tree_config = TreeConfig.from_dict(config["tree"])
    tuning_config = TuningConfig.from_dict(config["tuning"])

    run_updating(
        targets_path=targets,
        tree_config=tree_config,
        policy_config=policy_config,
        value_config=value_config,
        reinforce_config=tuning_config,
        reaction_rules_path=reaction_rules,
        building_blocks_path=building_blocks,
        results_root=results_dir,
    )


@synplan.command(name="planning")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the configuration file for retrosynthetic planning.",
)
@click.option(
    "--targets",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with target molecules for retrosynthetic planning.",
)
@click.option(
    "--reaction_rules",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with extracted reaction rules.",
)
@click.option(
    "--building_blocks",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with building blocks.",
)
@click.option(
    "--policy_network",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with trained policy network.",
)
@click.option(
    "--value_network",
    default=None,
    type=click.Path(exists=True),
    help="Path to the file with trained value network.",
)
@click.option(
    "--results_dir",
    default=".",
    type=click.Path(exists=False),
    help="Path to the file where retrosynthetic planning results will be stored.",
)
@click.option(
    "--reconcile-mapping",
    "reconcile_mapping",
    is_flag=True,
    default=False,
    help=(
        "Reconcile atom-map numbering across steps in exported routes (slow). "
        "By default the export skips the route CGR composition and uses "
        "per-step-local atom numbering, which is ~4x faster."
    ),
)
@click.option(
    "--export_routes",
    "--export-routes",
    "export_routes",
    is_flag=True,
    default=False,
    help="Also emit a consumable target-keyed routes artifact (results.json.gz) "
    "plus manifest.json for downstream consumers.",
)
def planning_cli(
    config_path: str,
    targets: str,
    reaction_rules: str,
    building_blocks: str,
    policy_network: str,
    value_network: str,
    results_dir: str,
    reconcile_mapping: bool,
    export_routes: bool,
):
    """Retrosynthetic planning."""

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # node_evaluation keys are read separately below; merging them here would
    # reach TreeConfig (extra="forbid") and reject every config that has them.
    search_config = dict(config["tree"])
    if "combined_policy" in config:
        # both heads come from the config; --policy_network has nothing to point at
        policy_config = CombinedPolicyConfig.from_dict(config["combined_policy"])
    else:
        policy_config = PolicyNetworkConfig.from_dict(
            {**config["node_expansion"], **{"weights_path": policy_network}}
        )

    # Create evaluation config based on evaluation_type
    node_evaluation = config.get("node_evaluation", {})
    evaluation_type = node_evaluation.get("evaluation_type", "rollout")

    if evaluation_type == "gcn":
        # Value network evaluation
        if value_network is None:
            raise ValueError("value_network is required when evaluation_type is 'gcn'")
        evaluation_config = ValueNetworkEvaluationConfig(
            weights_path=value_network,
            normalize=node_evaluation.get("normalize", False),
        )
    elif evaluation_type == "rollout":
        # Rollout evaluation - need to load resources
        policy_function = load_policy_function(weights_path=policy_network)
        reaction_rules_list = load_reaction_rules(reaction_rules)
        if Path(building_blocks).suffix.lower() == ".json":
            building_blocks_set, building_block_candidates = (
                load_building_block_indexes(building_blocks)
            )
        else:
            building_blocks_set = load_building_blocks(
                building_blocks, standardize=False
            )
            building_block_candidates = None
        evaluation_config = RolloutEvaluationConfig(
            policy_network=policy_function,
            reaction_rules=reaction_rules_list,
            building_blocks=building_blocks_set,
            building_block_candidates=building_block_candidates,
            min_mol_size=search_config.get("min_mol_size", 6),
            max_depth=search_config.get("max_depth", 6),
            normalize=node_evaluation.get("normalize", False),
        )
    elif evaluation_type == "random":
        evaluation_config = RandomEvaluationConfig(
            normalize=node_evaluation.get("normalize", False),
        )
    elif evaluation_type == "policy":
        evaluation_config = PolicyEvaluationConfig(
            normalize=node_evaluation.get("normalize", False),
        )
    elif evaluation_type == "rdkit":
        evaluation_config = RDKitEvaluationConfig(
            score_function=node_evaluation.get("score_function", "sascore"),
            normalize=node_evaluation.get("normalize", False),
        )
    else:
        raise ValueError(
            f"Unknown evaluation_type: {evaluation_type}. "
            f"Expected one of: 'gcn', 'rollout', 'random', 'policy', 'rdkit'"
        )

    run_search(
        targets_path=targets,
        search_config=search_config,
        policy_config=policy_config,
        evaluation_config=evaluation_config,
        reaction_rules_path=reaction_rules,
        building_blocks_path=building_blocks,
        results_root=results_dir,
        reconcile_atom_mapping=reconcile_mapping,
        export_routes=export_routes,
    )


@synplan.command(name="clustering")
@click.option(
    "--targets",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with target molecules for retrosynthetic planning.",
)
@click.option(
    "--routes_file",
    default=".",
    type=click.Path(exists=False),
    help="Path to the file where the planning results are stored.",
)
@click.option(
    "--cluster_results_dir",
    default=".",
    type=click.Path(exists=False),
    help="Path to the file where clustering results will be stored.",
)
@click.option(
    "--perform_subcluster",
    is_flag=True,
    default=False,
    help="Perform subclustering.",
)
@click.option(
    "--subcluster_results_dir",
    default=".",
    type=click.Path(exists=False),
    help="Path to the file where subclustering results will be stored.",
)
def cluster_route_from_file_cli(
    targets: str,
    routes_file: str,
    cluster_results_dir: str,
    perform_subcluster: bool,
    subcluster_results_dir: str,
):
    """Clustering the routes from planning"""
    run_cluster_cli(
        routes_file=routes_file,
        cluster_results_dir=cluster_results_dir,
        perform_subcluster=perform_subcluster,
        subcluster_results_dir=subcluster_results_dir if perform_subcluster else None,
    )


def _synthon_config(config_path: str | None) -> SynthonConfig:
    return SynthonConfig.from_yaml(config_path) if config_path else SynthonConfig()


@synplan.command(name="bb_classifying")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with building blocks to be classified.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(),
    help="Path to the file where the classes will be stored.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to the synthonisation configuration file.",
)
def bb_classifying_cli(
    input_file: str, output_file: str, config_path: str | None
) -> None:
    """Assigns Synt-On building-block classes."""
    written = classify_file(input_file, output_file, _synthon_config(config_path))
    click.echo(f"{written} building blocks classified")


@synplan.command(name="bb_synthonizing")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with building blocks to be synthonised.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(),
    help="Path to the file where the synthon stock will be stored.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to the synthonisation configuration file.",
)
@click.option(
    "--keep-pg/--no-keep-pg",
    "keep_pg",
    default=None,
    help="Keep protected intermediates. Overrides the configuration file.",
)
def bb_synthonizing_cli(
    input_file: str, output_file: str, config_path: str | None, keep_pg: bool | None
) -> None:
    """Turns building blocks into synthons."""
    written, forced = synthonise_file(
        input_file, output_file, _synthon_config(config_path), keep_pg
    ) or (0, 0)
    click.echo(f"{written} synthons written")
    click.echo(f"{forced} building blocks had protecting-group retention forced on")


@synplan.command(name="synthon_fragment")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with target molecules.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(),
    help="Path to the file where the disconnection pathways will be stored.",
)
@click.option(
    "--stock",
    "stock_file",
    default=None,
    type=click.Path(exists=True),
    help="Path to the synthon stock file.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to the synthonisation configuration file.",
)
def synthon_fragment_cli(
    input_file: str, output_file: str, stock_file: str | None, config_path: str | None
) -> None:
    """Cuts targets into synthons."""
    written = fragment_file(
        input_file, output_file, stock_file, _synthon_config(config_path)
    )
    click.echo(f"{written} pathways written")


@synplan.command(name="synthon_enumerate")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to a synthon_fragment output file.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(),
    help="Path to the file where the enumerated library will be stored.",
)
@click.option(
    "--stock",
    "stock_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the synthon stock file.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to the synthonisation configuration file.",
)
def synthon_enumerate_cli(
    input_file: str, output_file: str, stock_file: str, config_path: str | None
) -> None:
    """Recombines stocked synthons into molecules."""
    written = enumerate_file(
        input_file, output_file, stock_file, _synthon_config(config_path)
    )
    click.echo(f"{written} molecules written")


@synplan.command(name="bb_scaffolds")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with building blocks.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(),
    help="Path to the file where the scaffolds will be stored.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to the synthonisation configuration file.",
)
def bb_scaffolds_cli(
    input_file: str, output_file: str, config_path: str | None
) -> None:
    """Bemis-Murcko scaffolds after protecting-group removal."""
    written = scaffolds_file(input_file, output_file, _synthon_config(config_path))
    click.echo(f"{written} scaffolds written")


@synplan.command(name="synthon_coverage")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to a file of atom-mapped reactions.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(),
    help="Path to the file where the kept reactions will be stored.",
)
@click.option(
    "--keep",
    type=click.Choice(["uncovered", "covered"]),
    default="uncovered",
    show_default=True,
    help="Which side to write: the reactions the synthon rules do NOT already provide "
    "(training corpus), or the ones they do (audit).",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to the synthonisation configuration file.",
)
def synthon_coverage_cli(
    input_file: str, output_file: str, keep: str, config_path: str | None
) -> None:
    """Filters reactions the shipped synthon disconnections already cover."""
    written, read = coverage_file(
        input_file, output_file, _synthon_config(config_path), keep=keep
    )
    click.echo(f"{written} of {read} reactions kept ({keep})")


if __name__ == "__main__":
    synplan()
