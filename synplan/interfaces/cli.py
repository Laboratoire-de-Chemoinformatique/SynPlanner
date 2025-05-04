"""Module containing commands line scripts for training and planning steps."""

import os
import warnings
from pathlib import Path

import click
import yaml

from synplan.chem.data.filtering import ReactionFilterConfig, filter_reactions_from_file
from synplan.chem.data.standardizing import (
    ReactionStandardizationConfig,
    standardize_reactions_from_file,
)
from synplan.chem.reaction_rules.extraction import extract_rules_from_reactions
from synplan.chem.utils import standardize_building_blocks
from synplan.mcts.search import run_search
from synplan.ml.training.supervised import create_policy_dataset, run_policy_training
from synplan.ml.training.reinforcement import run_updating
from synplan.utils.config import (
    PolicyNetworkConfig,
    RuleExtractionConfig,
    TreeConfig,
    TuningConfig,
    ValueNetworkConfig,
)
from synplan.utils.loading import download_all_data

warnings.filterwarnings("ignore")


@click.group(name="synplan")
def synplan():
    """SynPlanner command line interface."""


@synplan.command(name="download_all_data")
@click.option(
    "--save_to",
    "save_to",
    help="Path to the folder where downloaded data will be stored.",
)
def download_all_data_cli(save_to: str = ".") -> None:
    """Downloads all data for training, planning and benchmarking SynPlanner."""
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
    """Standardizes building blocks."""
    standardize_building_blocks(input_file=input_file, output_file=output_file)


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
def reaction_standardizing_cli(
    config_path: str, input_file: str, output_file: str, num_cpus: int
) -> None:
    """Standardizes reactions and remove duplicates."""
    stand_config = ReactionStandardizationConfig.from_yaml(config_path)
    standardize_reactions_from_file(
        config=stand_config,
        input_reaction_data_path=input_file,
        standardized_reaction_data_path=output_file,
        num_cpus=num_cpus,
        batch_size=100,
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
def reaction_filtering_cli(
    config_path: str, input_file: str, output_file: str, num_cpus: int
):
    """Filters erroneous reactions."""
    reaction_check_config = ReactionFilterConfig().from_yaml(config_path)
    filter_reactions_from_file(
        config=reaction_check_config,
        input_reaction_data_path=input_file,
        filtered_reaction_data_path=output_file,
        num_cpus=num_cpus,
        batch_size=100,
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
def rule_extracting_cli(
    config_path: str, input_file: str, output_file: str, num_cpus: int
):
    """Reaction rules extraction."""
    reaction_rule_config = RuleExtractionConfig.from_yaml(config_path)
    extract_rules_from_reactions(
        config=reaction_rule_config,
        reaction_data_path=input_file,
        reaction_rules_path=output_file,
        num_cpus=num_cpus,
        batch_size=100,
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
    "--reaction_data",
    required=True,
    type=click.Path(exists=True),
    help="Path to the file with reactions for ranking policy training.",
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
    default=4,
    type=int,
    help="The number of CPUs to use for training set preparation.",
)
def ranking_policy_training_cli(
    config_path: str,
    reaction_data: str,
    reaction_rules: str,
    results_dir: str,
    num_cpus: int,
) -> None:
    """Ranking policy network training."""
    policy_config = PolicyNetworkConfig.from_yaml(config_path)
    policy_config.policy_type = "ranking"
    policy_dataset_file = os.path.join(results_dir, "policy_dataset.dt")

    datamodule = create_policy_dataset(
        reaction_rules_path=reaction_rules,
        molecules_or_reactions_path=reaction_data,
        output_path=policy_dataset_file,
        dataset_type="ranking",
        batch_size=policy_config.batch_size,
        num_cpus=num_cpus,
    )

    run_policy_training(datamodule, config=policy_config, results_path=results_dir)


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
def filtering_policy_training_cli(
    config_path: str,
    molecule_data: str,
    reaction_rules: str,
    results_dir: str,
    num_cpus: int,
):
    """Filtering policy network training."""

    policy_config = PolicyNetworkConfig.from_yaml(config_path)
    policy_config.policy_type = "filtering"
    policy_dataset_file = os.path.join(results_dir, "policy_dataset.ckpt")

    datamodule = create_policy_dataset(
        reaction_rules_path=reaction_rules,
        molecules_or_reactions_path=molecule_data,
        output_path=policy_dataset_file,
        dataset_type="filtering",
        batch_size=policy_config.batch_size,
        num_cpus=num_cpus,
    )

    run_policy_training(datamodule, config=policy_config, results_path=results_dir)


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

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    policy_config = PolicyNetworkConfig.from_dict(config["node_expansion"])
    policy_config.weights_path = policy_network

    value_config = ValueNetworkConfig.from_dict(config["value_network"])
    if value_network is None:
        value_config.weights_path = os.path.join(
            results_dir, "weights", "value_network.ckpt"
        )

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
def planning_cli(
    config_path: str,
    targets: str,
    reaction_rules: str,
    building_blocks: str,
    policy_network: str,
    value_network: str,
    results_dir: str,
):
    """Retrosynthetic planning."""

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    search_config = {**config["tree"], **config["node_evaluation"]}
    policy_config = PolicyNetworkConfig.from_dict(
        {**config["node_expansion"], **{"weights_path": policy_network}}
    )

    run_search(
        targets_path=targets,
        search_config=search_config,
        policy_config=policy_config,
        reaction_rules_path=reaction_rules,
        building_blocks_path=building_blocks,
        value_network_path=value_network,
        results_root=results_dir,
    )


if __name__ == "__main__":
    synplan()
