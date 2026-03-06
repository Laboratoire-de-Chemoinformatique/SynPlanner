"""Dagster ops for SynPlanner model training tasks.

Wraps policy network training (ranking/filtering), rule extraction,
and value network RL tuning.
"""

from pathlib import Path

import yaml
from dagster import In, MetadataValue, OpExecutionContext, Out, op

from dagster_synplanner.resources.config import SynPlannerResource


@op(
    ins={
        "config_path": In(str, description="Path to rule extraction YAML config"),
        "input_file": In(str, description="Path to mapped reaction data"),
    },
    out=Out(str, description="Path to extracted reaction rules file"),
    tags={"kind": "training"},
    description="Extract reaction rules/templates from reaction data.",
)
def rule_extracting_op(
    context: OpExecutionContext,
    synplanner: SynPlannerResource,
    config_path: str,
    input_file: str,
) -> str:
    from synplan.chem.reaction_rules.extraction import extract_rules_from_reactions
    from synplan.utils.config import RuleExtractionConfig

    output_file = str(synplanner.output_dir / "reaction_rules.pickle")
    error_file = str(synplanner.output_dir / "rule_extraction.errors.tsv")

    context.log.info(f"Extracting reaction rules from: {input_file}")

    reaction_rule_config = RuleExtractionConfig.from_yaml(config_path)
    extract_rules_from_reactions(
        config=reaction_rule_config,
        reaction_data_path=input_file,
        reaction_rules_path=output_file,
        num_cpus=synplanner.num_cpus,
        batch_size=synplanner.batch_size,
        ignore_errors=True,
        error_file_path=error_file,
    )

    context.add_output_metadata({
        "output_file": MetadataValue.path(output_file),
    })
    return output_file


@op(
    ins={
        "config_path": In(str, description="Path to policy training YAML config"),
        "policy_data": In(str, description="Path to policy training data TSV"),
    },
    out=Out(str, description="Path to trained ranking policy network weights"),
    tags={"kind": "training", "compute": "gpu"},
    description="Train ranking policy network (supervised learning).",
)
def ranking_policy_training_op(
    context: OpExecutionContext,
    synplanner: SynPlannerResource,
    config_path: str,
    policy_data: str,
) -> str:
    from synplan.ml.training.supervised import create_policy_dataset, run_policy_training
    from synplan.utils.config import PolicyNetworkConfig

    results_dir = str(synplanner.output_dir / "ranking_policy")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    context.log.info(f"Training ranking policy from: {policy_data}")

    policy_config = PolicyNetworkConfig.from_yaml(config_path)
    policy_config.policy_type = "ranking"

    datamodule = create_policy_dataset(
        policy_data_path=policy_data,
        results_dir=results_dir,
        dataset_type="ranking",
        batch_size=policy_config.batch_size,
        num_workers=0,
        cache=True,
    )

    run_policy_training(datamodule, config=policy_config, results_path=results_dir)

    weights_path = str(Path(results_dir) / "ranking_policy_network.ckpt")
    context.add_output_metadata({
        "results_dir": MetadataValue.path(results_dir),
        "weights_path": MetadataValue.path(weights_path),
    })
    return weights_path


@op(
    ins={
        "config_path": In(str, description="Path to policy training YAML config"),
        "molecule_data": In(str, description="Path to molecule data"),
        "reaction_rules": In(str, description="Path to extracted reaction rules"),
    },
    out=Out(str, description="Path to trained filtering policy network weights"),
    tags={"kind": "training", "compute": "gpu"},
    description="Train filtering policy network (supervised learning).",
)
def filtering_policy_training_op(
    context: OpExecutionContext,
    synplanner: SynPlannerResource,
    config_path: str,
    molecule_data: str,
    reaction_rules: str,
) -> str:
    from synplan.ml.training.supervised import create_policy_dataset, run_policy_training
    from synplan.utils.config import PolicyNetworkConfig

    results_dir = str(synplanner.output_dir / "filtering_policy")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    context.log.info(f"Training filtering policy with rules: {reaction_rules}")

    policy_config = PolicyNetworkConfig.from_yaml(config_path)
    policy_config.policy_type = "filtering"

    datamodule = create_policy_dataset(
        reaction_rules_path=reaction_rules,
        molecules_or_reactions_path=molecule_data,
        results_dir=results_dir,
        dataset_type="filtering",
        batch_size=policy_config.batch_size,
        num_cpus=synplanner.num_cpus,
        cache=True,
    )

    run_policy_training(datamodule, config=policy_config, results_path=results_dir)

    weights_path = str(Path(results_dir) / "filtering_policy_network.ckpt")
    context.add_output_metadata({
        "results_dir": MetadataValue.path(results_dir),
        "weights_path": MetadataValue.path(weights_path),
    })
    return weights_path


@op(
    ins={
        "config_path": In(str, description="Path to value network tuning YAML config"),
        "targets": In(str, description="Path to target molecules"),
        "reaction_rules": In(str, description="Path to reaction rules"),
        "building_blocks": In(str, description="Path to building blocks"),
        "policy_network": In(str, description="Path to trained policy network"),
    },
    out=Out(str, description="Path to tuned value network weights"),
    tags={"kind": "training", "compute": "gpu"},
    description="Fine-tune value network via reinforcement learning.",
)
def value_network_tuning_op(
    context: OpExecutionContext,
    synplanner: SynPlannerResource,
    config_path: str,
    targets: str,
    reaction_rules: str,
    building_blocks: str,
    policy_network: str,
    value_network: str | None = None,
) -> str:
    from synplan.ml.training.reinforcement import run_updating
    from synplan.utils.config import (
        PolicyNetworkConfig,
        TreeConfig,
        TuningConfig,
        ValueNetworkConfig,
    )

    results_dir = str(synplanner.output_dir / "value_network")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    context.log.info("Tuning value network via RL")

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

    weights_path = str(Path(results_dir) / "value_network.ckpt")
    context.add_output_metadata({
        "results_dir": MetadataValue.path(results_dir),
        "weights_path": MetadataValue.path(weights_path),
    })
    return weights_path
