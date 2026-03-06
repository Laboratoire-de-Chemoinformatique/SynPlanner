"""Dagster ops for SynPlanner retrosynthetic planning and route clustering."""

from pathlib import Path

import yaml
from dagster import In, MetadataValue, OpExecutionContext, Out, op

from dagster_synplanner.resources.config import SynPlannerResource


@op(
    ins={
        "config_path": In(str, description="Path to planning YAML config"),
        "targets": In(str, description="Path to target molecules file"),
        "reaction_rules": In(str, description="Path to reaction rules"),
        "building_blocks": In(str, description="Path to building blocks"),
        "policy_network": In(str, description="Path to trained policy network"),
    },
    out=Out(str, description="Path to planning results directory"),
    tags={"kind": "planning", "compute": "gpu"},
    description="Run retrosynthetic planning (MCTS) on target molecules.",
)
def planning_op(
    context: OpExecutionContext,
    synplanner: SynPlannerResource,
    config_path: str,
    targets: str,
    reaction_rules: str,
    building_blocks: str,
    policy_network: str,
    value_network: str | None = None,
) -> str:
    from synplan.mcts.search import run_search
    from synplan.utils.config import (
        PolicyEvaluationConfig,
        PolicyNetworkConfig,
        RandomEvaluationConfig,
        RDKitEvaluationConfig,
        RolloutEvaluationConfig,
        ValueNetworkEvaluationConfig,
    )
    from synplan.utils.loading import (
        load_building_blocks,
        load_policy_function,
        load_reaction_rules,
    )

    results_dir = str(synplanner.output_dir / "planning_results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    context.log.info(f"Running retrosynthetic planning on: {targets}")

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    search_config = {**config["tree"], **config.get("node_evaluation", {})}
    policy_config = PolicyNetworkConfig.from_dict(
        {**config["node_expansion"], **{"weights_path": policy_network}}
    )

    node_evaluation = config.get("node_evaluation", {})
    evaluation_type = node_evaluation.get("evaluation_type", "rollout")

    if evaluation_type == "gcn":
        if value_network is None:
            raise ValueError("value_network required for gcn evaluation")
        evaluation_config = ValueNetworkEvaluationConfig(
            weights_path=value_network,
            normalize=node_evaluation.get("normalize", False),
        )
    elif evaluation_type == "rollout":
        policy_function = load_policy_function(weights_path=policy_network)
        reaction_rules_list = load_reaction_rules(reaction_rules)
        building_blocks_set = load_building_blocks(building_blocks, standardize=False)
        evaluation_config = RolloutEvaluationConfig(
            policy_network=policy_function,
            reaction_rules=reaction_rules_list,
            building_blocks=building_blocks_set,
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
        raise ValueError(f"Unknown evaluation_type: {evaluation_type}")

    run_search(
        targets_path=targets,
        search_config=search_config,
        policy_config=policy_config,
        evaluation_config=evaluation_config,
        reaction_rules_path=reaction_rules,
        building_blocks_path=building_blocks,
        results_root=results_dir,
    )

    context.add_output_metadata({
        "results_dir": MetadataValue.path(results_dir),
        "targets": MetadataValue.path(targets),
    })
    return results_dir


@op(
    ins={
        "routes_file": In(str, description="Path to planning routes JSON"),
    },
    out=Out(str, description="Path to clustering results directory"),
    tags={"kind": "planning"},
    description="Cluster discovered synthesis routes by strategic bonds.",
)
def clustering_op(
    context: OpExecutionContext,
    synplanner: SynPlannerResource,
    routes_file: str,
    perform_subcluster: bool = False,
) -> str:
    from synplan.chem.reaction_routes.clustering import run_cluster_cli

    cluster_dir = str(synplanner.output_dir / "clustering_results")
    subcluster_dir = str(synplanner.output_dir / "subclustering_results")
    Path(cluster_dir).mkdir(parents=True, exist_ok=True)

    context.log.info(f"Clustering routes from: {routes_file}")

    run_cluster_cli(
        routes_file=routes_file,
        cluster_results_dir=cluster_dir,
        perform_subcluster=perform_subcluster,
        subcluster_results_dir=subcluster_dir if perform_subcluster else None,
    )

    context.add_output_metadata({
        "cluster_dir": MetadataValue.path(cluster_dir),
    })
    return cluster_dir
