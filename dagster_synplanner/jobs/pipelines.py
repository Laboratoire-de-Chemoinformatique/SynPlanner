"""Dagster job definitions composing SynPlanner ops into end-to-end workflows.

Jobs represent the main workflows users will trigger:
- data_preparation_job: Raw data -> standardized, filtered, mapped reactions
- full_training_pipeline_job: Data prep + rule extraction + policy training
- planning_job: Run retrosynthesis on targets with trained models
- retrain_and_plan_job: Full pipeline from raw data to planning results
"""

from dagster import graph, job, op

from dagster_synplanner.ops.data_processing import (
    building_blocks_standardizing_op,
    reaction_filtering_op,
    reaction_mapping_op,
    reaction_standardizing_op,
)
from dagster_synplanner.ops.model_training import (
    ranking_policy_training_op,
    rule_extracting_op,
    value_network_tuning_op,
)
from dagster_synplanner.ops.planning import clustering_op, planning_op


@graph
def data_preparation_graph():
    """Standardize -> Filter -> Map reactions."""
    standardized = reaction_standardizing_op()
    filtered = reaction_filtering_op(input_file=standardized)
    reaction_mapping_op(input_file=filtered)


data_preparation_job = data_preparation_graph.to_job(
    name="data_preparation",
    description=(
        "Full data preparation pipeline: standardize reactions, "
        "filter erroneous ones, and map atoms."
    ),
    tags={"pipeline": "data_prep"},
)


@graph
def full_training_pipeline_graph():
    """Data prep -> Rule extraction -> Policy training."""
    standardized = reaction_standardizing_op()
    filtered = reaction_filtering_op(input_file=standardized)
    mapped = reaction_mapping_op(input_file=filtered)
    rules = rule_extracting_op(input_file=mapped)
    ranking_policy_training_op(policy_data=rules)


full_training_pipeline_job = full_training_pipeline_graph.to_job(
    name="full_training_pipeline",
    description=(
        "End-to-end training pipeline from raw reaction data to "
        "trained ranking policy network."
    ),
    tags={"pipeline": "training"},
)


@graph
def planning_graph():
    """Run retrosynthetic planning and cluster results."""
    results_dir = planning_op()
    clustering_op(routes_file=results_dir)


planning_job = planning_graph.to_job(
    name="planning",
    description="Run retrosynthetic planning on targets and cluster routes.",
    tags={"pipeline": "planning"},
)


@graph
def retrain_and_plan_graph():
    """Full pipeline: data prep -> training -> planning."""
    standardized = reaction_standardizing_op()
    filtered = reaction_filtering_op(input_file=standardized)
    mapped = reaction_mapping_op(input_file=filtered)
    rules = rule_extracting_op(input_file=mapped)
    policy_weights = ranking_policy_training_op(policy_data=rules)
    results_dir = planning_op(policy_network=policy_weights)
    clustering_op(routes_file=results_dir)


retrain_and_plan_job = retrain_and_plan_graph.to_job(
    name="retrain_and_plan",
    description="Complete workflow from raw data to planning results.",
    tags={"pipeline": "full"},
)
