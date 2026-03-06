"""Dagster repository definition - the main entry point for dagster.

This module defines the Dagster Definitions object that registers all
jobs, ops, resources, sensors, and schedules for the SynPlanner platform.

Run with:
    dagster dev -m dagster_synplanner.repository
"""

from dagster import Definitions, EnvVar

from dagster_synplanner.jobs.pipelines import (
    data_preparation_job,
    full_training_pipeline_job,
    planning_job,
    retrain_and_plan_job,
)
from dagster_synplanner.resources.config import SynPlannerResource
from dagster_synplanner.sensors.file_sensors import new_targets_sensor

defs = Definitions(
    jobs=[
        data_preparation_job,
        full_training_pipeline_job,
        planning_job,
        retrain_and_plan_job,
    ],
    resources={
        "synplanner": SynPlannerResource(
            output_dir=EnvVar("SYNPLANNER_OUTPUT_DIR").get_value(
                default="/data/synplanner"
            ),
            num_cpus=int(EnvVar("SYNPLANNER_NUM_CPUS").get_value(default="4")),
            batch_size=int(EnvVar("SYNPLANNER_BATCH_SIZE").get_value(default="100")),
            device=EnvVar("SYNPLANNER_DEVICE").get_value(default="cpu"),
        ),
    },
    sensors=[new_targets_sensor],
)
