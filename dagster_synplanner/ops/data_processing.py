"""Dagster ops for SynPlanner data processing tasks.

These are the long-running data pipeline steps: standardization, filtering,
mapping of reactions and building blocks.
"""

from pathlib import Path

from dagster import In, MetadataValue, OpExecutionContext, Out, op

from dagster_synplanner.resources.config import SynPlannerResource


@op(
    ins={
        "input_file": In(str, description="Path to raw building blocks file"),
    },
    out=Out(str, description="Path to standardized building blocks file"),
    tags={"kind": "data_processing"},
    description="Standardize building block molecules.",
)
def building_blocks_standardizing_op(
    context: OpExecutionContext,
    synplanner: SynPlannerResource,
    input_file: str,
) -> str:
    from synplan.chem.utils import standardize_building_blocks

    output_file = str(
        synplanner.output_dir / "building_blocks_standardized.smi"
    )
    context.log.info(f"Standardizing building blocks: {input_file} -> {output_file}")

    standardize_building_blocks(input_file=input_file, output_file=output_file)

    context.add_output_metadata(
        {"output_file": MetadataValue.path(output_file)}
    )
    return output_file


@op(
    ins={
        "config_path": In(str, description="Path to standardization YAML config"),
        "input_file": In(str, description="Path to raw reaction data"),
    },
    out=Out(str, description="Path to standardized reactions file"),
    tags={"kind": "data_processing"},
    description="Standardize reactions and remove duplicates.",
)
def reaction_standardizing_op(
    context: OpExecutionContext,
    synplanner: SynPlannerResource,
    config_path: str,
    input_file: str,
) -> str:
    from synplan.chem.data.standardizing import (
        ReactionStandardizationConfig,
        standardize_reactions_from_file,
    )

    output_file = str(
        synplanner.output_dir / "reactions_standardized.smi"
    )
    error_file = str(synplanner.output_dir / "reactions_standardized.errors.tsv")

    context.log.info(f"Standardizing reactions: {input_file}")

    stand_config = ReactionStandardizationConfig.from_yaml(config_path)
    standardize_reactions_from_file(
        config=stand_config,
        input_reaction_data_path=input_file,
        standardized_reaction_data_path=output_file,
        num_cpus=synplanner.num_cpus,
        batch_size=synplanner.batch_size,
        ignore_errors=True,
        error_file_path=error_file,
    )

    context.add_output_metadata({
        "output_file": MetadataValue.path(output_file),
        "error_file": MetadataValue.path(error_file),
    })
    return output_file


@op(
    ins={
        "config_path": In(str, description="Path to filtering YAML config"),
        "input_file": In(str, description="Path to reaction data to filter"),
    },
    out=Out(str, description="Path to filtered reactions file"),
    tags={"kind": "data_processing"},
    description="Filter erroneous reactions from the dataset.",
)
def reaction_filtering_op(
    context: OpExecutionContext,
    synplanner: SynPlannerResource,
    config_path: str,
    input_file: str,
) -> str:
    from synplan.chem.data.filtering import ReactionFilterConfig, filter_reactions_from_file

    output_file = str(
        synplanner.output_dir / "reactions_filtered.smi"
    )
    error_file = str(synplanner.output_dir / "reactions_filtered.errors.tsv")

    context.log.info(f"Filtering reactions: {input_file}")

    reaction_check_config = ReactionFilterConfig().from_yaml(config_path)
    filter_reactions_from_file(
        config=reaction_check_config,
        input_reaction_data_path=input_file,
        filtered_reaction_data_path=output_file,
        num_cpus=synplanner.num_cpus,
        batch_size=synplanner.batch_size,
        ignore_errors=True,
        error_file_path=error_file,
    )

    context.add_output_metadata({
        "output_file": MetadataValue.path(output_file),
        "error_file": MetadataValue.path(error_file),
    })
    return output_file


@op(
    ins={
        "input_file": In(str, description="Path to reaction data to map"),
    },
    out=Out(str, description="Path to mapped reactions file"),
    tags={"kind": "data_processing"},
    description="Map reaction atoms using a neural attention model.",
)
def reaction_mapping_op(
    context: OpExecutionContext,
    synplanner: SynPlannerResource,
    input_file: str,
    config_path: str | None = None,
) -> str:
    from synplan.chem.data.mapping import MappingConfig, map_reactions_from_file

    output_file = str(synplanner.output_dir / "reactions_mapped.smi")
    error_file = str(synplanner.output_dir / "reactions_mapped.errors.tsv")

    context.log.info(f"Mapping reactions: {input_file}")

    config = MappingConfig.from_yaml(config_path) if config_path else MappingConfig()
    config.device = synplanner.device

    map_reactions_from_file(
        config=config,
        input_reaction_data_path=input_file,
        mapped_reaction_data_path=output_file,
        num_workers=synplanner.num_cpus,
        silent=False,
        ignore_errors=True,
        error_file_path=error_file,
    )

    context.add_output_metadata({
        "output_file": MetadataValue.path(output_file),
        "error_file": MetadataValue.path(error_file),
    })
    return output_file
