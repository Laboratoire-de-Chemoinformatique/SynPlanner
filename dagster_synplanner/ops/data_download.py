"""Dagster op for downloading SynPlanner presets from HuggingFace."""

from dagster import MetadataValue, OpExecutionContext, Out, op

from dagster_synplanner.resources.config import SynPlannerResource


@op(
    out=Out(dict, description="Dictionary of downloaded file paths"),
    tags={"kind": "data_download"},
    description="Download a ready-to-use data preset from HuggingFace.",
)
def download_preset_op(
    context: OpExecutionContext,
    synplanner: SynPlannerResource,
    preset_name: str = "synplanner-article",
) -> dict:
    from synplan.utils.loading import download_preset

    save_to = str(synplanner.output_dir / "presets")
    context.log.info(f"Downloading preset '{preset_name}' to {save_to}")

    paths = download_preset(preset_name=preset_name, save_to=save_to)

    context.add_output_metadata({
        k: MetadataValue.path(str(v)) for k, v in paths.items() if v is not None
    })
    return {k: str(v) for k, v in paths.items() if v is not None}
