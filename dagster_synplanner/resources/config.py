"""Dagster resource providing shared SynPlanner configuration."""

from pathlib import Path

from dagster import ConfigurableResource


class SynPlannerResource(ConfigurableResource):
    """Shared configuration resource for all SynPlanner ops.

    Provides common settings like output directory, CPU/GPU configuration,
    and batch processing parameters.
    """

    output_dir: str = "/data/synplanner"
    num_cpus: int = 4
    batch_size: int = 100
    device: str = "cpu"

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
