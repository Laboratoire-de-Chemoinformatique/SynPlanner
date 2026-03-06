"""File-based sensors that trigger jobs when new data appears.

Useful for automated workflows where users drop target molecule files
into a watched directory to trigger planning runs.
"""

import os
from pathlib import Path

from dagster import RunRequest, SensorEvaluationContext, sensor


@sensor(
    minimum_interval_seconds=30,
    description=(
        "Watches a directory for new target molecule files (*.smi) "
        "and triggers planning jobs automatically."
    ),
)
def new_targets_sensor(context: SensorEvaluationContext):
    """Sensor that detects new .smi files in the targets watch directory."""
    watch_dir = os.environ.get("SYNPLANNER_TARGETS_DIR", "/data/synplanner/targets")
    watch_path = Path(watch_dir)

    if not watch_path.exists():
        return

    last_mtime = float(context.cursor) if context.cursor else 0.0

    new_files = []
    max_mtime = last_mtime

    for smi_file in watch_path.glob("*.smi"):
        file_mtime = smi_file.stat().st_mtime
        if file_mtime > last_mtime:
            new_files.append(smi_file)
            max_mtime = max(max_mtime, file_mtime)

    for target_file in new_files:
        yield RunRequest(
            run_key=f"planning_{target_file.name}_{target_file.stat().st_mtime}",
            run_config={
                "ops": {
                    "planning_op": {
                        "inputs": {
                            "targets": str(target_file),
                        }
                    }
                }
            },
            tags={"source": "file_sensor", "target_file": target_file.name},
        )

    if max_mtime > last_mtime:
        context.update_cursor(str(max_mtime))
