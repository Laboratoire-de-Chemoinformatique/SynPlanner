"""CSV codec for route reaction dictionaries."""

from __future__ import annotations

import csv
import json

from chython import smiles as read_smiles

from synplan.chem.reaction.routes.io.metadata import (
    reaction_metadata,
    restore_reaction_metadata,
)


def read_routes_csv(file_path="routes.csv"):
    """Read route reactions from a CSV file."""

    routes_dict = {}
    with open(file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            route_id = int(row["route_id"])
            step_id = int(row["step_id"])
            reaction = read_smiles(row["smiles"])
            raw_meta = (row.get("meta") or "").strip()
            if raw_meta:
                try:
                    metadata = json.loads(raw_meta)
                except json.JSONDecodeError:
                    restore_reaction_metadata(reaction, {"legacy_csv_meta": raw_meta})
                else:
                    restore_reaction_metadata(reaction, metadata)
            routes_dict.setdefault(route_id, {})[step_id] = reaction
    return routes_dict


def write_routes_csv(routes_dict, file_path="routes.csv"):
    """Write route reactions with stable JSON metadata cells."""

    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["route_id", "step_id", "smiles", "meta"])
        for route_id in sorted(routes_dict):
            for step_id in sorted(routes_dict[route_id]):
                reaction = routes_dict[route_id][step_id]
                metadata = reaction_metadata(reaction)
                meta = (
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                    if metadata
                    else ""
                )
                writer.writerow([route_id, step_id, format(reaction, "m"), meta])


__all__ = ["read_routes_csv", "write_routes_csv"]
