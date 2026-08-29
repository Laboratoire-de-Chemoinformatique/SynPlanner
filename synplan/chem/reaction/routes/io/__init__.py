"""Stable public facade for route JSON, CSV, and Tree export APIs."""

from synplan.chem.reaction.routes.io.csv import read_routes_csv, write_routes_csv
from synplan.chem.reaction.routes.io.json import (
    build_route_trees,
    make_dict,
    make_json,
    read_route_tree,
    read_routes_json,
    write_routes_json,
)
from synplan.chem.reaction.routes.io.tree import (
    export_tree_to_csv,
    export_tree_to_json,
)

__all__ = [
    "build_route_trees",
    "export_tree_to_csv",
    "export_tree_to_json",
    "make_dict",
    "make_json",
    "read_route_tree",
    "read_routes_csv",
    "read_routes_json",
    "write_routes_csv",
    "write_routes_json",
]


def __getattr__(name: str):
    if name == "route_tree_has_null_node":
        from synplan.chem.reaction.routes.io.json import route_tree_has_null_node

        return route_tree_has_null_node
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
