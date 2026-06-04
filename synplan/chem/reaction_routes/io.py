"""Compatibility wrapper for :mod:`synplan.routes.io`."""

from synplan.routes.io import (
    export_tree_to_csv,
    export_tree_to_json,
    make_dict,
    make_json,
    read_routes_csv,
    read_routes_json,
    write_routes_csv,
    write_routes_json,
)

__all__ = [
    "export_tree_to_csv",
    "export_tree_to_json",
    "make_dict",
    "make_json",
    "read_routes_csv",
    "read_routes_json",
    "write_routes_csv",
    "write_routes_json",
]
