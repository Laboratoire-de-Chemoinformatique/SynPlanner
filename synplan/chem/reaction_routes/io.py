"""Compatibility facade for the historical route I/O module.

JSON, CSV, and Tree adapters now live below ``routes.io``. The old module is
kept so existing imports continue to work while callers migrate.
"""

from __future__ import annotations

import importlib
import warnings

_EXPORTS = {
    "_collect_reactions": (
        "synplan.chem.reaction.routes.io.json",
        "_collect_reactions",
    ),
    "build_route_trees": (
        "synplan.chem.reaction.routes.io",
        "build_route_trees",
    ),
    "export_tree_to_csv": (
        "synplan.chem.reaction.routes.io",
        "export_tree_to_csv",
    ),
    "export_tree_to_json": (
        "synplan.chem.reaction.routes.io",
        "export_tree_to_json",
    ),
    "make_dict": ("synplan.chem.reaction.routes.io", "make_dict"),
    "make_json": ("synplan.chem.reaction.routes.io", "make_json"),
    "read_routes_csv": (
        "synplan.chem.reaction.routes.io",
        "read_routes_csv",
    ),
    "read_routes_json": (
        "synplan.chem.reaction.routes.io",
        "read_routes_json",
    ),
    "write_routes_csv": (
        "synplan.chem.reaction.routes.io",
        "write_routes_csv",
    ),
    "write_routes_json": (
        "synplan.chem.reaction.routes.io",
        "write_routes_json",
    ),
}

__all__ = sorted(name for name in _EXPORTS if not name.startswith("_"))


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"{__name__}.{name} is deprecated; import it from {target[0]} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(_EXPORTS)
