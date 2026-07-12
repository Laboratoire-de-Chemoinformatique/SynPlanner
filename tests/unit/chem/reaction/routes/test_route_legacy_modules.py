from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "symbol", "canonical_name"),
    [
        (
            "synplan.chem.reaction_routes.route_cgr",
            "compose_route_cgr",
            "synplan.chem.reaction.routes.representation.route_cgr",
        ),
        (
            "synplan.chem.reaction_routes.clustering",
            "cluster_routes",
            "synplan.chem.reaction.routes.clustering",
        ),
        (
            "synplan.chem.reaction_routes.io",
            "make_json",
            "synplan.chem.reaction.routes.io",
        ),
    ],
)
def test_main_branch_route_modules_reexport_canonical_symbols(
    module_name, symbol, canonical_name
):
    legacy_module = importlib.import_module(module_name)
    canonical_module = importlib.import_module(canonical_name)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy_symbol = getattr(legacy_module, symbol)

    assert legacy_symbol is getattr(canonical_module, symbol)


def test_tree_wrapper_is_removed_from_route_io_namespaces():
    legacy_module = importlib.import_module("synplan.chem.reaction_routes.io")
    canonical_module = importlib.import_module("synplan.chem.reaction.routes.io")

    assert not hasattr(legacy_module, "TreeWrapper")
    assert not hasattr(canonical_module, "TreeWrapper")


def test_canonical_route_exports_remain_available():
    from synplan.chem.reaction.routes.io import (
        export_tree_to_csv,
        export_tree_to_json,
    )

    assert callable(export_tree_to_csv)
    assert callable(export_tree_to_json)


def test_mcts_tree_no_longer_defines_route_exports():
    source = Path("synplan/mcts/tree.py").read_text()

    assert "def export_tree_to_csv" not in source
    assert "def export_tree_to_json" not in source
