"""Fast (per-step) vs reconciled (route CGR) route export.

``extract_reactions(..., reconcile_atom_mapping=False)`` (the default) builds the
exported routes_dict straight from ``tree.synthesis_route`` with per-step-local
atom numbering, skipping the expensive ``compose_route_cgr`` route-CGR
composition. ``reconcile_atom_mapping=True`` restores the legacy path with
cross-step-reconciled numbering.

The contract: both paths produce a byte-identical ``make_json`` mol-skeleton
(type / smiles / in_stock / children) -- only the ``reaction`` node mapped
``smiles`` numbering differs (fast = per-step-local, reconciled = cross-step).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

from synplan.chem.reaction.routes.io import (
    make_json,
    read_routes_csv,
    route_tree_has_null_node,
)
from synplan.chem.reaction.routes.representation import extract_reactions


class _MockTree:
    """Minimal tree exposing ``synthesis_route`` and ``winning_nodes``.

    ``synthesis_route`` returns the per-step reactions in chronological order
    (step 0 first .. final last), exactly like the real ``Tree``.
    """

    def __init__(self, routes_dict):
        self._routes_dict = routes_dict
        self.winning_nodes = list(routes_dict)
        self.config = SimpleNamespace(min_mol_size=6)
        self.building_blocks = set()

    def synthesis_route(self, route_id):
        steps = self._routes_dict[route_id]
        return tuple(steps[sid] for sid in sorted(steps))


def _strip_reaction_smiles(node):
    """Deep copy of a route tree with reaction ``smiles`` blanked out."""
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key == "smiles" and node.get("type") == "reaction":
            out[key] = "<RXN>"
        elif key == "children":
            out[key] = [_strip_reaction_smiles(c) for c in (value or [])]
        else:
            out[key] = copy.deepcopy(value)
    return out


def test_fast_matches_reconciled_skeleton_on_real_routes():
    """Fast and reconciled exports share a byte-identical mol-skeleton."""
    data = read_routes_csv("tests/data/routes_mol_1.csv")
    assert data, "fixture routes not loaded"

    json_fast = make_json(
        extract_reactions(_MockTree(copy.deepcopy(data)), reconcile_atom_mapping=False)
    )
    json_rec = make_json(
        extract_reactions(_MockTree(copy.deepcopy(data)), reconcile_atom_mapping=True)
    )

    common = sorted(set(json_fast) & set(json_rec))
    assert common, "no routes exported"
    for route_id in common:
        assert _strip_reaction_smiles(json_fast[route_id]) == _strip_reaction_smiles(
            json_rec[route_id]
        ), f"mol-skeleton differs between fast and reconciled for route {route_id}"


def test_fast_default_produces_valid_no_null_tree():
    """The fast path (default) yields valid, null-free bipartite route trees."""
    data = read_routes_csv("tests/data/routes_mol_1.csv")
    out = make_json(extract_reactions(_MockTree(copy.deepcopy(data))))
    assert out
    for route_id, tree in out.items():
        assert not route_tree_has_null_node(tree), f"route {route_id} has a null node"
        assert tree["type"] == "mol" and tree.get("smiles")


def test_reconcile_default_is_false_signature():
    """``reconcile_atom_mapping`` defaults to the fast path."""
    import inspect

    default = (
        inspect.signature(extract_reactions)
        .parameters["reconcile_atom_mapping"]
        .default
    )
    assert default is False


def test_fast_path_equals_synthesis_route_verbatim():
    """The fast path stores ``tree.synthesis_route`` reactions verbatim.

    No CGR composition, no renumbering: each exported reaction's mapped SMILES
    is exactly the per-step reaction the tree produced, keyed 0..n-1 in
    chronological order.
    """
    data = read_routes_csv("tests/data/routes_mol_1.csv")
    tree = _MockTree(copy.deepcopy(data))

    fast = extract_reactions(tree, reconcile_atom_mapping=False)
    for route_id, steps in fast.items():
        route = tree.synthesis_route(route_id)
        assert sorted(steps) == list(range(len(route)))
        for step_id, reaction in steps.items():
            assert format(reaction, "m") == format(route[step_id], "m")


def test_toggle_changes_exported_reactions():
    """Reconciled (cross-step CGR) reactions differ from the fast per-step ones.

    The reconciled path runs ``compose_route_cgr``, which renumbers atoms across
    steps and folds route history into each step, so its exported reaction
    SMILES differ from the verbatim per-step reactions of the fast path. This
    proves the toggle actually selects a different code path with the documented
    effect (while leaving the mol-skeleton identical, asserted elsewhere).
    """
    data = read_routes_csv("tests/data/routes_mol_1.csv")

    fast = extract_reactions(
        _MockTree(copy.deepcopy(data)), reconcile_atom_mapping=False
    )
    rec = extract_reactions(_MockTree(copy.deepcopy(data)), reconcile_atom_mapping=True)

    def _step_smiles(d):
        return {
            rid: {sid: format(rxn, "m") for sid, rxn in steps.items()}
            for rid, steps in d.items()
        }

    fast_s = _step_smiles(fast)
    rec_s = _step_smiles(rec)

    # At least one route's exported reactions differ between the two paths.
    changed = [rid for rid in set(fast_s) & set(rec_s) if fast_s[rid] != rec_s[rid]]
    assert changed, (
        "reconcile_atom_mapping toggle produced identical reactions on every "
        "route; the reconciled path is not exercising compose_route_cgr"
    )
