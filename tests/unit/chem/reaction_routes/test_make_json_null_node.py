"""Route export must drop malformed routes instead of emitting a null node.

``make_json`` builds each route via ``build_mol_node``, which returns ``None``
for a malformed node (e.g. a node holding more than one molecule). Without the
guard, that ``None`` is written into the tree as a JSON ``null`` child, which
downstream consumers (retrocast) reject as schema-invalid. The fix drops the
whole malformed route and logs a warning; valid routes are untouched.
"""

from __future__ import annotations

import logging

from chython import smiles as read_smiles

from synplan.chem.reaction.routes.io import _route_tree_has_null_node, make_json


def _malformed_route_steps():
    """Two-step route whose nested ``build_mol_node`` returns ``None``.

    The final step's reactant SMILES matches an earlier step's product, so the
    prior-step lookup recurses into ``build_mol_node``. That earlier product
    carries atom numbers disjoint from the final target's atoms, so the overlap
    check fails and ``build_mol_node`` hits its ``return None`` branch.
    """
    final_step = read_smiles("[C:1][C:2]>>[O:1]=[O:2]")
    prior_step = read_smiles("[Br:60]>>[C:50][C:51]")
    return {0: prior_step, 1: final_step}


def _valid_route_steps():
    """A single-step route that serializes to a well-formed tree."""
    return {0: read_smiles("[C:1][C:2][C:3].[N:10]>>[C:1][C:2][C:3][N:10]")}


def test_route_tree_has_null_node_detects_nested_none():
    assert _route_tree_has_null_node(None) is True
    nested = {"type": "mol", "children": [{"type": "reaction", "children": [None]}]}
    assert _route_tree_has_null_node(nested) is True
    clean = {"type": "mol", "children": [{"type": "reaction", "children": []}]}
    assert _route_tree_has_null_node(clean) is False


def test_make_json_drops_malformed_route(caplog):
    routes_dict = {5: _valid_route_steps(), 9: _malformed_route_steps()}

    with caplog.at_level(logging.WARNING, logger="synplan.chem.reaction.routes.io"):
        out = make_json(routes_dict)

    # (a) malformed route absent; no null-child tree emitted.
    assert 9 not in out
    assert all(not _route_tree_has_null_node(tree) for tree in out.values())

    # (b) valid route unaffected.
    assert 5 in out
    assert out[5]["type"] == "mol"
    assert out[5]["smiles"] == "CCCN"

    # (c) warning logged (not printed) naming the dropped route.
    assert any(
        record.levelno == logging.WARNING and "9" in record.getMessage()
        for record in caplog.records
    )


def test_make_json_keep_ids_false_excludes_malformed_route():
    routes_dict = {5: _valid_route_steps(), 9: _malformed_route_steps()}
    out = make_json(routes_dict, keep_ids=False)
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["smiles"] == "CCCN"
    assert not _route_tree_has_null_node(out[0])
