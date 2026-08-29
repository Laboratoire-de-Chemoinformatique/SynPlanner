"""Route export recovers disjoint-numbering routes; guard stays as safety net.

``make_json`` builds each precursor mol-node by selecting the producing step's
product fragment. Selecting by atom-number overlap with the target fails on deep
routes where per-step atom-number chaining leaves the relevant fragment numbered
disjoint from the target, yielding a ``None`` node (JSON ``null``) that
downstream consumers reject. The fix selects the fragment that *is* the consuming
reactant by chython structural equality, recovering the route regardless of
numbering. The drop guard (``route_tree_has_null_node``) remains as a last-resort
safety net for any genuinely unrecoverable tree.
"""

from __future__ import annotations

from chython import smiles as read_smiles

from synplan.chem.reaction.routes.io import make_json, route_tree_has_null_node


def _disjoint_numbering_route_steps():
    """Two-step route whose precursor fragment is numbered disjoint from target.

    The final step's reactant ``CC`` matches an earlier step's product ``CC``, so
    the prior-step lookup recurses into ``build_mol_node``. That earlier product
    carries atom numbers (50, 51) disjoint from the final target ``O=O`` atoms
    (1, 2). Atom-number overlap selection fails; structural selection (the
    product fragment that *equals* the consuming reactant) recovers the route.
    """
    final_step = read_smiles("[C:1][C:2]>>[O:1]=[O:2]")
    prior_step = read_smiles("[Br:60]>>[C:50][C:51]")
    return {0: prior_step, 1: final_step}


def _valid_route_steps():
    """A single-step route that serializes to a well-formed tree."""
    return {0: read_smiles("[C:1][C:2][C:3].[N:10]>>[C:1][C:2][C:3][N:10]")}


def _assert_well_formed(node, expect="mol"):
    """Valid mol/reaction alternation, no null node, non-empty mol smiles."""
    assert isinstance(node, dict), f"expected dict node, got {node!r}"
    assert node.get("type") == expect, f"expected {expect}, got {node.get('type')!r}"
    children = node.get("children") or []
    if expect == "mol":
        assert node.get("smiles"), "mol node missing smiles"
        for child in children:
            _assert_well_formed(child, "reaction")
    else:
        assert children, "reaction node must have at least one mol child"
        for child in children:
            _assert_well_formed(child, "mol")


def test_route_tree_has_null_node_detects_nested_none():
    assert route_tree_has_null_node(None) is True
    nested = {"type": "mol", "children": [{"type": "reaction", "children": [None]}]}
    assert route_tree_has_null_node(nested) is True
    clean = {"type": "mol", "children": [{"type": "reaction", "children": []}]}
    assert route_tree_has_null_node(clean) is False


def test_make_json_recovers_disjoint_numbering_route():
    """The disjoint-numbering route is recovered (well-formed), not dropped."""
    routes_dict = {5: _valid_route_steps(), 9: _disjoint_numbering_route_steps()}

    out = make_json(routes_dict)

    # Both routes present; neither contains a null node.
    assert 5 in out and 9 in out
    assert all(not route_tree_has_null_node(tree) for tree in out.values())

    # Recovered route is a well-formed tree with correct mol/reaction alternation.
    _assert_well_formed(out[9], "mol")
    # The precursor fragment selected by structure is the ``CC`` reactant, which
    # recurses into the prior step that produces it.
    assert out[9]["smiles"] == "O=O"
    rxn_node = out[9]["children"][0]
    assert rxn_node["type"] == "reaction"
    cc_node = rxn_node["children"][0]
    assert cc_node["type"] == "mol" and cc_node["smiles"] == "CC"
    assert cc_node["in_stock"] is False
    assert cc_node["children"][0]["type"] == "reaction"

    # Valid route unaffected.
    assert out[5]["type"] == "mol"
    assert out[5]["smiles"] == "CCCN"


def test_make_json_keep_ids_false_recovers_disjoint_numbering_route():
    routes_dict = {5: _valid_route_steps(), 9: _disjoint_numbering_route_steps()}
    out = make_json(routes_dict, keep_ids=False)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(not route_tree_has_null_node(tree) for tree in out)
    assert {tree["smiles"] for tree in out} == {"CCCN", "O=O"}


def test_make_json_drops_route_when_selection_unrecoverable(monkeypatch):
    """The drop guard stays the safety net for genuinely unrecoverable trees.

    Structural and atom-number selection both succeed for any route whose
    precursor producer is found via ``prod_map`` (same route SMILES implies
    structural equality), so the make_json-level ``None`` path is unreachable
    through normal routes. We force the unrecoverable case by making the nested
    precursor selection return ``None`` for the disjoint-numbering route and
    assert ``make_json`` drops the whole route rather than emitting a ``null``
    child, while the valid route is untouched.
    """
    import synplan.chem.reaction.routes.io as io_mod

    routes_dict = {5: _valid_route_steps(), 9: _disjoint_numbering_route_steps()}

    # Force the structural ``want_react`` branch off: with no structural match
    # and atom numbers disjoint from the target, build_mol_node hits ``None``
    # only for route 9 (route 5 has a single step with no prior-producer
    # recursion, so its root selection still resolves by atom overlap).
    from chython.containers import MoleculeContainer

    monkeypatch.setattr(MoleculeContainer, "__eq__", lambda self, other: False)

    out = io_mod.make_json(routes_dict)

    # Route 9 is unrecoverable under forced selection failure -> dropped, not null.
    assert 9 not in out
    assert all(not route_tree_has_null_node(tree) for tree in out.values())
    # Valid single-step route still exported.
    assert 5 in out and out[5]["smiles"] == "CCCN"
