from types import SimpleNamespace

from synplan.chem.reaction.routes.representation.state import _set_symmetric_bond


def test_set_symmetric_bond_stores_one_shared_bond_object():
    cgr = SimpleNamespace(_bonds={})
    bond = object()

    _set_symmetric_bond(cgr, 2, 1, bond)

    assert cgr._bonds[1][2] is bond
    assert cgr._bonds[2][1] is bond


def test_set_symmetric_bond_preserves_existing_adjacency_maps():
    existing = {3: object()}
    cgr = SimpleNamespace(_bonds={1: existing})
    bond = object()

    _set_symmetric_bond(cgr, 1, 2, bond)

    assert cgr._bonds[1] is existing
    assert cgr._bonds[1][3] is not bond
    assert cgr._bonds[1][2] is bond
