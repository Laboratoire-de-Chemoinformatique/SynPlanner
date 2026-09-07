from chython.containers import CGRContainer
from chython.containers.bonds import DynamicBond

from synplan.chem.reaction.routes.representation.state import set_symmetric_bond


def test_set_symmetric_bond_stores_one_shared_bond_object():
    cgr = CGRContainer()
    cgr.add_atom("C", 1)
    cgr.add_atom("C", 2)
    bond = DynamicBond(1, 2)

    set_symmetric_bond(cgr, 2, 1, bond)

    assert cgr.bond(1, 2) is bond
    assert cgr.bond(2, 1) is bond


def test_set_symmetric_bond_preserves_other_bonds():
    cgr = CGRContainer()
    for n in (1, 2, 3):
        cgr.add_atom("C", n)
    existing = DynamicBond(1, 1)
    cgr.add_bond(1, 3, existing)
    bond = DynamicBond(1, 2)

    set_symmetric_bond(cgr, 1, 2, bond)

    assert cgr.bond(1, 3) is existing
    assert cgr.bond(1, 2) is cgr.bond(2, 1) is bond
