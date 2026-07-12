from chython import smiles
from chython.containers import MoleculeContainer

from synplan.chem.reaction.routes.clustering.pseudo_atoms import (
    DynamicX,
    MarkedAt,
    MarkedY,
)
from synplan.chem.reaction.routes.clustering.subclustering import lg_process_reset


def _one_atom_smiles(atom):
    mol = MoleculeContainer()
    mol.add_atom(atom, 1)
    return str(mol)


def test_marked_atoms_keep_route_marker_representation():
    x = MarkedAt()
    y = MarkedY()

    assert repr(x) == str(x) == "X(0)"
    assert repr(y) == str(y) == "Y(0)"
    assert x.atomic_symbol == x.symbol == "X"
    assert y.atomic_symbol == y.symbol == "Y"
    assert _one_atom_smiles(x) == "[X]"
    assert _one_atom_smiles(y) == "[Y]"

    x.mark = y.mark = 3
    x.isotope = y.isotope = 3

    assert repr(x) == str(x) == "X(3)"
    assert repr(y) == str(y) == "Y(3)"
    assert _one_atom_smiles(x) == "[3X]"
    assert _one_atom_smiles(y) == "[3Y]"
    assert x != y
    assert hash(x) != hash(y)
    assert len({x, y}) == 2


def test_marked_atom_roles_participate_in_equality():
    leaving_group = MarkedAt(mark=1)
    supporting_group = MarkedY(mark=1)
    leaving_group.isotope = supporting_group.isotope = 1

    assert leaving_group != supporting_group
    assert hash(leaving_group) != hash(supporting_group)
    assert len({leaving_group, supporting_group}) == 2


def test_dynamic_x_keeps_route_marker_representation():
    x = DynamicX()

    assert repr(x) == str(x) == "DynamicX()"
    assert x.atomic_symbol == x.symbol == "X"
    assert x.mark is None
    assert x.isotope is None

    x.mark = 3
    x.isotope = 3

    assert repr(x) == str(x) == "DynamicX()"
    assert x.mark == 3
    assert x.isotope == 3


def test_lg_process_reset_syncs_radical_state_to_atom_object():
    reaction = smiles("[CH3:1][Cl:2]>>[CH4:1].[ClH:2]")
    leaving_group_cgr = lg_process_reset(~reaction, 2)

    assert leaving_group_cgr._radicals[2] is True
    assert leaving_group_cgr._atoms[2].is_radical is True
