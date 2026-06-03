from chython.containers import MoleculeContainer

from synplan.chem.reaction_routes.leaving_groups import DynamicX, MarkedAt, MarkedY


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
