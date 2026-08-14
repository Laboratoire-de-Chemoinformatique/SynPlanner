from chython import smiles

from synplan.chem.utils import _clean_molecule, safe_canonicalization


def test_clean_molecule_returns_a_copy_when_no_cleanup_is_requested():
    molecule = smiles("CCO")

    cleaned = _clean_molecule(
        molecule,
        standardize=False,
        clean_stereo=False,
        clean2d=False,
    )

    assert cleaned is not molecule
    assert str(cleaned) == str(molecule)


def test_safe_canonicalization_preserves_input_atom_order():
    molecule = smiles("OC[C@@H](F)C")
    original_atom_order = tuple(molecule._atoms)
    original_smiles = str(molecule)

    cleaned = safe_canonicalization(molecule, clean_stereo=False)

    assert cleaned is not molecule
    assert tuple(molecule._atoms) == original_atom_order
    assert str(molecule) == original_smiles


def test_safe_canonicalization_can_preserve_tetrahedral_stereo():
    left = safe_canonicalization(smiles("N[C@@H](C)C(=O)O"), clean_stereo=False)
    right = safe_canonicalization(smiles("N[C@H](C)C(=O)O"), clean_stereo=False)

    assert left != right
    assert "@" in str(left)
    assert "@" in str(right)


def test_safe_canonicalization_retains_historical_stereo_cleanup_default():
    left = safe_canonicalization(smiles("N[C@@H](C)C(=O)O"))
    right = safe_canonicalization(smiles("N[C@H](C)C(=O)O"))

    assert left == right
    assert "@" not in str(left)


def test_safe_canonicalization_can_preserve_double_bond_stereo():
    trans = safe_canonicalization(smiles("F/C=C/F"), clean_stereo=False)
    cis = safe_canonicalization(smiles("F/C=C\\F"), clean_stereo=False)

    assert trans != cis
