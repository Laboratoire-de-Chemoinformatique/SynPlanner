from chython import smiles

from synplan.chem.utils import _clean_molecule


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
