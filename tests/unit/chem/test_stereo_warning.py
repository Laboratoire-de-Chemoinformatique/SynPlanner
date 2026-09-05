"""The clean_stereo strip must state itself instead of flattening silently."""

import warnings

import pytest
from chython import smiles

from synplan.chem.utils import (
    StereoDiscardedWarning,
    clean_molecule,
    mol_from_smiles,
    safe_canonicalization,
)

TETRAHEDRAL = "C[C@H]1CCCN(C)C1"
GEOMETRIC = "C/C=C/C(=O)O"
FLAT = "CN1CCCC(C)C1"


@pytest.mark.parametrize("smi", [TETRAHEDRAL, GEOMETRIC])
def test_clean_molecule_warns_before_discarding_stereo(smi: str) -> None:
    with pytest.warns(StereoDiscardedWarning):
        cleaned = clean_molecule(smiles(smi))
    assert not any(a.stereo is not None for _, a in cleaned.atoms())


@pytest.mark.parametrize("smi", [TETRAHEDRAL, GEOMETRIC])
def test_safe_canonicalization_warns_too(smi: str) -> None:
    with pytest.warns(StereoDiscardedWarning):
        safe_canonicalization(smiles(smi))


def test_no_warning_for_a_flat_molecule() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", StereoDiscardedWarning)
        safe_canonicalization(smiles(FLAT))
        clean_molecule(smiles(FLAT))


def test_no_warning_when_clean_stereo_is_off() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", StereoDiscardedWarning)
        kept = mol_from_smiles(TETRAHEDRAL, clean_stereo=False, clean2d=False)
    assert any(a.stereo is not None for _, a in kept.atoms())


def test_safe_canonicalization_flag_controls_stereo_preservation() -> None:
    molecule = smiles(TETRAHEDRAL, ignore_stereo=False)
    with pytest.warns(StereoDiscardedWarning):
        flattened = safe_canonicalization(molecule)
    preserved = safe_canonicalization(molecule, clean_stereo=False)

    assert not any(atom.stereo is not None for _, atom in flattened.atoms())
    assert any(atom.stereo is not None for _, atom in preserved.atoms())
    assert any(atom.stereo is not None for _, atom in molecule.atoms())


def test_the_warning_can_be_promoted_to_a_refusal() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", StereoDiscardedWarning)
        with pytest.raises(StereoDiscardedWarning):
            mol_from_smiles(TETRAHEDRAL)
