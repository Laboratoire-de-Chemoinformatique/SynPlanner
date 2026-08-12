import pytest
from chython import smiles
from rdkit.Chem import rdinchi

from synplan.chem.building_blocks.identity import (
    MoleculeIdentityError,
    inchi_to_inchi_key,
    molecule_identity,
    molecule_to_inchi,
    molecule_to_inchi_key,
)


def test_water_identity_matches_standard_inchi():
    identity = molecule_identity(smiles("O"))

    assert identity.canonical_smiles == "O"
    assert identity.standard_inchi == "InChI=1S/H2O/h1H2"
    assert identity.inchi_key == "XLYOFNOQVPJJNP-UHFFFAOYSA-N"
    assert identity.return_code == 0
    assert identity.warnings == ()


def test_identity_preserves_stereo_and_ignores_atom_mapping():
    left = smiles("F[C@H](Cl)Br")
    right = smiles("F[C@@H](Cl)Br")
    mapped = smiles("[F:1][C@H:2]([Cl:3])[Br:4]")
    original = str(mapped)

    assert molecule_to_inchi_key(left) != molecule_to_inchi_key(right)
    assert molecule_to_inchi_key(mapped) == molecule_to_inchi_key(left)
    assert str(mapped) == original


def test_identity_preserves_complete_salt_record():
    identity = molecule_identity(smiles("CCO.O"))

    assert identity.standard_inchi == "InChI=1S/C2H6O.H2O/c1-2-3;/h3H,2H2,1H3;1H2"
    assert identity.inchi_key == "IDGUHHHQCWSQLU-UHFFFAOYSA-N"


def test_warning_return_code_is_success_and_captured():
    identity = molecule_identity(smiles("[Na+].[O-]C(=O)C"))

    assert identity.return_code == 1
    assert "Proton(s) added/removed" in identity.warnings
    assert identity.standard_inchi.startswith("InChI=1S/")


def test_helpers_derive_key_from_exact_inchi():
    molecule = smiles("C/C=C/C")
    inchi = molecule_to_inchi(molecule)

    assert molecule_to_inchi_key(molecule) == rdinchi.InchiToInchiKey(inchi)
    assert inchi_to_inchi_key(inchi) == rdinchi.InchiToInchiKey(inchi)


@pytest.mark.parametrize(
    "inchi",
    ["InChI=1/CH4/h1H4", "InChI=", "not-an-inchi", ""],
)
def test_inchi_to_inchi_key_rejects_non_standard_or_malformed(inchi):
    with pytest.raises(MoleculeIdentityError):
        inchi_to_inchi_key(inchi)


def test_molecule_identity_rejects_failed_rdkit_return_code(monkeypatch):
    monkeypatch.setattr(
        "synplan.chem.building_blocks.identity.rdinchi.MolToInchi",
        lambda _mol: ("", 2, "bad molecule", "", ""),
    )

    with pytest.raises(MoleculeIdentityError, match="return code 2"):
        molecule_identity(smiles("O"))
