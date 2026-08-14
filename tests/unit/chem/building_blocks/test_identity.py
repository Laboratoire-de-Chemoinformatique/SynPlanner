import pytest
from chython import smiles
from rdkit.Chem import inchi as rdkit_inchi
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


def test_metal_disconnection_warning_is_valid_identity_provenance():
    identity = molecule_identity(smiles("C[Mg]Br"))

    assert identity.return_code == 1
    assert "Metal was disconnected" in identity.warnings
    assert identity.standard_inchi.startswith("InChI=1S/")
    assert len(identity.inchi_key) == 27


def test_direct_and_inchi_key_helpers_are_equivalent():
    molecule = smiles("C/C=C/C")
    inchi = molecule_to_inchi(molecule)
    rdkit_molecule = molecule.to_rdkit(keep_mapping=False)

    assert molecule_to_inchi_key(molecule) == rdinchi.InchiToInchiKey(inchi)
    assert inchi_to_inchi_key(inchi) == rdinchi.InchiToInchiKey(inchi)
    assert molecule_to_inchi_key(molecule) == rdkit_inchi.MolToInchiKey(rdkit_molecule)


def test_molecule_to_inchi_key_does_not_generate_intermediate_inchi(monkeypatch):
    monkeypatch.setattr(
        "synplan.chem.building_blocks.identity.rdinchi.MolToInchi",
        lambda _mol: pytest.fail("direct key generation must not materialize InChI"),
    )

    assert molecule_to_inchi_key(smiles("O")) == "XLYOFNOQVPJJNP-UHFFFAOYSA-N"


def test_identity_distinguishes_e_and_z_stereoisomers():
    trans = smiles("F/C=C/F")
    cis = smiles("F/C=C\\F")

    assert molecule_to_inchi_key(trans) != molecule_to_inchi_key(cis)


@pytest.mark.parametrize(
    "first,second",
    [
        ("CCO.[Na+]", "[Na+].OCC"),
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("O=c1[nH]cccc1", "Oc1ncccc1"),
        ("NCC(=O)O", "[NH3+]CC(=O)[O-]"),
    ],
)
def test_identity_normalizes_equivalent_component_aromatic_and_proton_forms(
    first, second
):
    assert molecule_to_inchi_key(smiles(first)) == molecule_to_inchi_key(smiles(second))


@pytest.mark.parametrize(
    "first,second",
    [
        ("CC(=O)O", "CC(=O)[O-]"),
        ("[13CH3]CO", "CCO"),
    ],
)
def test_identity_retains_charge_and_isotope_layers(first, second):
    assert molecule_to_inchi_key(smiles(first)) != molecule_to_inchi_key(smiles(second))


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
