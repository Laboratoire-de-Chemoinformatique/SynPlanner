import pytest
from chython import smiles

from synplan.chem.building_blocks import (
    BuildingBlockStock,
    coerce_building_block_stock,
    molecule_to_inchi,
    molecule_to_inchi_key,
)


def test_smiles_stock_membership_and_removal_are_immutable():
    molecule = smiles("OCC")
    stock = BuildingBlockStock(frozenset({"CCO", "CCN"}), "smiles")

    assert stock.key_for_molecule(molecule) == "CCO"
    assert stock.contains_molecule(molecule)
    reduced = stock.without_molecule(molecule)
    assert reduced == BuildingBlockStock(frozenset({"CCN"}), "smiles")
    assert len(stock) == 2
    assert set(stock) == {"CCO", "CCN"}


def test_inchikey_stock_uses_full_stereochemical_key():
    left = smiles("F[C@H](Cl)Br")
    right = smiles("F[C@@H](Cl)Br")
    stock = BuildingBlockStock(frozenset({molecule_to_inchi_key(left)}), "inchikey")

    assert stock.contains_molecule(left)
    assert not stock.contains_molecule(right)


def test_coerce_migrates_homogeneous_legacy_inchi_stock():
    molecule = smiles("O")
    stock = coerce_building_block_stock(frozenset({molecule_to_inchi(molecule)}))

    assert stock.identity_format == "inchikey"
    assert stock.keys == frozenset({molecule_to_inchi_key(molecule)})


def test_coerce_detects_existing_inchikeys_and_retains_typed_stock():
    key = molecule_to_inchi_key(smiles("O"))
    detected = coerce_building_block_stock({key})

    assert detected.identity_format == "inchikey"
    assert coerce_building_block_stock(detected) is detected


@pytest.mark.parametrize(
    "values",
    [
        {"InChI=1S/H2O/h1H2", "CCO"},
        {"XLYOFNOQVPJJNP-UHFFFAOYSA-N", "CCO"},
    ],
)
def test_coerce_rejects_mixed_legacy_representations(values):
    with pytest.raises(ValueError, match="mixes"):
        coerce_building_block_stock(values)


def test_stock_rejects_non_standard_inchikey():
    with pytest.raises(ValueError, match="Standard InChIKey"):
        BuildingBlockStock(frozenset({"UHOVQNZJYSORNB-UHFFFAOYNA-N"}), "inchikey")
