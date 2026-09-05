import logging

import pytest
from chython import smiles
from chython.exceptions import InvalidAromaticRing
from frozendict import frozendict

from synplan.chem.building_blocks import BuildingBlock, molecule_to_inchikey
from synplan.chem.precursor import Precursor


def test_len_eq_hash(simple_molecule):
    p1 = Precursor(simple_molecule)
    p2 = Precursor(simple_molecule)
    assert len(p1) == 3
    assert p1 == p2
    assert hash(p1) == hash(p2)


def test_is_building_block_default(simple_molecule):
    p = Precursor(simple_molecule)
    # default min_mol_size=6, so anything ≤6 is a BB
    assert p.is_building_block(bb_stock=set())


def test_is_building_block_custom_size(simple_molecule, complex_molecule):
    # Test with custom min_mol_size
    p_small = Precursor(simple_molecule)
    p_large = Precursor(complex_molecule)

    # Small molecule should be BB with min_mol_size=10
    assert p_small.is_building_block(bb_stock=set(), min_mol_size=10)
    # Large molecule should not be BB with min_mol_size=10
    assert not p_large.is_building_block(bb_stock=set(), min_mol_size=10)


def test_is_building_block_with_stock(simple_molecule, complex_molecule):
    # Test with predefined building block stock
    p1 = Precursor(simple_molecule)
    p2 = Precursor(complex_molecule)

    # Add complex molecule to stock
    bb = complex_molecule.copy()
    bb.canonicalize()
    bb.clean_stereo()
    stock = {str(bb)}
    assert not p1.is_building_block(bb_stock=stock, min_mol_size=0)
    assert p2.is_building_block(bb_stock=stock, min_mol_size=0)


def test_ring_molecule_handling(ring_molecule):
    p = Precursor(ring_molecule)
    assert len(p) == 6
    assert p.is_building_block(bb_stock=set())  # Should be BB as size ≤6


def test_precursor_canonicalizes_molecule():
    # Kekule benzene must be stored aromatized, unless canonicalization is off
    assert str(Precursor(smiles("C1=CC=CC=C1"))) == "c1ccccc1"
    assert str(Precursor(smiles("C1=CC=CC=C1"), canonicalize=False)) == "C1=CC=CC=C1"


def test_precursor_inequality(simple_molecule, complex_molecule):
    p1 = Precursor(simple_molecule)
    p2 = Precursor(complex_molecule)
    assert p1 != p2
    assert hash(p1) != hash(p2)


def test_precursor_with_invalid_input():
    with pytest.raises(Exception):  # noqa: B017
        Precursor(None)


def _stereo_block(smiles_value: str) -> BuildingBlock:
    molecule = smiles(smiles_value, ignore_stereo=False)
    return BuildingBlock(
        smiles=str(molecule),
        inchikey=molecule_to_inchikey(molecule),
        vendors=frozendict({"vendor": 1.0}),
        has_stereo=True,
    )


def test_inchikey_membership_is_connectivity_only():
    r_block = _stereo_block("C[C@H](O)C(=O)O")
    s_precursor = Precursor(smiles("C[C@@H](O)C(=O)O", ignore_stereo=False))
    catalogue = frozendict({r_block.inchikey[:14]: (r_block,)})

    assert s_precursor.is_building_block(catalogue, min_mol_size=0)
    assert not any(atom.stereo is not None for _, atom in s_precursor.molecule.atoms())
    assert s_precursor.inchi_key[:14] == r_block.inchikey[:14]
    assert s_precursor.inchi_key != r_block.inchikey


def test_precursor_generates_its_inchikey_only_once(monkeypatch):
    import synplan.chem.precursor as precursor_module

    block = _stereo_block("C[C@H](O)C(=O)O")
    catalogue = frozendict({block.inchikey[:14]: (block,)})
    original = precursor_module.molecule_to_inchikey
    calls = 0

    def counted(molecule):
        nonlocal calls
        calls += 1
        return original(molecule)

    monkeypatch.setattr(precursor_module, "molecule_to_inchikey", counted)
    precursor = Precursor(smiles("C[C@H](O)C(=O)O", ignore_stereo=False))
    for _ in range(3):
        assert precursor.is_building_block(catalogue, min_mol_size=0)

    assert calls == 1


def test_unrepresentable_inchikey_is_cached_and_not_purchasable(monkeypatch, caplog):
    import synplan.chem.precursor as precursor_module

    molecule = smiles("c1c(O)[n-]ccc1", ignore=True)
    precursor = Precursor(molecule)
    original = precursor_module.molecule_to_inchikey
    calls = 0

    def counted(candidate):
        nonlocal calls
        calls += 1
        return original(candidate)

    monkeypatch.setattr(precursor_module, "molecule_to_inchikey", counted)
    catalogue = frozendict()
    with caplog.at_level(logging.WARNING):
        assert not precursor.is_building_block(catalogue, min_mol_size=0)
        assert not precursor.is_building_block(catalogue, min_mol_size=0)

    assert calls == 1
    assert caplog.text.count("treating it as not purchasable") == 1
    with pytest.raises(InvalidAromaticRing):
        _ = precursor.inchi_key
