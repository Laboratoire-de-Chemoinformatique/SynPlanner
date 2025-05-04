import pytest
from CGRtools import smiles
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


def test_precursor_from_reaction_components(sample_reactions):
    # Test creating precursors from reaction components
    for rxn_smiles in sample_reactions:
        # Split reaction into components
        reactants, _, products = rxn_smiles.split(">")
        # Create precursor from first reactant
        reactant_mol = smiles(reactants.split(".")[0])
        p = Precursor(reactant_mol)
        assert p is not None
        assert len(p) > 0


def test_precursor_inequality(simple_molecule, complex_molecule):
    p1 = Precursor(simple_molecule)
    p2 = Precursor(complex_molecule)
    assert p1 != p2
    assert hash(p1) != hash(p2)


def test_precursor_with_invalid_input():
    with pytest.raises(Exception):
        Precursor(None)
