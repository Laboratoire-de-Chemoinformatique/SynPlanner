"""Tests for the functional group and halogen detection modules."""

import pytest
from chython import smiles
from pydantic import ValidationError

from synplan.route_quality.protection.functional_groups import (
    FunctionalGroupDetector,
    FunctionalGroupMatch,
    HalogenDetector,
    HalogenMatch,
)
from synplan.route_quality.protection.config import ProtectionConfig


@pytest.fixture
def detector():
    """Create a FunctionalGroupDetector using the default competing_groups.yaml."""
    cfg = ProtectionConfig()
    return FunctionalGroupDetector(cfg.competing_groups_path)


@pytest.fixture
def halogen_detector():
    """Create a HalogenDetector using the default halogen_groups.yaml."""
    cfg = ProtectionConfig()
    return HalogenDetector(cfg.halogen_groups_path)


# --- FunctionalGroupDetector tests ---


def test_detect_alcohol_in_ethanol(detector):
    """Ethanol (CCO) should contain a PrimaryAlcoholAliphatic group."""
    mol = smiles("CCO")
    matches = detector.detect_all(mol)
    names = [m.name for m in matches]
    assert "PrimaryAlcoholAliphatic" in names


def test_detect_multiple_fgs_amino_acid(detector):
    """An amino acid (e.g. glycine) has amine, acid, and amino acid FGs."""
    mol = smiles("NCC(=O)O")
    matches = detector.detect_all(mol)
    names = {m.name for m in matches}
    assert "Amine_Primary_SaturatedAliphatic" in names
    assert "Acid_SaturatedAliphatic" in names


def test_detect_aldehyde(detector):
    """Acetaldehyde (CC=O) should be detected as an aldehyde."""
    mol = smiles("[CH3][CH]=O")
    matches = detector.detect_all(mol)
    names = [m.name for m in matches]
    assert "Aldehyde_SaturatedAliphatic" in names


def test_detect_competing_fgs_excludes_reaction_center(detector):
    """Competing FG detection should exclude groups at the reaction center."""
    # Molecule with alcohol and carboxylic acid
    mol = smiles("OCC(=O)O")
    all_matches = detector.detect_all(mol)
    # Use atom indices of acid group as "reaction center"
    acid_matches = [m for m in all_matches if m.name == "Acid_SaturatedAliphatic"]
    assert len(acid_matches) > 0
    center_atoms = set(acid_matches[0].atom_indices)
    competing = detector.detect_competing(mol, center_atoms)
    # The acid at the reaction center should be excluded
    for m in competing:
        assert not set(m.atom_indices) & center_atoms


def test_detect_competing_with_empty_center(detector):
    """With empty reaction center, all detected FGs are competing."""
    mol = smiles("OCC(=O)O")
    all_matches = detector.detect_all(mol)
    competing = detector.detect_competing(mol, reaction_center_atoms=set())
    assert len(competing) == len(all_matches)


def test_deduplication_of_symmetry_equivalent_matches(detector):
    """Symmetric molecules should not produce duplicate matches for the same atoms."""
    # 1,4-butanediol: HO-CH2-CH2-CH2-CH2-OH (two alcohol groups)
    mol = smiles("OCCCCO")
    matches = detector.detect_all(mol)
    alcohol_matches = [m for m in matches if m.name == "PrimaryAlcoholAliphatic"]
    # Should have exactly 2 alcohol matches (one per OH), not more
    assert len(alcohol_matches) == 2
    # The atom_indices should be different
    indices_set = {m.atom_indices for m in alcohol_matches}
    assert len(indices_set) == 2


def test_no_fgs_in_methane(detector):
    """Methane has no reactive functional groups."""
    mol = smiles("C")
    matches = detector.detect_all(mol)
    assert len(matches) == 0


def test_no_fgs_in_ethane(detector):
    """Ethane (CC) has no reactive functional groups."""
    mol = smiles("CC")
    matches = detector.detect_all(mol)
    assert len(matches) == 0


def test_match_dataclass_fields():
    """FunctionalGroupMatch should store name, category, and atom_indices."""
    m = FunctionalGroupMatch(
        name="hydroxyl", category="nucleophile", atom_indices=(1, 2)
    )
    assert m.name == "hydroxyl"
    assert m.category == "nucleophile"
    assert m.atom_indices == (1, 2)


def test_match_is_frozen():
    """FunctionalGroupMatch should be immutable (frozen model)."""
    m = FunctionalGroupMatch(
        name="hydroxyl", category="nucleophile", atom_indices=(1, 2)
    )
    with pytest.raises(ValidationError):
        m.name = "other"


def test_detector_with_custom_yaml(tmp_path):
    """Detector should load from a custom YAML config."""
    yaml_content = "custom_group:\n" "  - name: test_alkene\n" '    smarts: "C=C"\n'
    cfg_file = tmp_path / "fg.yaml"
    cfg_file.write_text(yaml_content)
    det = FunctionalGroupDetector(str(cfg_file))
    mol = smiles("C=CC")
    matches = det.detect_all(mol)
    names = [m.name for m in matches]
    assert "test_alkene" in names


def test_detector_skips_invalid_smarts(tmp_path):
    """Detector should skip entries with unparseable SMARTS and continue."""
    yaml_content = (
        "group:\n"
        "  - name: bad_pattern\n"
        '    smarts: "[INVALID"\n'
        "  - name: test_alkene\n"
        '    smarts: "C=C"\n'
    )
    cfg_file = tmp_path / "fg.yaml"
    cfg_file.write_text(yaml_content)
    det = FunctionalGroupDetector(str(cfg_file))
    mol = smiles("C=CC")
    matches = det.detect_all(mol)
    names = [m.name for m in matches]
    assert "bad_pattern" not in names
    assert "test_alkene" in names


def test_detect_reacting_fg(detector):
    """detect_reacting should return the FG overlapping with the reaction center."""
    mol = smiles("OCC(=O)O")
    all_matches = detector.detect_all(mol)
    acid_matches = [m for m in all_matches if m.name == "Acid_SaturatedAliphatic"]
    assert len(acid_matches) > 0
    center_atoms = set(acid_matches[0].atom_indices)
    reacting = detector.detect_reacting(mol, center_atoms)
    assert reacting is not None
    assert set(reacting.atom_indices) & center_atoms


def test_detect_reacting_fg_none_when_no_overlap(detector):
    """detect_reacting should return None when no FG overlaps the center."""
    mol = smiles("CCO")
    # Use atom indices that don't overlap any FG
    reacting = detector.detect_reacting(mol, {999})
    assert reacting is None


def test_caching(detector):
    """detect_all results should be cached."""
    mol = smiles("CCO")
    result1 = detector.detect_all(mol)
    result2 = detector.detect_all(mol)
    assert result1 is result2  # same object from cache


def test_clear_cache(detector):
    """clear_cache should invalidate the cache."""
    mol = smiles("CCO")
    result1 = detector.detect_all(mol)
    detector.clear_cache()
    result2 = detector.detect_all(mol)
    assert result1 is not result2  # new object after cache clear
    assert len(result1) == len(result2)


# --- HalogenDetector tests ---


def test_halogen_match_dataclass():
    """HalogenMatch should store name, family, and atom_indices."""
    m = HalogenMatch(name="aryl_bromide", family="bromide", atom_indices=(1, 2))
    assert m.name == "aryl_bromide"
    assert m.family == "bromide"
    assert m.atom_indices == (1, 2)


def test_halogen_match_is_frozen():
    """HalogenMatch should be immutable."""
    m = HalogenMatch(name="aryl_bromide", family="bromide", atom_indices=(1, 2))
    with pytest.raises(ValidationError):
        m.name = "other"


def test_detect_aryl_bromide(halogen_detector):
    """Bromobenzene should contain a bromide-family halogen."""
    mol = smiles("c1ccccc1Br")
    matches = halogen_detector.detect_all(mol)
    families = [m.family for m in matches]
    assert "bromide" in families


def test_detect_aryl_chloride(halogen_detector):
    """Chlorobenzene should contain a chloride-family halogen."""
    mol = smiles("c1ccccc1Cl")
    matches = halogen_detector.detect_all(mol)
    families = [m.family for m in matches]
    assert "chloride" in families


def test_detect_alkyl_bromide(halogen_detector):
    """Bromoethane should contain a bromide-family halogen."""
    mol = smiles("CCBr")
    matches = halogen_detector.detect_all(mol)
    families = [m.family for m in matches]
    assert "bromide" in families


def test_no_halogens_in_ethanol(halogen_detector):
    """Ethanol has no halogens."""
    mol = smiles("CCO")
    matches = halogen_detector.detect_all(mol)
    assert len(matches) == 0


def test_detect_competing_halogens(halogen_detector):
    """Competing halogens exclude the reaction center."""
    # Molecule with two bromines: BrCCBr
    mol = smiles("BrCCBr")
    all_matches = halogen_detector.detect_all(mol)
    # Should have at least 2 matches (may be more due to pattern overlap)
    assert len(all_matches) >= 2
    bromide_matches = [m for m in all_matches if m.family == "bromide"]
    assert len(bromide_matches) >= 2
    # Use one match's atom indices as "reaction center"
    center_atoms = set(bromide_matches[0].atom_indices)
    competing = halogen_detector.detect_competing_halogens(mol, center_atoms)
    # At least one should be competing (not overlapping center)
    assert len(competing) >= 1
    for m in competing:
        assert not set(m.atom_indices) & center_atoms


def test_count_same_family_competing_halogens(halogen_detector):
    """Same-family competing halogens should be counted."""
    # BrCCBr: both in "bromide" family
    mol = smiles("BrCCBr")
    all_matches = halogen_detector.detect_all(mol)
    bromide_matches = [m for m in all_matches if m.family == "bromide"]
    assert len(bromide_matches) >= 2
    # Use one match's atom indices as reaction center
    center_atoms = set(bromide_matches[0].atom_indices)
    count = halogen_detector.count_same_family_competing(mol, center_atoms)
    assert count >= 1


def test_count_different_family_competing_zero(halogen_detector):
    """Different-family competing halogens should not count."""
    # BrCCCl: one bromide and one chloride
    mol = smiles("BrCCCl")
    all_matches = halogen_detector.detect_all(mol)
    bromides = [m for m in all_matches if m.family == "bromide"]
    assert len(bromides) > 0
    # Use the bromide as reaction center
    center_atoms = set(bromides[0].atom_indices)
    count = halogen_detector.count_same_family_competing(mol, center_atoms)
    # The chloride is a different family -> not counted
    assert count == 0


def test_count_same_family_no_center_halogens(halogen_detector):
    """When no halogens at center, count should be 0."""
    mol = smiles("BrCCO")
    # Use oxygen atoms as reaction center (no halogen there)
    # Get the O atom index
    o_atoms = set()
    for n, atom in mol.atoms():
        if atom.atomic_symbol == "O":
            o_atoms.add(n)
    count = halogen_detector.count_same_family_competing(mol, o_atoms)
    assert count == 0


def test_halogen_detector_with_custom_yaml(tmp_path):
    """HalogenDetector should load from a custom YAML config."""
    yaml_content = "test_chloride:\n" '  smarts: "[Cl]C"\n' "  family: chloride\n"
    cfg_file = tmp_path / "hal.yaml"
    cfg_file.write_text(yaml_content)
    det = HalogenDetector(str(cfg_file))
    mol = smiles("CCCl")
    matches = det.detect_all(mol)
    names = [m.name for m in matches]
    assert "test_chloride" in names


def test_halogen_detector_skips_invalid_smarts(tmp_path):
    """HalogenDetector should skip entries with bad SMARTS."""
    yaml_content = (
        "bad_hal:\n"
        '  smarts: "[INVALID"\n'
        "  family: bad\n"
        "test_chloride:\n"
        '  smarts: "[Cl]C"\n'
        "  family: chloride\n"
    )
    cfg_file = tmp_path / "hal.yaml"
    cfg_file.write_text(yaml_content)
    det = HalogenDetector(str(cfg_file))
    mol = smiles("CCCl")
    matches = det.detect_all(mol)
    names = [m.name for m in matches]
    assert "bad_hal" not in names
    assert "test_chloride" in names
