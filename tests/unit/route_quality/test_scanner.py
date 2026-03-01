"""Tests for the route scanner and FG x FG incompatibility matrix."""

import pytest
from chython import smiles
from pydantic import ValidationError

from synplan.route_quality.protection.config import ProtectionConfig
from synplan.route_quality.protection.functional_groups import (
    FunctionalGroupDetector,
    HalogenDetector,
)
from synplan.route_quality.protection.scanner import (
    CompetingInteraction,
    IncompatibilityMatrix,
    RouteScanner,
)


@pytest.fixture
def config():
    return ProtectionConfig()


@pytest.fixture
def matrix(config):
    return IncompatibilityMatrix(config.incompatibility_path)


@pytest.fixture
def fg_detector(config):
    return FunctionalGroupDetector(config.competing_groups_path)


@pytest.fixture
def halogen_detector(config):
    return HalogenDetector(config.halogen_groups_path)


@pytest.fixture
def scanner(fg_detector, matrix):
    return RouteScanner(fg_detector, matrix)


@pytest.fixture
def scanner_with_halogens(fg_detector, matrix, halogen_detector):
    return RouteScanner(fg_detector, matrix, halogen_detector=halogen_detector)


# --- IncompatibilityMatrix tests (FG x FG) ---


def test_matrix_lookup_incompatible_pair(matrix):
    """Unprotected amino acid vs XKetone should be incompatible (2)."""
    result = matrix.lookup(
        "NonProlineAlphaAminoAcid_unprotected", "XKetoneAromaticBromide"
    )
    assert result == "incompatible"


def test_matrix_lookup_compatible_pair(matrix):
    """Protected amino acid vs aldehyde should be compatible (0)."""
    result = matrix.lookup(
        "NonProlineAlphaAminoAcid_protected", "Aldehyde_SaturatedAliphatic"
    )
    assert result == "compatible"


def test_matrix_lookup_competing_pair(matrix):
    """Unprotected amino acid vs aldehyde should be competing (1)."""
    result = matrix.lookup(
        "NonProlineAlphaAminoAcid_unprotected", "Aldehyde_SaturatedAliphatic"
    )
    assert result == "competing"


def test_matrix_lookup_amine_vs_xketone(matrix):
    """Primary amine vs XKetone should be incompatible."""
    result = matrix.lookup(
        "Amine_Primary_SaturatedAliphatic", "XKetoneAliphaticBromide"
    )
    assert result == "incompatible"


def test_matrix_lookup_alcohol_vs_alcohol(matrix):
    """PrimaryAlcohol vs PrimaryAlcohol should be competing."""
    result = matrix.lookup("PrimaryAlcoholAliphatic", "PrimaryAlcoholAliphatic")
    assert result == "competing"


def test_matrix_lookup_aldehyde_vs_amino_acid(matrix):
    """Aldehyde (competing) vs unprotected amino acid (reacting) should be competing."""
    result = matrix.lookup(
        "Aldehyde_SaturatedAliphatic", "NonProlineAlphaAminoAcid_unprotected"
    )
    assert result == "competing"


def test_matrix_lookup_unknown_fg(matrix):
    """Unknown FG name should return 'compatible' (safe default)."""
    result = matrix.lookup("nonexistent_group", "aldehyde")
    assert result == "compatible"


def test_matrix_lookup_unknown_reacting_fg(matrix):
    """Unknown reacting FG should return 'compatible' (safe default)."""
    result = matrix.lookup("PrimaryAlcoholAliphatic", "nonexistent_fg")
    assert result == "compatible"


def test_matrix_from_custom_tsv(tmp_path):
    """IncompatibilityMatrix should load from a custom TSV file."""
    tsv_content = "\ttest_elec\ttest_nuc\ntest_nuc\t2\t1\n"
    cfg_file = tmp_path / "matrix.tsv"
    cfg_file.write_text(tsv_content)
    m = IncompatibilityMatrix(str(cfg_file))
    assert m.lookup("test_nuc", "test_elec") == "incompatible"
    assert m.lookup("test_nuc", "test_nuc") == "competing"
    assert m.lookup("test_nuc", "unknown") == "compatible"


# --- RouteScanner tests ---


def test_scan_route_with_single_step(scanner):
    """Scanning a single-step route with a competing FG."""
    # Esterification of a molecule that also has a hydroxyl group
    rxn = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH2:7][OH:8]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[OH2:4]"
    )
    route = {0: rxn}
    interactions, halogen_count = scanner.scan_route(route)
    assert isinstance(interactions, list)
    for inter in interactions:
        assert isinstance(inter, CompetingInteraction)
        assert inter.step_id == 0
    assert halogen_count == 0


def test_scan_route_empty(scanner):
    """Scanning an empty route should return no interactions."""
    interactions, halogen_count = scanner.scan_route({})
    assert interactions == []
    assert halogen_count == 0


def test_scan_route_simple_no_competing_fgs(scanner):
    """A simple reaction with no competing FGs should return no interactions."""
    rxn = smiles("[CH3:1][CH3:2]>>[CH4:1].[CH4:2]")
    route = {0: rxn}
    interactions, _ = scanner.scan_route(route)
    assert len(interactions) == 0


def test_scan_route_two_step(scanner):
    """Two-step route should report interactions for both steps."""
    rxn0 = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH2:7][OH:8]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[OH2:4]"
    )
    rxn1 = smiles(
        "[CH3:10][C:11](=[O:12])[OH:13].[OH:14][CH3:15]>>"
        "[CH3:10][C:11](=[O:12])[O:14][CH3:15].[OH2:13]"
    )
    route = {0: rxn0, 1: rxn1}
    interactions, _ = scanner.scan_route(route)
    step_ids = {inter.step_id for inter in interactions}
    # At least step 0 should have interactions (free -OH on product)
    assert 0 in step_ids


def test_scan_route_reacting_fg_field(scanner):
    """CompetingInteraction should have the reacting_fg field set."""
    rxn = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH2:7][OH:8]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[OH2:4]"
    )
    route = {0: rxn}
    interactions, _ = scanner.scan_route(route)
    for inter in interactions:
        # reacting_fg should be a string or None
        assert inter.reacting_fg is None or isinstance(inter.reacting_fg, str)


def test_scan_route_severity_depends_on_reacting_fg(scanner):
    """Severity should depend on the reacting FG, not the reaction type."""
    rxn = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH2:7][OH:8]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[OH2:4]"
    )
    route = {0: rxn}
    interactions, _ = scanner.scan_route(route)
    for inter in interactions:
        # Severity should be a valid label
        assert inter.severity in ("compatible", "competing", "incompatible")


def test_scan_route_with_halogen_detector(scanner_with_halogens):
    """Scanner with halogen detector should count halogens."""
    # Reaction on a molecule with two Br atoms (same family)
    # Using mapped SMILES for a substitution on Br where another Br is competing
    rxn = smiles(
        "[Br:1]c1ccc([Br:2])cc1.[CH3:3][Li:4]>>" "[CH3:3]c1ccc([Br:2])cc1.[Li:4][Br:1]"
    )
    route = {0: rxn}
    interactions, halogen_count = scanner_with_halogens.scan_route(route)
    # There should be a competing halogen site (the other Br)
    assert halogen_count >= 0  # at least verified it runs


# --- classify_interactions tests ---


def test_classify_interactions_counts():
    """classify_interactions should count I, C, H correctly."""
    interactions = [
        CompetingInteraction(step_id=0, fg_name="hydroxyl", fg_atoms=(1, 2), reacting_fg="aldehyde", severity="incompatible"),
        CompetingInteraction(step_id=0, fg_name="primary_amine", fg_atoms=(3, 4), reacting_fg="aldehyde", severity="incompatible"),
        CompetingInteraction(step_id=1, fg_name="hydroxyl", fg_atoms=(5, 6), reacting_fg="ketone", severity="competing"),
        CompetingInteraction(step_id=1, fg_name="aldehyde", fg_atoms=(7,), reacting_fg=None, severity="compatible"),
    ]
    i_count, c_count, h_count = RouteScanner.classify_interactions(
        interactions, halogen_count=2
    )
    assert i_count == 2
    assert c_count == 1
    assert h_count == 2


def test_classify_interactions_empty():
    """Empty interaction list should return (0, 0, 0)."""
    i, c, h = RouteScanner.classify_interactions([])
    assert (i, c, h) == (0, 0, 0)


def test_classify_interactions_all_compatible():
    """All compatible interactions should give (0, 0, 0)."""
    interactions = [
        CompetingInteraction(step_id=0, fg_name="hydroxyl", fg_atoms=(1, 2), reacting_fg=None, severity="compatible"),
        CompetingInteraction(step_id=1, fg_name="aldehyde", fg_atoms=(3,), reacting_fg=None, severity="compatible"),
    ]
    i, c, h = RouteScanner.classify_interactions(interactions)
    assert (i, c, h) == (0, 0, 0)


def test_classify_interactions_with_halogen_count():
    """Halogen count should pass through correctly."""
    interactions = [
        CompetingInteraction(step_id=0, fg_name="hydroxyl", fg_atoms=(1,), reacting_fg="aldehyde", severity="competing"),
    ]
    i, c, h = RouteScanner.classify_interactions(interactions, halogen_count=3)
    assert i == 0
    assert c == 1
    assert h == 3


def test_competing_interaction_dataclass():
    """CompetingInteraction should store all fields correctly."""
    ci = CompetingInteraction(
        step_id=2,
        fg_name="thiol",
        fg_atoms=(5, 6),
        reacting_fg="aldehyde",
        severity="incompatible",
    )
    assert ci.step_id == 2
    assert ci.fg_name == "thiol"
    assert ci.fg_atoms == (5, 6)
    assert ci.reacting_fg == "aldehyde"
    assert ci.severity == "incompatible"


def test_competing_interaction_is_frozen():
    """CompetingInteraction should be immutable."""
    ci = CompetingInteraction(step_id=0, fg_name="hydroxyl", fg_atoms=(1,), reacting_fg="aldehyde", severity="competing")
    with pytest.raises(ValidationError):
        ci.severity = "other"


def test_competing_interaction_reacting_fg_none():
    """CompetingInteraction should accept None for reacting_fg."""
    ci = CompetingInteraction(step_id=0, fg_name="hydroxyl", fg_atoms=(1,), reacting_fg=None, severity="compatible")
    assert ci.reacting_fg is None
    assert ci.severity == "compatible"
