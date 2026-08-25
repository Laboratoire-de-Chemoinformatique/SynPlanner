"""Tests for the reaction classifier module."""

import pytest
from chython import smiles

from synplan.chem.reaction.routes.quality.protection.reaction_classifier import (
    classify_reaction_type,
    classify_reaction_type_broad,
    classify_reaction_type_detailed,
    get_reaction_center_atoms,
)


@pytest.fixture(scope="module")
def esterification():
    """Fischer esterification: bond formation + bond breaking = substitution."""
    return smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH3:6]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH3:6].[OH2:4]"
    )


@pytest.fixture(scope="module")
def diels_alder():
    """Diels-Alder: pure bond formation (cycloaddition)."""
    return smiles(
        "[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH2:6]>>"
        "[CH2:1]1[CH:2]=[CH:3][CH2:4][CH2:5][CH2:6]1"
    )


@pytest.fixture(scope="module")
def bond_breaking_rxn():
    """Simple C-C bond breaking reaction."""
    return smiles("[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH3:1][CH3:2].[CH3:3][CH3:4]")


# --- get_reaction_center_atoms tests ---


def test_reaction_center_extraction_esterification(esterification):
    """Esterification should have reaction center atoms."""
    center = get_reaction_center_atoms(esterification)
    assert isinstance(center, set)
    assert len(center) > 0


def test_reaction_center_extraction_diels_alder(diels_alder):
    """Diels-Alder should have reaction center atoms."""
    center = get_reaction_center_atoms(diels_alder)
    assert isinstance(center, set)
    assert len(center) > 0


def test_reaction_center_no_change():
    """A no-change reaction should have an empty or minimal center."""
    rxn = smiles("[CH3:1][CH3:2]>>[CH3:1][CH3:2]")
    center = get_reaction_center_atoms(rxn)
    assert len(center) == 0


def test_reaction_center_with_precomputed_cgr(esterification):
    """get_reaction_center_atoms should accept a pre-computed CGR."""
    cgr = ~esterification
    center = get_reaction_center_atoms(esterification, cgr=cgr)
    assert isinstance(center, set)
    assert len(center) > 0


# --- classify_reaction_type (broad) tests ---


def test_classify_substitution(esterification):
    """Esterification involves both bond formation and breaking -> substitution."""
    rtype = classify_reaction_type(esterification)
    assert rtype == "substitution"


def test_classify_bond_formation(diels_alder):
    """Diels-Alder cycloaddition forms new bonds."""
    rtype = classify_reaction_type(diels_alder)
    assert rtype in ("bond_formation", "substitution")


def test_classify_bond_breaking(bond_breaking_rxn):
    """Pure bond breaking reaction."""
    rtype = classify_reaction_type(bond_breaking_rxn)
    assert rtype == "bond_breaking"


def test_classify_no_change():
    """A no-change reaction should be classified as 'other'."""
    rxn = smiles("[CH3:1][CH3:2]>>[CH3:1][CH3:2]")
    rtype = classify_reaction_type(rxn)
    assert rtype == "other"


def test_classify_sn2_substitution():
    """SN2 substitution: bond formed + bond broken."""
    rxn = smiles("[C-:3]#[N:4].[CH3:1][Br:2]>>[N:4]#[C:3][CH3:1].[Br-:2]")
    rtype = classify_reaction_type(rxn)
    assert rtype == "substitution"


def test_classify_returns_string():
    """classify_reaction_type should always return a string."""
    rxn = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH3:6]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH3:6].[OH2:4]"
    )
    result = classify_reaction_type(rxn)
    assert isinstance(result, str)
    assert result in ("bond_formation", "bond_breaking", "substitution", "other")


# --- classify_reaction_type_broad tests ---


def test_broad_classifier_matches_default(esterification):
    """classify_reaction_type and classify_reaction_type_broad should agree."""
    assert classify_reaction_type(esterification) == classify_reaction_type_broad(
        esterification
    )


# --- classify_reaction_type_detailed tests ---


def test_detailed_classifier_esterification(esterification):
    """New C-O bond next to a preserved carbonyl -> ester_formation."""
    assert classify_reaction_type_detailed(esterification) == "ester_formation"


def test_detailed_classifier_sn2():
    """SN2 with cyanide forms a C-C bond, which outranks the broken C-Br."""
    rxn = smiles("[C-:3]#[N:4].[CH3:1][Br:2]>>[N:4]#[C:3][CH3:1].[Br-:2]")
    assert classify_reaction_type_detailed(rxn) == "cross_coupling"


def test_detailed_classifier_dehalogenation():
    """C-Br broken without any C-halogen formed -> dehalogenation, not alkylation."""
    rxn = smiles("[CH3:1][Br:2].[OH2:3]>>[CH3:1][OH:3].[Br:2]")
    assert classify_reaction_type_detailed(rxn) == "dehalogenation"


def test_detailed_classifier_reduction():
    """Ketone C=O dropping to C-O is a net bond order decrease -> reduction."""
    rxn = smiles("[CH3:1][C:2](=[O:3])[CH3:4]>>[CH3:1][CH:2]([OH:3])[CH3:4]")
    assert classify_reaction_type_detailed(rxn) == "reduction"


def test_detailed_classifier_no_change():
    """No-change reaction should be 'other' in detailed classification too."""
    rxn = smiles("[CH3:1][CH3:2]>>[CH3:1][CH3:2]")
    rtype = classify_reaction_type_detailed(rxn)
    assert rtype == "other"


def test_detailed_accepts_precomputed_cgr(esterification):
    """A pre-computed CGR must give the same label as composing it internally."""
    cgr = ~esterification
    assert classify_reaction_type_detailed(esterification, cgr=cgr) == "ester_formation"
