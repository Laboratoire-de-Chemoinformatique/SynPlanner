from CGRtools import smiles
from synplan.chem.data.filtering import (
    CCsp3BreakingFilter,
    DynamicBondsFilter,
    NoReactionFilter,
    WrongCHBreakingFilter,
    CCRingBreakingFilter,
)


def test_ccsp3_no_break(sample_reactions):
    # Test first reaction from sample_reactions which should not have C(sp3)-C bond breaking
    rxn = smiles(sample_reactions[0])  # Fischer esterification
    f = CCsp3BreakingFilter()
    assert not f(rxn)


def test_ccsp3_with_break():
    # Test a reaction with C(sp3)-C bond breaking
    rxn = smiles(
        "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH3:1][CH2:2][CH3:3].[CH3:4]"
    )  # Breaking C-C bond in butane
    f = CCsp3BreakingFilter()
    assert f(rxn)


def test_dynamic_bonds_in_range():
    # Test a reaction with number of dynamic bonds within range
    rxn = smiles("[CH2:1]=[O:2]>>[CH2:1]-[O:2]")  # 1 dynamic bond
    f = DynamicBondsFilter(min_bonds_number=1, max_bonds_number=2)
    assert not f(rxn)


def test_dynamic_bonds_too_many():
    # Test a reaction with too many dynamic bonds
    rxn = smiles(
        "[CH2:1]=[CH:2]-[CH:3]=[CH:4]-[CH:5]=[O:6]>>[CH2:1]-[CH2:2]-[CH2:3]-[CH2:4]-[CH2:5]-[O:6]"
    )  # 5 dynamic bonds
    f = DynamicBondsFilter(min_bonds_number=1, max_bonds_number=2)
    assert f(rxn)


def test_dynamic_bonds_too_few():
    # Test a reaction with too few dynamic bonds
    rxn = smiles("[CH3:1]-[CH3:2]>>[CH3:1]-[CH3:2]")  # No dynamic bonds
    f = DynamicBondsFilter(min_bonds_number=1, max_bonds_number=2)
    assert f(rxn)


def test_no_reaction():
    # Test a reaction that doesn't change anything
    rxn = smiles("[CH3:1]-[CH3:2]>>[CH3:1]-[CH3:2]")  # No changes
    f = NoReactionFilter()
    assert f(rxn)


def test_wrong_ch_breaking():
    # Test a reaction with incorrect C-H bond breaking and C-C bond formation
    rxn = smiles(
        "[CH3:1][CH2:2][H:3]>>[CH3:1][CH2:2][CH3:4]"
    )  # Breaking C-H and forming C-C incorrectly
    f = WrongCHBreakingFilter()
    assert f(rxn)


def test_cc_ring_breaking():
    # Test a reaction with C-C ring breaking
    rxn = smiles(
        "[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5]1>>[CH2:1]-[CH2:2]-[CH2:3]-[CH2:4]-[CH2:5]"
    )  # Breaking cyclopentane ring
    f = CCRingBreakingFilter()
    assert f(rxn)


def test_cc_ring_no_breaking():
    # Test a reaction without C-C ring breaking but with other changes
    rxn = smiles(
        "[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5]1.[O:6]>>[CH2:1]1[CH2:2][CH2:3][CH2:4][CH2:5]1.[OH:6]"
    )  # No ring breaking, just adding H to O
    f = CCRingBreakingFilter()
    assert not f(rxn)
