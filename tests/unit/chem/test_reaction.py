from __future__ import annotations

import pytest
from chython import smiles

from synplan.chem.reaction import apply_reaction_rule
from synplan.chem.reaction.reactor import iter_reaction_applications
from synplan.chem.target_bonds import (
    TargetAtomProvenance,
    TargetBondConstraints,
)


class FakeReactor:
    def __init__(self, reactions):
        self.reactions = reactions

    def __call__(self, *reactants):
        return self.reactions


class ShrinkingReactor:
    def __call__(self, reactant):
        if len(reactant) == 4:
            return [smiles("[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH3:1][CH2:2][CH3:3]")]
        if len(reactant) == 3:
            return [smiles("[CH3:1][CH2:2][CH3:3]>>[CH3:1][CH3:2]")]
        return []


def _breaking_reaction():
    return smiles("[CH3:1][CH2:2][CH3:3]>>[CH3:1][CH3:2].[CH4:3]")


def _alternative_reaction():
    return smiles("[CH3:1][CH2:2][CH3:3]>>[CH4:1].[CH3:2][CH3:3]")


def _apply(reactions, bonds_state=None, *, top_reactions_num=5):
    target = reactions[0].reactants[0]
    return list(
        iter_reaction_applications(
            molecule=target,
            reaction_rule=FakeReactor(reactions),
            provenance=TargetAtomProvenance.for_target(target),
            constraints=TargetBondConstraints.from_state(target, bonds_state),
            top_reactions_num=top_reactions_num,
        )
    )


def test_apply_reaction_rule_rejects_reaction_breaking_frozen_bond():
    reaction = _breaking_reaction()
    applications = _apply([reaction], {(2, 3): 2})

    assert applications == []


def test_apply_reaction_rule_allows_reaction_breaking_selected_break_bond():
    reaction = _breaking_reaction()
    applications = _apply([reaction], {(2, 3): 1})

    assert [application.products for application in applications] == [
        tuple(reaction.products)
    ]


def test_apply_reaction_rule_allows_reaction_that_does_not_break_frozen_bond():
    reaction = _breaking_reaction()
    applications = _apply([reaction], {(1, 2): 2})

    assert [application.products for application in applications] == [
        tuple(reaction.products)
    ]


def test_apply_reaction_rule_matches_reversed_frozen_bond_key():
    reaction = _breaking_reaction()
    applications = _apply([reaction], {(3, 2): 2})

    assert applications == []


def test_apply_reaction_rule_keeps_later_allowed_candidate():
    blocked = _breaking_reaction()
    allowed = _alternative_reaction()
    applications = _apply([blocked, allowed], {(2, 3): 2}, top_reactions_num=1)

    assert [application.products for application in applications] == [
        tuple(allowed.products)
    ]


def test_state_zero_uses_empty_provenance_fast_path():
    reaction = _breaking_reaction()
    applications = _apply([reaction], {(1, 2): 0})

    assert applications
    assert all(
        not provenance.pairs
        for application in applications
        for provenance in application.provenances
    )


def test_apply_reaction_rule_top_zero_yields_no_candidates():
    reaction = _breaking_reaction()

    assert (
        list(
            apply_reaction_rule(
                reaction.reactants[0], FakeReactor([reaction]), top_reactions_num=0
            )
        )
        == []
    )


def test_apply_reaction_rule_rejects_negative_top_limit():
    reaction = _breaking_reaction()

    with pytest.raises(ValueError, match="cannot be negative"):
        list(
            apply_reaction_rule(
                reaction.reactants[0], FakeReactor([reaction]), top_reactions_num=-1
            )
        )


def test_multirule_applications_carry_provenance_between_steps():
    target = smiles("[CH3:1][CH2:2][CH2:3][CH3:4]")
    provenance = TargetAtomProvenance.for_target(target)
    constraints = TargetBondConstraints.from_state(target, {(1, 2): 2})

    applications = list(
        iter_reaction_applications(
            molecule=target,
            reaction_rule=ShrinkingReactor(),
            provenance=provenance,
            constraints=constraints,
            multirule=True,
            rm_dup=True,
        )
    )

    assert [tuple(map(str, application.products)) for application in applications] == [
        ("CCC",),
        ("CC",),
    ]
    assert [application.provenances[0].as_dict() for application in applications] == [
        {1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2},
    ]
