from __future__ import annotations

from chython import smiles

from synplan.chem.reaction import apply_reaction_rule


class FakeReactor:
    def __init__(self, reactions):
        self.reactions = reactions

    def __call__(self, *reactants):
        return self.reactions


def test_apply_reaction_rule_rejects_reaction_breaking_frozen_bond():
    reaction = smiles("[CH3:1][CH2:2][CH3:3]>>[CH3:1][CH3:2].[CH4:3]")
    target = reaction.reactants[0]
    reactor = FakeReactor([reaction])

    products = list(
        apply_reaction_rule(
            target,
            reactor,
            validate_products=False,
            bonds_state={(2, 3): 2},
        )
    )

    assert products == []


def test_apply_reaction_rule_allows_reaction_breaking_selected_break_bond():
    reaction = smiles("[CH3:1][CH2:2][CH3:3]>>[CH3:1][CH3:2].[CH4:3]")
    target = reaction.reactants[0]
    reactor = FakeReactor([reaction])

    products = list(
        apply_reaction_rule(
            target,
            reactor,
            validate_products=False,
            bonds_state={(2, 3): 1},
        )
    )

    assert products == [list(reaction.products)]


def test_apply_reaction_rule_allows_reaction_that_does_not_break_frozen_bond():
    reaction = smiles("[CH3:1][CH2:2][CH3:3]>>[CH3:1][CH3:2].[CH4:3]")
    target = reaction.reactants[0]
    reactor = FakeReactor([reaction])

    products = list(
        apply_reaction_rule(
            target,
            reactor,
            validate_products=False,
            bonds_state={(1, 2): 2},
        )
    )

    assert products == [list(reaction.products)]
