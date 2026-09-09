from __future__ import annotations

from inspect import Parameter, signature
from unittest.mock import Mock

import pytest
from chython import smiles
from chython.containers import MoleculeContainer

from synplan.chem.reaction import apply_reaction_rule
from synplan.chem.target_bonds import (
    _PROVENANCE_KEY,
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


def _apply(reactions, bonds_state=None, *, top_reactions_num=5, sort_reactions=False):
    target = reactions[0].reactants[0]
    return list(
        apply_reaction_rule(
            molecule=target,
            reaction_rule=FakeReactor(reactions),
            provenance=TargetAtomProvenance.for_target(target),
            constraints=TargetBondConstraints.from_state(target, bonds_state),
            top_reactions_num=top_reactions_num,
            sort_reactions=sort_reactions,
        )
    )


def test_apply_reaction_rule_rejects_reaction_breaking_frozen_bond():
    reaction = _breaking_reaction()
    applications = _apply([reaction], {(2, 3): 2})

    assert applications == []


def test_apply_reaction_rule_allows_reaction_breaking_selected_break_bond():
    reaction = _breaking_reaction()
    applications = _apply([reaction], {(2, 3): 1})

    assert applications == [list(reaction.products)]


def test_apply_reaction_rule_allows_reaction_that_does_not_break_frozen_bond():
    reaction = _breaking_reaction()
    applications = _apply([reaction], {(1, 2): 2})

    assert applications == [list(reaction.products)]


def test_apply_reaction_rule_matches_reversed_frozen_bond_key():
    reaction = _breaking_reaction()
    applications = _apply([reaction], {(3, 2): 2})

    assert applications == []


@pytest.mark.parametrize("sort_reactions", [False, True])
def test_apply_reaction_rule_keeps_later_allowed_candidate(sort_reactions):
    blocked = _breaking_reaction()
    allowed = _alternative_reaction()
    applications = _apply(
        [blocked, allowed],
        {(2, 3): 2},
        top_reactions_num=1,
        sort_reactions=sort_reactions,
    )

    assert applications == [list(allowed.products)]


def test_state_zero_uses_empty_provenance_fast_path():
    reaction = _breaking_reaction()
    applications = _apply([reaction], {(1, 2): 0})

    assert applications
    assert all(
        _PROVENANCE_KEY not in product.meta
        for products in applications
        for product in products
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
        apply_reaction_rule(
            molecule=target,
            reaction_rule=ShrinkingReactor(),
            provenance=provenance,
            constraints=constraints,
            multirule=True,
            rm_dup=True,
        )
    )

    assert [tuple(map(str, products)) for products in applications] == [
        ("CCC",),
        ("CC",),
    ]
    assert [
        products[0].meta[_PROVENANCE_KEY].as_dict() for products in applications
    ] == [
        {1: 1, 2: 2, 3: 3},
        {1: 1, 2: 2},
    ]


def test_apply_reaction_rule_preserves_existing_positional_interface():
    parameters = signature(apply_reaction_rule).parameters
    assert [
        name
        for name, p in parameters.items()
        if p.kind == Parameter.POSITIONAL_OR_KEYWORD
    ] == [
        "molecule",
        "reaction_rule",
        "sort_reactions",
        "top_reactions_num",
        "rebuild_with_cgr",
        "multirule",
        "rm_dup",
        "co_reactants",
        "max_mapping_work",
        "diagnostics",
    ]
    assert [parameters[name].default for name in list(parameters)[2:10]] == [
        False,
        5,
        False,
        False,
        False,
        (),
        100_000,
        None,
    ]
    assert all(
        parameters[name].kind == Parameter.KEYWORD_ONLY
        and parameters[name].default is None
        for name in ("provenance", "constraints")
    )
    reaction = _breaking_reaction()
    products = next(
        apply_reaction_rule(
            reaction.reactants[0],
            FakeReactor([reaction]),
            False,
            5,
            False,
            False,
            False,
            (),
            100_000,
            [],
        )
    )
    assert isinstance(products, list)
    assert all(isinstance(mol, MoleculeContainer) for mol in products)
    assert all(_PROVENANCE_KEY not in mol.meta for mol in products)


@pytest.mark.parametrize("rebuild_with_cgr", [False, True])
def test_constrained_products_carry_provenance_without_mutating_inputs(
    rebuild_with_cgr,
):
    reaction = _breaking_reaction()
    target = reaction.reactants[0]
    target.meta["source"] = "target"
    constraints = TargetBondConstraints.from_state(target, {(1, 2): 2})
    products = next(
        apply_reaction_rule(
            target,
            FakeReactor([reaction]),
            constraints=constraints,
            rebuild_with_cgr=rebuild_with_cgr,
        )
    )

    assert isinstance(products, list)
    assert all(isinstance(mol, MoleculeContainer) for mol in products)
    assert [mol.meta[_PROVENANCE_KEY].as_dict() for mol in products] == [
        {1: 1, 2: 2},
        {3: 3},
    ]
    assert target.meta == {"source": "target"}
    assert all(_PROVENANCE_KEY not in mol.meta for mol in reaction.products)


def test_multirule_reused_atom_numbers_do_not_recover_lost_provenance():
    target = smiles("[CH3:1][CH2:2][OH:3]")
    shrink = smiles("[CH3:1][CH2:2][OH:3]>>[CH3:1][CH3:2]")
    regrow = smiles("[CH3:1][CH3:2]>>[CH3:1][CH2:2][OH:3]")
    rule = Mock(side_effect=[[shrink], [regrow], []])
    constraints = TargetBondConstraints.from_state(target, {(1, 2): 2, (2, 3): 1})
    products = list(
        apply_reaction_rule(
            target,
            rule,
            constraints=constraints,
            multirule=True,
            rm_dup=True,
        )
    )

    assert len(products) == 2
    assert products[-1][0] == target
    assert all(
        group[0].meta[_PROVENANCE_KEY].as_dict() == {1: 1, 2: 2} for group in products
    )
    assert rule.call_count == 3
    assert _PROVENANCE_KEY not in target.meta

    # A subsequent call must use the carried map, not seed atom 3 as a target atom again.
    again = next(
        apply_reaction_rule(
            products[-1][0],
            FakeReactor([regrow]),
            constraints=constraints,
        )
    )
    assert again[0].meta[_PROVENANCE_KEY].as_dict() == {1: 1, 2: 2}
