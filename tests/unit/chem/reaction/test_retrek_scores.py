"""Unit tests for reaction-level ReTReK scores."""

import math

import pytest
from chython import smiles

from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.reaction.scoring import (
    ASScore,
    CDScore,
    RDScore,
    ReactionScoreContext,
    STScore,
    aggregate_retrek_score,
)
from synplan.chem.reaction.scoring.retrek import calculate_cdscore, calculate_stscore


def _mol(value: str):
    return smiles(value)


def _context(
    product="CCCC",
    precursors=("CC", "CC"),
    available_precursors=None,
):
    return ReactionScoreContext(
        product=_mol(product),
        new_precursors=tuple(_mol(value) for value in precursors),
        available_precursors=available_precursors,
    )


def test_cdscore_equal_heavy_atom_count_gives_one():
    assert CDScore().compute(_context()) == 1.0


def test_cdscore_one_precursor_gives_zero():
    assert CDScore().compute(_context("c1ccccc1", ("c1ccccc1",))) == 0.0


def test_cdscore_keeps_the_original_calculation_by_default():
    reaction = smiles("[C:1][C:2][C:3][C:4].[C:5][C:6]>>[C:1][C:2][C:3][C:4][C:5][C:6]")

    assert calculate_cdscore(
        reaction.products[0], tuple(reaction.reactants)
    ) == pytest.approx(0.5)


def test_cdscore_can_use_normalized_mapped_atom_contributions():
    reaction = smiles("[C:1][C:2][C:3][C:4].[C:5][C:6]>>[C:1][C:2][C:3][C:4][C:5][C:6]")

    assert calculate_cdscore(
        reaction.products[0],
        tuple(reaction.reactants),
        normalized_atom_contributions=True,
    ) == pytest.approx(2.0 / 3.0)


def test_normalized_cdscore_recognizes_the_four_component_ugi_reaction():
    reaction = smiles(
        "[CH3:1][CH:2]([CH3:3])[CH2:4][C@H:5]([CH2:6][C:7](=[O:8])"
        "[O:9][CH3:10])[C:11]([OH:12])=[O:13].[NH2:14][CH3:15]."
        "[cH:16]1[cH:17][cH:18][cH:19][cH:20][c:21]1-[c:22]2[cH:27]"
        "[cH:26][c:25]([cH:24][cH:23]2)[CH:28]=[O:29].[cH:34]1[cH:33]"
        "[cH:32][cH:31][cH:30][c:35]1[CH2:36][N+:37]#[C-:38]>>"
        "[CH3:1][CH:2]([CH3:3])[CH2:4][C@H:5]([CH2:6][C:7](=[O:8])"
        "[O:9][CH3:10])[C:11]([N:14]([CH3:15])[CH:28]([c:25]1[cH:24]"
        "[cH:23][c:22]([cH:27][cH:26]1)-[c:21]2[cH:16][cH:17][cH:18]"
        "[cH:19][cH:20]2)[C:38](=[O:12])[NH:37][CH2:36][c:35]3[cH:34]"
        "[cH:33][cH:32][cH:31][cH:30]3)=[O:13]"
    )

    assert calculate_cdscore(
        reaction.products[0], tuple(reaction.reactants)
    ) == pytest.approx(0.2)
    assert calculate_cdscore(
        reaction.products[0],
        tuple(reaction.reactants),
        normalized_atom_contributions=True,
    ) == pytest.approx(27.0 / 37.0)


def test_normalized_cdscore_ignores_noncontributing_precursors():
    reaction = smiles("[C:1][C:2].[Na+:3]>>[C:1][C:2]")

    assert (
        calculate_cdscore(
            reaction.products[0],
            tuple(reaction.reactants),
            normalized_atom_contributions=True,
        )
        == 0.0
    )


def test_normalized_cdscore_rejects_missing_atom_correspondence():
    product = smiles("[C:10][C:11]")
    precursors = (smiles("[C:1]"), smiles("[C:2]"))

    with pytest.raises(ValueError, match="No mapped heavy atoms"):
        calculate_cdscore(
            product,
            precursors,
            normalized_atom_contributions=True,
        )


def test_rdscore_ring_opening_gives_one():
    assert RDScore().compute(_context("C1CCCCC1", ("CCCCCC",))) == 1.0


def test_rdscore_non_ring_step_gives_zero():
    assert RDScore().compute(_context()) == 0.0


def test_asscore_uses_the_aligned_availability_verdict():
    assert ASScore().compute(_context(available_precursors=(True, False))) == 0.5


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (True, 1.0),
        (False, 0.0),
    ],
)
def test_asscore_scores_one_precursor_from_its_availability(available, expected):
    context = _context(precursors=("CC",), available_precursors=(available,))

    assert ASScore().compute(context) == expected


def test_asscore_rejects_an_empty_precursor_list():
    context = _context(precursors=(), available_precursors=())

    with pytest.raises(ValueError, match="At least one precursor"):
        ASScore().compute(context)


def test_asscore_is_unavailable_without_an_availability_verdict():
    assert math.isnan(ASScore().compute(_context()))


def test_asscore_rejects_a_verdict_of_the_wrong_length():
    with pytest.raises(ValueError, match="must align"):
        ASScore().compute(_context(available_precursors=(True,)))


def test_stscore_without_a_rule_is_unavailable():
    assert math.isnan(STScore().compute(_context()))


def test_stscore_counts_symmetry_distinct_reactive_sites():
    rule = CanonicalRetroReactor.from_smarts("[C:1]-[O:2]>>[C:1]")

    assert (
        STScore().compute(
            ReactionScoreContext(
                product=_mol("CCCO"),
                new_precursors=(_mol("CCC"),),
                rule=rule,
            )
        )
        == 0.5
    )


def test_stscore_rule_431_counts_two_distinct_reactive_sites():
    rule = CanonicalRetroReactor.from_smarts(
        "[c:1]:1(:[n:2]:[c:7](:[c:8](:[c:4]:1:[c:5])-[C:9])-[C:11]):"
        "[c:6]>>[C:7](-[C:8]-[C:9])(=[O:10])-[C:11]."
        "[c:1](-[N:2]-[N:3])(:[c:4]:[c:5]):[c:6]"
    )
    reaction = smiles(
        "[CH2:4]([C:2]([O:34][CH2:35][CH3:36])=[O:1])[CH:5]1[CH2:6]"
        "[CH2:7][CH2:8][C:9]1=[O:38].[cH:33]1[cH:32][c:14]([cH:13]"
        "[cH:12][c:11]1[NH:10][NH2:39])[O:15][CH3:37]>>[CH2:7]1[c:8]2"
        "[c:33]3[cH:32][c:14]([O:15][CH3:37])[cH:13][cH:12][c:11]3"
        "[nH:10][c:9]2[CH:5]([CH2:4][C:2]([O:34][CH2:35][CH3:36])=[O:1])"
        "[CH2:6]1"
    )
    context = ReactionScoreContext(
        product=reaction.products[0],
        new_precursors=tuple(reaction.reactants),
        rule=rule,
    )

    assert [
        sum(1 for _ in pattern.get_mapping(precursor))
        for pattern, precursor in zip(rule._products, reaction.reactants, strict=True)
    ] == [3, 2]
    assert STScore().compute(context) == 0.5
    assert calculate_stscore(
        rule,
        tuple(reaction.reactants),
        distinct_reactive_sites=False,
    ) == pytest.approx(1.0 / 6.0)
    assert (
        STScore().compute(
            ReactionScoreContext(
                product=reaction.products[0],
                new_precursors=tuple(reversed(reaction.reactants)),
                rule=rule,
            )
        )
        == 0.5
    )


def test_stscore_is_unavailable_when_a_rule_product_does_not_match():
    rule = CanonicalRetroReactor.from_smarts("[C:1]-[O:2]>>[C:1]")
    context = ReactionScoreContext(
        product=_mol("NO"),
        new_precursors=(_mol("N"),),
        rule=rule,
    )

    assert math.isnan(STScore().compute(context))


def test_aggregate_is_a_normalized_weighted_mean():
    context = _context("C1CCCCC1", ("CCC", "CCC"))

    result = aggregate_retrek_score(
        context,
        [(CDScore(), 5.0), (RDScore(), 2.0)],
    )

    expected = (5.0 * CDScore().compute(context) + 2.0) / 7.0
    assert result == pytest.approx(expected)
    assert 0.0 <= result <= 1.0


def test_aggregate_removes_unavailable_scores_from_both_sides():
    context = _context()

    with_st = aggregate_retrek_score(
        context,
        [(CDScore(), 5.0), (STScore(), 2.0)],
    )

    assert with_st == CDScore().compute(context)


def test_aggregate_is_unavailable_when_every_score_is_unavailable():
    assert math.isnan(aggregate_retrek_score(_context(), [(STScore(), 2.0)]))


@pytest.mark.parametrize("weight", [-1.0, math.inf, math.nan])
def test_aggregate_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match="finite and non-negative"):
        aggregate_retrek_score(_context(), [(CDScore(), weight)])
