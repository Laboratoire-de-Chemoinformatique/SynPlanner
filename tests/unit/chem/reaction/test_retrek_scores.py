"""Unit tests for reaction-level ReTReK scores."""

import math

import pytest
from chython import smiles

from synplan.chem.reaction.scoring import (
    ASScore,
    CDScore,
    RDScore,
    ReactionScoreContext,
    STScore,
    aggregate_retrek_score,
)


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
    # The STScore formula is intentionally not asserted here: its corrected
    # definition remains a blocker for publishing the ReTReK work.
    assert math.isnan(STScore().compute(_context()))


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
