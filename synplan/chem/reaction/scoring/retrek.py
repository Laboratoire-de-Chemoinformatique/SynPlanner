"""ReTReK reaction-level scoring functions and classes."""

import logging
import math

from chython import MoleculeContainer

from synplan.chem.reaction.scoring.base import (
    UNAVAILABLE,
    AbstractReactionScore,
    ReactionScoreContext,
)

logger = logging.getLogger(__name__)


def calculate_cdscore(
    product: MoleculeContainer,
    new_precursors: tuple[MoleculeContainer, ...],
) -> float:
    """Compute the Convergent Disconnection Score (CDScore) for a reaction.

    :param product: product molecule
    :param new_precursors: sequence of reactant molecules

    :returns: CDScore value, Range: (0, 1]
    """
    n = len(new_precursors)
    if n == 0:
        raise ValueError("At least one reactant is required.")
    if n == 1:
        return 0.0

    def heavy_atom_count(mol):
        return sum(1 for _, atom in mol.atoms() if atom.atomic_number != 1)

    a_P_n = heavy_atom_count(product) / n
    mae = sum(abs(a_P_n - heavy_atom_count(r)) for r in new_precursors) / n
    return 1.0 / (1.0 + mae)


def calculate_asscore(
    available_precursors: tuple[bool, ...],
) -> float:
    """Compute the Available Substances Score (ASScore) for a reaction.

    :param available_precursors: availability verdict for each precursor

    :returns: ASScore value, Range: [0, 1]
    """
    k = len(available_precursors)
    if k == 0:
        raise ValueError("At least one precursor is required.")
    return sum(available_precursors) / k


def calculate_rdscore(
    product: MoleculeContainer,
    new_precursors: tuple[MoleculeContainer, ...],
) -> float:
    """Compute the Ring Disconnection Score (RDScore) for a reaction.

    :param product: product molecule
    :param new_precursors: sequence of reactant molecules

    :returns: RDScore value, Range: {0, 1}
    """
    dP = product.rings_count
    dRi = sum(reactant.rings_count for reactant in new_precursors)
    return 1.0 if dP > dRi else 0.0


def calculate_stscore(
    rule,
    new_precursors: tuple[MoleculeContainer, ...],
) -> float:
    """Compute the Selectivity Transformation Score (STScore) for a reaction.

    :param rule: the canonical retro reactor
    :param new_precursors: sequence of precursor molecules

    :returns: STScore value, Range: (0, 1]
    """
    if rule is None:
        logger.warning("Rule is None. Cannot compute STScore.")
        return UNAVAILABLE

    if not rule._products:
        logger.warning("Rule has no products. Cannot compute STScore.")
        return UNAVAILABLE

    if len(rule._products) != len(new_precursors):
        logger.warning(
            "Number of products in the rule does not match the number of new precursors."
        )
        return UNAVAILABLE

    match_counts = []
    for pattern, reactant in zip(rule._products, new_precursors):
        n = sum(1 for _ in pattern.get_mapping(reactant))
        match_counts.append(n if n > 0 else 1)

    denom = 1
    for n in match_counts:
        denom *= n
    return 1.0 / denom


class CDScore(AbstractReactionScore):
    """Convergent Disconnection Score."""

    def compute(self, context: ReactionScoreContext) -> float:
        return calculate_cdscore(context.product, context.new_precursors)


class ASScore(AbstractReactionScore):
    """Available Substances Score."""

    def compute(self, context: ReactionScoreContext) -> float:
        if context.available_precursors is None:
            return UNAVAILABLE
        if len(context.available_precursors) != len(context.new_precursors):
            raise ValueError("available_precursors must align with new_precursors")
        return calculate_asscore(context.available_precursors)


class RDScore(AbstractReactionScore):
    """Ring Disconnection Score."""

    def compute(self, context: ReactionScoreContext) -> float:
        return calculate_rdscore(context.product, context.new_precursors)


class STScore(AbstractReactionScore):
    """Selectivity Transformation Score.

    .. warning::
        The current matching formula is provisional and must be corrected before
        the ReTReK work is published.
    """

    def compute(self, context: ReactionScoreContext) -> float:
        return calculate_stscore(context.rule, context.new_precursors)


def aggregate_retrek_score(
    context: ReactionScoreContext,
    scorers_and_weights: list[tuple[AbstractReactionScore, float]],
) -> float:
    """Compute the weighted mean of enabled ReTReK scores for a reaction.

    Scores returning UNAVAILABLE (NaN) are skipped in both the sum and
    the denominator. Returns UNAVAILABLE if all scores are unavailable.

    K = sum(w_i * score_i) / sum(w_i) over available, positive-weight scores.

    :param context: ReactionScoreContext carrying all reaction data.
    :param scorers_and_weights: List of (scorer, weight) pairs.
    :returns: Aggregated score as a float.
    """
    weighted_total = 0.0
    available_weight = 0.0
    for scorer, weight in scorers_and_weights:
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("ReTReK score weights must be finite and non-negative")
        if weight == 0.0:
            continue
        score = scorer.compute(context)
        if math.isnan(score):
            continue
        weighted_total += weight * score
        available_weight += weight
    if available_weight == 0.0:
        return UNAVAILABLE
    return weighted_total / available_weight
