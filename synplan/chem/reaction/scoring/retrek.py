"""ReTReK reaction-level scoring functions and classes."""

import functools
import logging
import math
from itertools import permutations

from chython import MoleculeContainer, ReactionContainer

from synplan.chem.reaction.scoring.base import (
    UNAVAILABLE,
    AbstractReactionScore,
    ReactionScoreContext,
)

logger = logging.getLogger(__name__)


@functools.cache
def _product_reaction_center_atoms(rule) -> tuple[tuple[int, ...], ...]:
    """Mapped reaction-center atoms belonging to each rule product pattern."""
    cgr = ~ReactionContainer(rule._patterns, rule._products)
    if hasattr(cgr, "center_atoms"):
        center_atoms = set(cgr.center_atoms)
    else:
        # QueryCGRContainer does not expose CGRContainer.center_atoms. Mirror its
        # definition: dynamic bonds and changing charge/radical state define the
        # reaction center.
        center_atoms = {
            atom_id
            for atom_id, atom in cgr.atoms()
            if atom.charge != atom.p_charge or atom.is_radical != atom.p_is_radical
        }
        center_atoms.update(
            center_id
            for atom_id, neighbor_id, bond in cgr.bonds()
            if bond.is_dynamic
            for center_id in (atom_id, neighbor_id)
        )

    return tuple(
        tuple(sorted(center_atoms.intersection(pattern))) for pattern in rule._products
    )


def _distinct_reactive_site_count(
    pattern,
    precursor: MoleculeContainer,
    center_atoms: tuple[int, ...],
) -> int:
    """Count reaction-center placements modulo precursor automorphisms."""
    automorphisms = tuple(precursor.get_automorphism_mapping())
    distinct_sites = set()

    for mapping in pattern.get_mapping(precursor):
        signature = tuple(mapping[atom_id] for atom_id in center_atoms)
        equivalent_signatures = [signature]
        equivalent_signatures.extend(
            tuple(automorphism.get(atom_id, atom_id) for atom_id in signature)
            for automorphism in automorphisms
        )
        distinct_sites.add(min(equivalent_signatures))

    return len(distinct_sites)


def calculate_cdscore(
    product: MoleculeContainer,
    new_precursors: tuple[MoleculeContainer, ...],
    *,
    normalized_atom_contributions: bool = False,
) -> float:
    """Compute the Convergent Disconnection Score (CDScore) for a reaction.

    :param product: product molecule
    :param new_precursors: sequence of reactant molecules
    :param normalized_atom_contributions: use mapped heavy-atom contributions to
        the selected product and a size-independent normalized deviation. The
        original heavy-atom-count formula remains the default.

    :returns: CDScore value, Range: [0, 1]
    """
    n = len(new_precursors)
    if n == 0:
        raise ValueError("At least one reactant is required.")
    if n == 1:
        return 0.0

    def heavy_atom_count(mol):
        return sum(1 for _, atom in mol.atoms() if atom.atomic_number != 1)

    if normalized_atom_contributions:
        product_atoms = {
            atom_id for atom_id, atom in product.atoms() if atom.atomic_number != 1
        }
        contributed_atoms = []
        assigned_atoms = set()
        for precursor in new_precursors:
            contribution = product_atoms.intersection(
                atom_id
                for atom_id, atom in precursor.atoms()
                if atom.atomic_number != 1
            )
            overlap = assigned_atoms.intersection(contribution)
            if overlap:
                raise ValueError(
                    "Product atom mappings occur in more than one precursor: "
                    f"{sorted(overlap)}"
                )
            if contribution:
                contributed_atoms.append(len(contribution))
                assigned_atoms.update(contribution)

        if not contributed_atoms:
            raise ValueError(
                "No mapped heavy atoms are shared by the product and its precursors"
            )

        n = len(contributed_atoms)
        if n == 1:
            return 0.0

        total_contribution = sum(contributed_atoms)
        equal_fraction = 1.0 / n
        deviation = sum(
            abs(contribution / total_contribution - equal_fraction)
            for contribution in contributed_atoms
        )
        maximum_deviation = 2.0 * (1.0 - equal_fraction)
        return 1.0 - deviation / maximum_deviation

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
    *,
    distinct_reactive_sites: bool = True,
) -> float:
    """Compute the Selectivity Transformation Score (STScore) for a reaction.

    :param rule: the canonical retro reactor
    :param new_precursors: sequence of precursor molecules
    :param distinct_reactive_sites: Count symmetry-distinct reaction-center
        placements when true. When false, reproduce the original calculation
        from all complete substructure embeddings.

    Full substructure embeddings are projected onto the reaction-center atoms,
    then placements related by a precursor automorphism are counted once.

    :returns: Reciprocal of the product of distinct reactive-site counts,
        or UNAVAILABLE when a rule product does not match its precursor.
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

    if not distinct_reactive_sites:
        match_counts = []
        for pattern, precursor in zip(rule._products, new_precursors):
            count = sum(1 for _ in pattern.get_mapping(precursor))
            match_counts.append(count if count > 0 else 1)
        return 1.0 / math.prod(match_counts)

    center_atoms_by_product = _product_reaction_center_atoms(rule)
    site_counts = tuple(
        tuple(
            _distinct_reactive_site_count(pattern, precursor, center_atoms)
            for precursor in new_precursors
        )
        for pattern, center_atoms in zip(
            rule._products,
            center_atoms_by_product,
            strict=True,
        )
    )

    denominators = {
        math.prod(
            site_counts[pattern_index][precursor_index]
            for pattern_index, precursor_index in enumerate(assignment)
        )
        for assignment in permutations(range(len(new_precursors)))
        if all(
            site_counts[pattern_index][precursor_index]
            for pattern_index, precursor_index in enumerate(assignment)
        )
    }
    if not denominators:
        logger.warning(
            "Rule product patterns cannot be matched one-to-one with the "
            "precursors. Cannot compute STScore."
        )
        return UNAVAILABLE
    if len(denominators) > 1:
        logger.warning(
            "Rule products have ambiguous precursor assignments with different "
            "reactive-site counts. Cannot compute STScore."
        )
        return UNAVAILABLE

    return 1.0 / denominators.pop()


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
    """Selectivity Transformation Score."""

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
