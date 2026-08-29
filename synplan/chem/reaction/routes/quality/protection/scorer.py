"""Competing sites scorer.

Computes the S(T) competing sites score for synthesis routes. Blending it
with the search score and ordering routes on the result is
:class:`~synplan.chem.reaction.routes.quality.scorer.ProtectionRouteScorer`'s
job.

The scoring formula is inspired by Eq. 6 of:

    Westerlund et al., "Toward lab-ready AI synthesis plans with protection
    strategies and route scoring", *J. Chem. Inf. Model.* 2026, 66, 6361.
    https://doi.org/10.1021/acs.jcim.6c01147

We use a **worst-per-step** variant of the formula:
each step contributes only the penalty of its most severe interaction
(1.0 for incompatible, 0.5 for competing, 0.0 for compatible).
This avoids overwhelming the score when drug-like molecules contain
many functional groups that each trigger a matrix lookup.
"""

from chython.containers import ReactionContainer
from tqdm.auto import tqdm

from synplan.chem.reaction.routes.quality.protection.scanner import (
    CompetingInteraction,
    RouteScanner,
)


class CompetingSitesScore:
    """Score routes by their competing functional group burden.

    Uses a RouteScanner to detect interactions and then computes
    a worst-per-step score:

        For each step s, w_s = max severity penalty among interactions.
        S(T) = max[1 - (sum(w_s) + H) / max(N, 1), 0]

    where H is the halogen competing-site count and N is the number
    of steps.  Each step contributes at most 1.0 (incompatible) or
    0.5 (competing) to the penalty, preventing highly functionalized
    molecules from overwhelming the score.

    :param scanner: A RouteScanner instance configured with a
        FunctionalGroupDetector and IncompatibilityMatrix.
    """

    # Severity label → numeric penalty
    _SEVERITY_PENALTY = {
        "incompatible": 1.0,
        "competing": 0.5,
        "compatible": 0.0,
    }

    def __init__(self, scanner: RouteScanner):
        self._scanner = scanner

    def score_route(
        self, route: dict[int, ReactionContainer]
    ) -> tuple[float, list[CompetingInteraction]]:
        """Compute the S(T) score for a single route.

        :param route: A dict mapping step_id -> ReactionContainer.
        :return: Tuple of (score, interactions) where score is in [0, 1]
            and interactions is the list of CompetingInteraction objects.
        """
        interactions, halogen_count = self._scanner.scan_route(route)

        # Worst-per-step: each step contributes at most max(severity).
        step_worst: dict[int, float] = {}
        for inter in interactions:
            p = self._SEVERITY_PENALTY.get(inter.severity, 0.0)
            if inter.step_id not in step_worst or p > step_worst[inter.step_id]:
                step_worst[inter.step_id] = p

        # Use the number of steps the scanner *actually processed*, not the
        # full route length. Steps whose CGR fails to compose are silently
        # skipped inside the scanner (logged but not raised); using
        # ``len(route)`` as the denominator collapses a total scan failure
        # into a perfect score (S(T) = 1.0) which is indistinguishable
        # from a genuinely clean route.
        n_processed = len(self._scanner.last_processed_steps)
        if n_processed == 0 and len(route) > 0:
            # Scanner could not process *any* step of a non-empty route.
            # Return 0.0 so downstream ranking surfaces this as a problem
            # without raising and breaking caller batches.
            return 0.0, interactions
        n_reactions = max(n_processed, 1)
        penalty = (sum(step_worst.values()) + halogen_count) / n_reactions
        score = max(1.0 - penalty, 0.0)
        return score, interactions

    def rerank_routes(
        self,
        routes: dict[int, dict[int, ReactionContainer]],
        existing_scores: dict[int, float] | None = None,
        weight: float = 0.5,
        progress: bool = False,
    ) -> list[tuple[int, float, float, float]]:
        """Re-rank routes on a tunable mix of search score and protection score.

        combined = (1 - weight) * original_score_normalized + weight * S(T)

        **This is deliberately not the paper's formula, and it is not a
        duplicate of** :meth:`~synplan.chem.reaction.routes.quality.scorer.ProtectionRouteScorer.score`.
        The paper multiplies the two numbers, which lets either one veto a
        route and gives back a single figure you cannot take apart. This
        blends them additively over a normalised search score, so ``weight``
        dials between trusting the search and trusting the protection
        penalty, and the result carries its own components -- you can see
        *why* a route ranked where it did. Keep both: one reproduces the
        published method, this one is for looking into it.

        If no existing scores are provided, the original score component
        is treated as 0.0 for all routes and only the protection score
        is used.

        :param routes: Dict mapping route_id -> {step_id: ReactionContainer}.
        :param existing_scores: Optional dict mapping route_id -> original
            route score (e.g. from Tree.route_score()).
        :param weight: Weight of the protection score in [0, 1].
            :attr:`ProtectionConfig.score_weight` holds the configured default.
        :param progress: Show a progress bar. Scoring is a SMARTS scan per
            route and runs to tens of seconds over a few hundred of them.
        :return: List of (route_id, combined_score, protection_score,
            original_score) tuples, sorted descending by combined_score.
        """
        if existing_scores is None:
            existing_scores = {}

        # Normalize existing scores to [0, 1] if any are present
        if existing_scores:
            max_score = max(existing_scores.values())
            if max_score <= 0:
                max_score = 1.0
            norm_scores = {rid: s / max_score for rid, s in existing_scores.items()}
        else:
            norm_scores = {}

        results: list[tuple[int, float, float, float]] = []
        items = routes.items()
        if progress:
            items = tqdm(items, desc="re-ranking routes", unit="route")
        for route_id, route in items:
            protection_score, _ = self.score_route(route)
            original = norm_scores.get(route_id, 0.0)
            combined = (1.0 - weight) * original + weight * protection_score
            results.append(
                (
                    route_id,
                    combined,
                    protection_score,
                    existing_scores.get(route_id, 0.0),
                )
            )

        results.sort(key=lambda x: x[1], reverse=True)
        return results
