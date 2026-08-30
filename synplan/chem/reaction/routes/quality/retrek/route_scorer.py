"""ReTReK route-level scorer."""

import math
from collections.abc import Callable

from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.reaction.routes.quality.retrek.config import RetrekRouteScoringConfig
from synplan.chem.reaction.routes.quality.scorer import RouteScorer
from synplan.chem.reaction.routes.route import Route, Step
from synplan.chem.reaction.scoring import (
    ASScore,
    CDScore,
    RDScore,
    ReactionScoreContext,
    STScore,
    aggregate_retrek_score,
)
from synplan.chem.utils import molecule_key

_SCORERS = {
    "cd": CDScore,
    "as": ASScore,
    "rd": RDScore,
    "st": STScore,
}


class RetrekRouteScorer(RouteScorer):
    """Route scorer based on ReTReK per-reaction scores.

    Walks :attr:`Route.steps`, builds one :class:`ReactionScoreContext` per
    step, computes its normalized weighted score, and returns the arithmetic
    mean over the route.

    STScore requires a rule resolver to map each route step to a
    CanonicalRetroReactor. When no resolver is given (default), STScore
    cannot be enabled.

    :param config: RetrekRouteScoringConfig instance.
    :param rule_resolver: Optional callable that maps a Step
        to a CanonicalRetroReactor or None.
    """

    def __init__(
        self,
        config: RetrekRouteScoringConfig | None = None,
        rule_resolver: Callable[[Step], "CanonicalRetroReactor | None"] | None = None,
    ) -> None:
        self._config = config or RetrekRouteScoringConfig()
        if "st" in self._config.enabled_scores and rule_resolver is None:
            raise ValueError("STScore requires a rule_resolver for Route steps")
        self._rule_resolver = rule_resolver
        self._scorers_and_weights = [
            (_SCORERS[name](), self._config.weights[name])
            for name in self._config.enabled_scores
        ]

    def _build_context(
        self, step: Step, available_leaf_ids: frozenset[int]
    ) -> ReactionScoreContext:
        """Build ReactionScoreContext from one detached route step.

        :param step: One forward-direction route step.
        :param available_leaf_ids: Identities of purchasable route leaves.
        :return: ReactionScoreContext for scoring.
        """
        rule = self._rule_resolver(step) if self._rule_resolver is not None else None

        return ReactionScoreContext(
            product=step.product,
            new_precursors=tuple(step.reaction.reactants),
            available_precursors=tuple(
                id(precursor) in available_leaf_ids
                for precursor in step.reaction.reactants
            ),
            rule=rule,
        )

    def step_scores(self, route: Route) -> tuple[float, ...]:
        """One normalized ReTReK score per step, in route order.

        :param route: Detached route to judge.
        :return: Step scores in [0, 1].
        :raises ValueError: If no configured score is available for a step.
        """
        unresolved = route.unresolved
        available_leaf_ids = frozenset(
            id(leaf) for leaf in route.leaves() if molecule_key(leaf) not in unresolved
        )
        scores = []
        for index, step in enumerate(route.steps):
            context = self._build_context(step, available_leaf_ids)
            step_score = aggregate_retrek_score(context, self._scorers_and_weights)
            if math.isnan(step_score):
                raise ValueError(
                    f"no configured ReTReK score is available for route step {index}"
                )
            scores.append(step_score)

        return tuple(scores)

    def score(self, route: Route) -> float:
        """Arithmetic mean of :meth:`step_scores` for a detached route.

        :param route: Detached route to judge.
        :return: Mean score in [0, 1].
        """
        step_scores = self.step_scores(route)

        return sum(step_scores) / len(step_scores)


class CDRouteScorer(RetrekRouteScorer):
    """RetrekRouteScorer using CDScore only."""

    def __init__(self, **kwargs):
        config = RetrekRouteScoringConfig(enabled_scores=("cd",))
        super().__init__(config=config, **kwargs)


class ASRouteScorer(RetrekRouteScorer):
    """RetrekRouteScorer using ASScore only."""

    def __init__(self, **kwargs):
        config = RetrekRouteScoringConfig(enabled_scores=("as",))
        super().__init__(config=config, **kwargs)


class RDRouteScorer(RetrekRouteScorer):
    """RetrekRouteScorer using RDScore only."""

    def __init__(self, **kwargs):
        config = RetrekRouteScoringConfig(enabled_scores=("rd",))
        super().__init__(config=config, **kwargs)


class STRouteScorer(RetrekRouteScorer):
    """RetrekRouteScorer using the provisional STScore only."""

    def __init__(self, **kwargs):
        config = RetrekRouteScoringConfig(enabled_scores=("st",))
        super().__init__(config=config, **kwargs)
