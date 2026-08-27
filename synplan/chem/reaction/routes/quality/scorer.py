"""General route scorer interface for post-search re-ranking.

Provides an abstract base class for judging finished routes, and concrete
implementations for different scoring strategies.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from synplan.chem.reaction.routes.quality.protection.scorer import CompetingSitesScore
from synplan.chem.reaction.routes.route import Route


class RouteScorer(ABC):
    """Abstract base for post-search route re-ranking.

    Subclasses implement :meth:`score` to evaluate a synthesis route and
    optionally override :meth:`rescore` to customise how the quality
    score is blended with the search score the route carries.
    """

    @abstractmethod
    def score(self, route: Route) -> float:
        """Evaluate a synthesis route.

        :param route: The route to judge.
        :return: Quality score, typically in [0, 1].
        """

    def rescore(self, route: Route) -> float:
        """Combine the route's search score with this scorer's assessment.

        Default: ``search_score * score(route)`` (multiplicative weighting
        as in Westerlund et al., 2025).  Override for custom blending.

        :param route: The route to judge.
        :return: Adjusted score.
        """
        return _search_score(route) * self.score(route)

    def rank(self, routes: Iterable[Route]) -> list[Route]:
        """The routes, best :meth:`rescore` first.

        :param routes: Routes from one tree, from several, or read from a file.
        :return: A new list, best first.
        """
        return sorted(routes, key=self.rescore, reverse=True)


def _search_score(route: Route) -> float:
    """The search's own number, or 1.0 for a route with no search behind it."""
    provenance = route.provenance
    if provenance is None or provenance.search_score is None:
        return 1.0
    return provenance.search_score


class ProtectionRouteScorer(RouteScorer):
    """Route scorer based on competing functional-group incompatibility.

    Wraps a :class:`CompetingSitesScore` and applies the paper's
    re-ranking formula::

        rescored = search_score * ((1 - w) + w * S(T))

    With the default ``weight=1.0`` this reduces to ``search_score * S(T)``.

    :param scorer: A configured :class:`CompetingSitesScore` instance.
    :param weight: Strength of the protection penalty in [0, 1].
        1.0 matches the paper exactly; lower values soften the penalty.
    """

    def __init__(self, scorer: CompetingSitesScore, weight: float = 1.0):
        self._scorer = scorer
        self._weight = weight

    @classmethod
    def from_config(cls, config=None, weight: float = 1.0) -> "ProtectionRouteScorer":
        """Build a scorer from a :class:`ProtectionConfig`.

        :param config: A ProtectionConfig instance.  If ``None``, uses
            default paths bundled with SynPlanner.
        :param weight: Protection penalty weight.
        :return: Configured ProtectionRouteScorer.
        """
        from synplan.chem.reaction.routes.quality.protection.config import (
            ProtectionConfig,
        )
        from synplan.chem.reaction.routes.quality.protection.functional_groups import (
            FunctionalGroupDetector,
            HalogenDetector,
        )
        from synplan.chem.reaction.routes.quality.protection.scanner import (
            IncompatibilityMatrix,
            RouteScanner,
        )

        if config is None:
            config = ProtectionConfig()

        detector = FunctionalGroupDetector(config.competing_groups_path)
        matrix = IncompatibilityMatrix(config.incompatibility_path)
        halogen = HalogenDetector(config.halogen_groups_path)
        scanner = RouteScanner(detector, matrix, halogen)
        scorer = CompetingSitesScore(scanner)
        return cls(scorer, weight=weight)

    def score(self, route: Route) -> float:
        """Compute the competing-sites score S(T) for a route.

        :param route: The route to judge.
        :return: S(T) in [0, 1].
        """
        st, _ = self._scorer.score_route(
            dict(enumerate(step.reaction for step in route))
        )
        return st

    def rescore(self, route: Route) -> float:
        """Apply weighted protection penalty.

        :param route: The route to judge.
        :return: ``search_score * ((1 - w) + w * S(T))``.
        """
        w = self._weight
        return _search_score(route) * ((1.0 - w) + w * self.score(route))
