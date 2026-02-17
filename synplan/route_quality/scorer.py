"""General route scorer interface for post-search re-ranking.

Provides an abstract base class that Tree uses to adjust route scores
after the search loop, and concrete implementations for different
scoring strategies.
"""

from abc import ABC, abstractmethod

from chython.containers import ReactionContainer

from synplan.route_quality.protection.scorer import CompetingSitesScore


class RouteScorer(ABC):
    """Abstract base for post-search route re-ranking.

    Subclasses implement :meth:`score` to evaluate a synthesis route and
    optionally override :meth:`rescore` to customise how the quality
    score is blended with the original tree search score.
    """

    @abstractmethod
    def score(self, route: tuple[ReactionContainer, ...]) -> float:
        """Evaluate a synthesis route.

        :param route: Ordered tuple of reactions (forward direction).
        :return: Quality score, typically in [0, 1].
        """

    def rescore(
        self,
        original_score: float,
        route: tuple[ReactionContainer, ...],
    ) -> float:
        """Combine the original tree score with this scorer's assessment.

        Default: ``original * score(route)`` (multiplicative weighting
        as in Westerlund et al., 2025).  Override for custom blending.

        :param original_score: Raw score from the tree search.
        :param route: Ordered tuple of reactions.
        :return: Adjusted score.
        """
        return original_score * self.score(route)


class ProtectionRouteScorer(RouteScorer):
    """Route scorer based on competing functional-group incompatibility.

    Wraps a :class:`CompetingSitesScore` and applies the paper's
    re-ranking formula::

        rescored = original * ((1 - w) + w * S(T))

    With the default ``weight=1.0`` this reduces to ``original * S(T)``.

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
        from synplan.route_quality.protection.config import ProtectionConfig
        from synplan.route_quality.protection.functional_groups import (
            FunctionalGroupDetector,
            HalogenDetector,
        )
        from synplan.route_quality.protection.scanner import (
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

    def score(self, route: tuple[ReactionContainer, ...]) -> float:
        """Compute the competing-sites score S(T) for a route.

        :param route: Ordered tuple of reactions.
        :return: S(T) in [0, 1].
        """
        route_dict = dict(enumerate(route))
        st, _ = self._scorer.score_route(route_dict)
        return st

    def rescore(
        self,
        original_score: float,
        route: tuple[ReactionContainer, ...],
    ) -> float:
        """Apply weighted protection penalty.

        :param original_score: Raw tree search score.
        :param route: Ordered tuple of reactions.
        :return: ``original * ((1 - w) + w * S(T))``.
        """
        st = self.score(route)
        w = self._weight
        return original_score * ((1.0 - w) + w * st)
