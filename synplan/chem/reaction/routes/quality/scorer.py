"""General route scorer interface for post-search re-ranking.

Provides an abstract base class for judging finished routes, and concrete
implementations for different scoring strategies.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from synplan.chem.reaction.routes.quality.protection.config import ProtectionConfig
from synplan.chem.reaction.routes.quality.protection.functional_groups import (
    FunctionalGroupDetector,
    HalogenDetector,
)
from synplan.chem.reaction.routes.quality.protection.scanner import (
    IncompatibilityMatrix,
    RouteScanner,
)
from synplan.chem.reaction.routes.quality.protection.scorer import CompetingSitesScore
from synplan.chem.reaction.routes.route import Route


class RouteScorer(ABC):
    """Abstract base for judging a finished route.

    A scorer stands behind one number. :meth:`score` is it, :meth:`rank` puts
    routes in order by it, and how the number is arrived at is the scorer's own
    business -- read the search's verdict and weight it, ignore it, or produce a
    number from nothing at all. The base class does not decide that for you: a
    yield or a cost model has no reason to be a multiple of a tree search score.
    """

    @abstractmethod
    def score(self, route: Route) -> float:
        """The number this scorer stands behind, higher being better.

        :param route: The route to judge.
        :return: This scorer's verdict.
        """

    def rank(self, routes: Iterable[Route]) -> list[Route]:
        """The routes, best :meth:`score` first.

        This scores every route, so it costs one :meth:`score` per route --
        tens of seconds for a few hundred under :class:`ProtectionRouteScorer`,
        whose scan runs ~64 ms per molecule its cache has not seen. It is not
        the cheap ``sorted`` its name suggests.

        :param routes: The routes to order.
        :return: A new list, best first.
        """
        return sorted(routes, key=self.score, reverse=True)


class ProtectionRouteScorer(RouteScorer):
    """Route scorer based on competing functional-group incompatibility.

    Wraps a :class:`CompetingSitesScore`, so :meth:`rank` orders routes by
    ``search_score * S(T)``: the paper's re-ranking, unweighted, because the
    paper has no weight. :meth:`competing_sites_score` is S(T) on its own, for
    routes no search produced.

    :param scorer: A configured :class:`CompetingSitesScore` instance.
    """

    def __init__(self, scorer: CompetingSitesScore):
        self._scorer = scorer

    @classmethod
    def from_config(
        cls, config: ProtectionConfig | None = None
    ) -> "ProtectionRouteScorer":
        """Build a scorer from a :class:`ProtectionConfig`.

        :param config: A ProtectionConfig instance.  If ``None``, uses
            default paths bundled with SynPlanner.
        :return: Configured ProtectionRouteScorer.
        """
        if config is None:
            config = ProtectionConfig()

        detector = FunctionalGroupDetector(config.competing_groups_path)
        matrix = IncompatibilityMatrix(config.incompatibility_path)
        halogen = HalogenDetector(config.halogen_groups_path)
        scanner = RouteScanner(detector, matrix, halogen)
        return cls(CompetingSitesScore(scanner))

    def competing_sites_score(self, route: Route) -> float:
        """The competing-sites score S(T), this scorer's own opinion of a route.

        Judges the route on its own terms, without the search: use it to order
        routes no search produced, such as routes read back out of a file.

        :param route: The route to judge.
        :return: S(T) in [0, 1].
        """
        st, _ = self._scorer.score_route(
            dict(enumerate(step.reaction for step in route))
        )
        return st

    def score(self, route: Route) -> float:
        """``search_score * S(T)`` -- the re-ranking of Westerlund et al., 2025.

        The paper weights the search's own state score by the competing sites
        score, so this scorer's verdict is defined only for a route a search
        produced. A route read from a file has no search score and no way to be
        put on this scale; order those by :meth:`competing_sites_score`, which
        needs no search behind it.

        :param route: The route to judge.
        :raises ValueError: If the route carries no search score.
        :return: The search score weighted by S(T).
        """
        provenance = route.provenance
        search = provenance.search_score if provenance is not None else None
        if search is None:
            raise ValueError(
                "this score is the search score weighted by S(T), and the route "
                "carries no search score; order it by competing_sites_score()."
            )
        return search * self.competing_sites_score(route)
