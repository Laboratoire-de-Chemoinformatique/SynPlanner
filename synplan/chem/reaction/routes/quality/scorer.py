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


def _competing_sites(config: ProtectionConfig | None) -> CompetingSitesScore:
    """The competing-sites machinery both protection scorers are built on."""
    if config is None:
        config = ProtectionConfig()
    return CompetingSitesScore(
        RouteScanner(
            FunctionalGroupDetector(config.competing_groups_path),
            IncompatibilityMatrix(config.incompatibility_path),
            HalogenDetector(config.halogen_groups_path),
        )
    )


class CompetingSitesRouteScorer(RouteScorer):
    """Judges a route on competing functional groups alone, S(T) in [0, 1].

    1.0 means no competing interactions were detected; lower values mean steps
    that may need a protecting-group strategy. It never looks at the search, so
    it ranks routes no search produced -- routes read back out of a file, or
    from two trees that cannot be compared on their own scores.

    :param scorer: A configured :class:`CompetingSitesScore` instance.
    """

    def __init__(self, scorer: CompetingSitesScore):
        self._scorer = scorer

    @classmethod
    def from_config(
        cls, config: ProtectionConfig | None = None
    ) -> "CompetingSitesRouteScorer":
        """Build a scorer from a :class:`ProtectionConfig`.

        :param config: A ProtectionConfig instance. If ``None``, uses the
            default paths bundled with SynPlanner.
        :return: Configured CompetingSitesRouteScorer.
        """
        return cls(_competing_sites(config))

    def score(self, route: Route) -> float:
        """S(T) for a route, judged on its own terms.

        :param route: The route to judge.
        :return: S(T) in [0, 1].
        """
        st, _ = self._scorer.score_route(
            dict(enumerate(step.reaction for step in route))
        )
        return st


class ProtectionRouteScorer(RouteScorer):
    """Route scorer based on competing functional-group incompatibility.

    The search's own verdict weighted by :class:`CompetingSitesRouteScorer`'s,
    which is the re-ranking of Westerlund et al., 2026 -- unweighted, because
    the paper has no weight. Rank with :class:`CompetingSitesRouteScorer`
    instead when the routes have no search behind them.

    :param quality: The competing-sites scorer whose verdict is applied.
    """

    def __init__(self, quality: CompetingSitesRouteScorer):
        self._quality = quality

    @classmethod
    def from_config(
        cls, config: ProtectionConfig | None = None
    ) -> "ProtectionRouteScorer":
        """Build a scorer from a :class:`ProtectionConfig`.

        :param config: A ProtectionConfig instance.  If ``None``, uses
            default paths bundled with SynPlanner.
        :return: Configured ProtectionRouteScorer.
        """
        return cls(CompetingSitesRouteScorer(_competing_sites(config)))

    def score(self, route: Route) -> float:
        """``search_score * S(T)`` -- the re-ranking of Westerlund et al., 2026.

        The paper weights the search's own state score by the competing sites
        score, so this scorer's verdict is defined only for a route a search
        produced. A route read from a file has no search score and no way to be
        put on this scale; rank those with :class:`CompetingSitesRouteScorer`,
        which needs no search behind it.

        :param route: The route to judge.
        :raises ValueError: If the route carries no search score.
        :return: The search score weighted by S(T).
        """
        provenance = route.provenance
        search = provenance.search_score if provenance is not None else None
        if search is None:
            raise ValueError(
                "this score is the search score weighted by S(T), and the route "
                "carries no search score; rank it with CompetingSitesRouteScorer."
            )
        return search * self._quality.score(route)
