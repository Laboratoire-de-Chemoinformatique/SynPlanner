from chython.containers import ReactionContainer

from synplan.chem.reaction.routes.clustering import cluster_routes
from synplan.chem.reaction.routes.clustering.pseudo_atoms import DynamicX
from synplan.chem.reaction.routes.quality import ProtectionConfig, RouteScorer
from synplan.chem.reaction.routes.quality.protection.functional_groups import (
    FunctionalGroupDetector,
    HalogenDetector,
)
from synplan.chem.reaction.routes.quality.protection.scorer import CompetingSitesScore
from synplan.chem.reaction.routes.representation import (
    compose_all_route_cgrs,
    compose_route_cgr,
    compose_sb_cgr,
    extract_reactions,
)
from synplan.chem.reaction.routes.representation.hash import (
    BUCKET_HASH_SCHEMA,
    HASH_EXCLUDES,
    HASH_INCLUDES,
    HASH_SCHEMA,
    route_cgr_hash,
)
from synplan.chem.reaction.routes.visualisation import cgr_display


def test_route_hash_exports_constants_and_functions():
    assert HASH_SCHEMA
    assert BUCKET_HASH_SCHEMA
    assert HASH_INCLUDES
    assert HASH_EXCLUDES
    assert callable(route_cgr_hash)


def test_clustering_exports_route_helpers():
    assert DynamicX.__name__ == "DynamicX"
    assert cgr_display.__name__ == "cgr_display"
    assert cluster_routes.__name__ == "cluster_routes"
    assert compose_all_route_cgrs.__name__ == "compose_all_route_cgrs"


def test_route_quality_exports_meaningful_helpers():
    assert CompetingSitesScore.__name__ == "CompetingSitesScore"
    assert ReactionContainer.__name__ == "ReactionContainer"
    assert FunctionalGroupDetector.__name__ == "FunctionalGroupDetector"
    assert HalogenDetector.__name__ == "HalogenDetector"
    assert ProtectionConfig.__name__ == "ProtectionConfig"
    assert RouteScorer.__name__ == "RouteScorer"


def test_route_cgr_exports_meaningful_helpers():
    assert compose_route_cgr.__name__ == "compose_route_cgr"
    assert compose_sb_cgr.__name__ == "compose_sb_cgr"
    assert compose_all_route_cgrs.__name__ == "compose_all_route_cgrs"
    assert extract_reactions.__name__ == "extract_reactions"
