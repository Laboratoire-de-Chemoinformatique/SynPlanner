import subprocess
import sys


def test_legacy_hash_route_exports_constants_and_functions():
    import synplan.chem.reaction_routes.hash_route as legacy_hash_route
    from synplan.chem.reaction.routes.representation.hash import (
        BUCKET_HASH_SCHEMA as NEW_BUCKET_HASH_SCHEMA,
    )
    from synplan.chem.reaction.routes.representation.hash import (
        HASH_EXCLUDES as NEW_HASH_EXCLUDES,
    )
    from synplan.chem.reaction.routes.representation.hash import (
        HASH_INCLUDES as NEW_HASH_INCLUDES,
    )
    from synplan.chem.reaction.routes.representation.hash import (
        HASH_SCHEMA as NEW_HASH_SCHEMA,
    )
    from synplan.chem.reaction.routes.representation.hash import (
        route_cgr_hash as new_route_cgr_hash,
    )

    assert legacy_hash_route.HASH_SCHEMA == NEW_HASH_SCHEMA
    assert legacy_hash_route.BUCKET_HASH_SCHEMA == NEW_BUCKET_HASH_SCHEMA
    assert legacy_hash_route.HASH_INCLUDES == NEW_HASH_INCLUDES
    assert legacy_hash_route.HASH_EXCLUDES == NEW_HASH_EXCLUDES
    assert legacy_hash_route.route_cgr_hash is new_route_cgr_hash
    assert "HASH_SCHEMA" in legacy_hash_route.__all__
    assert "route_cgr_hash" in legacy_hash_route.__all__


def test_legacy_clustering_exports_old_module_level_helpers():
    import synplan.chem.reaction_routes.clustering as legacy_clustering
    from synplan.chem.reaction.routes.clustering import (
        cluster_routes as new_cluster_routes,
    )
    from synplan.chem.reaction.routes.leaving_groups import DynamicX as NewDynamicX
    from synplan.chem.reaction.routes.representation import (
        compose_all_route_cgrs as new_compose_all_route_cgrs,
    )
    from synplan.chem.reaction.routes.visualisation import (
        cgr_display as new_cgr_display,
    )

    assert legacy_clustering.DynamicX is NewDynamicX
    assert legacy_clustering.cgr_display is new_cgr_display
    assert legacy_clustering.cluster_routes is new_cluster_routes
    assert legacy_clustering.compose_all_route_cgrs is new_compose_all_route_cgrs
    for name in (
        "DynamicX",
        "cgr_display",
        "cluster_routes",
        "compose_all_route_cgrs",
    ):
        assert name in legacy_clustering.__all__


def test_legacy_route_quality_exports_old_meaningful_helpers():
    from chython.containers import ReactionContainer as NewReactionContainer

    from synplan.chem.reaction.routes.quality.protection.config import (
        ProtectionConfig as NewProtectionConfig,
    )
    from synplan.chem.reaction.routes.quality.protection.functional_groups import (
        FunctionalGroupDetector as NewFunctionalGroupDetector,
    )
    from synplan.chem.reaction.routes.quality.protection.functional_groups import (
        HalogenDetector as NewHalogenDetector,
    )
    from synplan.chem.reaction.routes.quality.protection.scorer import (
        CompetingSitesScore as NewCompetingSitesScore,
    )
    from synplan.chem.reaction.routes.quality.scorer import (
        RouteScorer as NewRouteScorer,
    )
    from synplan.route_quality import ProtectionConfig, RouteScorer
    from synplan.route_quality.protection.scanner import (
        FunctionalGroupDetector,
        HalogenDetector,
    )
    from synplan.route_quality.scorer import (
        CompetingSitesScore,
        ReactionContainer,
    )

    assert CompetingSitesScore is NewCompetingSitesScore
    assert ReactionContainer is NewReactionContainer
    assert FunctionalGroupDetector is NewFunctionalGroupDetector
    assert HalogenDetector is NewHalogenDetector
    assert ProtectionConfig is NewProtectionConfig
    assert RouteScorer is NewRouteScorer


def test_legacy_route_cgr_exports_old_meaningful_helpers():
    from synplan.chem.reaction.routes.representation import (
        compose_all_route_cgrs as new_compose_all_route_cgrs,
    )
    from synplan.chem.reaction.routes.representation import (
        compose_route_cgr as new_compose_route_cgr,
    )
    from synplan.chem.reaction.routes.representation import (
        compose_sb_cgr as new_compose_sb_cgr,
    )
    from synplan.chem.reaction.routes.representation import (
        extract_reactions as new_extract_reactions,
    )
    from synplan.chem.reaction_routes.route_cgr import (
        compose_all_route_cgrs,
        compose_route_cgr,
        compose_sb_cgr,
        extract_reactions,
    )

    assert compose_route_cgr is new_compose_route_cgr
    assert compose_sb_cgr is new_compose_sb_cgr
    assert compose_all_route_cgrs is new_compose_all_route_cgrs
    assert extract_reactions is new_extract_reactions


def test_legacy_reaction_routes_package_root_is_lightweight():
    code = """
import sys
import synplan.chem.reaction_routes
assert 'matplotlib' not in sys.modules
assert 'synplan.utils.visualisation' not in sys.modules
assert 'synplan.mcts.tree' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True)
