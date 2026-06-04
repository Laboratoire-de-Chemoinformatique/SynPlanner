import subprocess
import sys


def test_legacy_hash_route_exports_constants_and_functions():
    import synplan.chem.reaction_routes.hash_route as legacy_hash_route
    from synplan.chem.reaction_routes.hash_route import (
        BUCKET_HASH_SCHEMA,
        HASH_EXCLUDES,
        HASH_INCLUDES,
        HASH_SCHEMA,
        route_cgr_hash,
    )
    from synplan.routes.route_cgr.hash import (
        BUCKET_HASH_SCHEMA as NEW_BUCKET_HASH_SCHEMA,
    )
    from synplan.routes.route_cgr.hash import (
        HASH_EXCLUDES as NEW_HASH_EXCLUDES,
    )
    from synplan.routes.route_cgr.hash import (
        HASH_INCLUDES as NEW_HASH_INCLUDES,
    )
    from synplan.routes.route_cgr.hash import (
        HASH_SCHEMA as NEW_HASH_SCHEMA,
    )
    from synplan.routes.route_cgr.hash import (
        route_cgr_hash as new_route_cgr_hash,
    )

    assert HASH_SCHEMA == NEW_HASH_SCHEMA
    assert BUCKET_HASH_SCHEMA == NEW_BUCKET_HASH_SCHEMA
    assert HASH_INCLUDES == NEW_HASH_INCLUDES
    assert HASH_EXCLUDES == NEW_HASH_EXCLUDES
    assert route_cgr_hash is new_route_cgr_hash
    assert "HASH_SCHEMA" in legacy_hash_route.__all__
    assert "route_cgr_hash" in legacy_hash_route.__all__


def test_legacy_clustering_exports_old_module_level_helpers():
    import synplan.chem.reaction_routes.clustering as legacy_clustering
    from synplan.chem.reaction_routes.clustering import (
        DynamicX,
        cgr_display,
        cluster_routes,
        compose_all_route_cgrs,
    )
    from synplan.routes.clustering import cluster_routes as new_cluster_routes
    from synplan.routes.clustering.leaving_groups import DynamicX as NewDynamicX
    from synplan.routes.depiction import cgr_display as new_cgr_display
    from synplan.routes.route_cgr import (
        compose_all_route_cgrs as new_compose_all_route_cgrs,
    )

    assert DynamicX is NewDynamicX
    assert cgr_display is new_cgr_display
    assert cluster_routes is new_cluster_routes
    assert compose_all_route_cgrs is new_compose_all_route_cgrs
    for name in (
        "DynamicX",
        "cgr_display",
        "cluster_routes",
        "compose_all_route_cgrs",
    ):
        assert name in legacy_clustering.__all__


def test_legacy_route_quality_exports_old_meaningful_helpers():
    from chython.containers import ReactionContainer as NewReactionContainer

    from synplan.route_quality import ProtectionConfig, RouteScorer
    from synplan.route_quality.protection.scanner import (
        FunctionalGroupDetector,
        HalogenDetector,
    )
    from synplan.route_quality.scorer import (
        CompetingSitesScore,
        ReactionContainer,
    )
    from synplan.routes.quality.protection.config import (
        ProtectionConfig as NewProtectionConfig,
    )
    from synplan.routes.quality.protection.functional_groups import (
        FunctionalGroupDetector as NewFunctionalGroupDetector,
    )
    from synplan.routes.quality.protection.functional_groups import (
        HalogenDetector as NewHalogenDetector,
    )
    from synplan.routes.quality.protection.scorer import (
        CompetingSitesScore as NewCompetingSitesScore,
    )
    from synplan.routes.quality.scorer import RouteScorer as NewRouteScorer

    assert CompetingSitesScore is NewCompetingSitesScore
    assert ReactionContainer is NewReactionContainer
    assert FunctionalGroupDetector is NewFunctionalGroupDetector
    assert HalogenDetector is NewHalogenDetector
    assert ProtectionConfig is NewProtectionConfig
    assert RouteScorer is NewRouteScorer


def test_legacy_route_cgr_exports_old_meaningful_helpers():
    from chython.containers import (
        CGRContainer as NewCGRContainer,
    )
    from chython.containers import (
        MoleculeContainer as NewMoleculeContainer,
    )
    from chython.containers import (
        ReactionContainer as NewReactionContainer,
    )
    from chython.containers.bonds import DynamicBond as NewDynamicBond

    from synplan.chem.reaction_routes.route_cgr import (
        CGRContainer,
        DynamicBond,
        MoleculeContainer,
        ReactionContainer,
        Tree,
    )
    from synplan.mcts.tree import Tree as NewTree

    assert CGRContainer is NewCGRContainer
    assert DynamicBond is NewDynamicBond
    assert MoleculeContainer is NewMoleculeContainer
    assert ReactionContainer is NewReactionContainer
    assert Tree is NewTree


def test_legacy_reaction_routes_package_root_is_lightweight():
    code = """
import sys
import synplan.chem.reaction_routes
assert 'matplotlib' not in sys.modules
assert 'synplan.utils.visualisation' not in sys.modules
assert 'synplan.mcts.tree' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True)
