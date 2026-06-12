from pathlib import Path

import pytest

from synplan.chem.reaction.routes.clustering import (
    cluster_routes,
    subcluster_all_clusters,
)
from synplan.chem.reaction.routes.io import read_routes_json
from synplan.chem.reaction.routes.representation import (
    compose_all_route_cgrs,
    compose_all_sb_cgrs,
)

TEST_DATA = Path(__file__).resolve().parent.parent / "data" / "clustering"
ROUTE_FIXTURES = {
    "simple": TEST_DATA / "routes_mol_simple.json",
    "medium": TEST_DATA / "routes_mol_medium.json",
    "complex": TEST_DATA / "routes_mol_complex.json",
}


@pytest.fixture(scope="module", params=ROUTE_FIXTURES, ids=ROUTE_FIXTURES)
def route_fixture_name(request):
    """Route fixture target name."""
    return request.param


@pytest.fixture(scope="module")
def route_fixture_path(route_fixture_name):
    """Path to a deterministic generated route fixture."""
    path = ROUTE_FIXTURES[route_fixture_name]
    assert path.exists(), f"Test data missing: {path}"
    return path


@pytest.fixture(scope="module")
def fixture_routes_dict(route_fixture_path):
    """Load deterministic route fixture data as {route_id: {step_id: Reaction}}."""
    routes = read_routes_json(route_fixture_path, to_dict=True)
    assert routes, "Route fixture should contain at least one route"
    return routes


@pytest.fixture(scope="module")
def fixture_route_cgrs(fixture_routes_dict):
    """Compose RouteCGRs from fixed route fixtures."""
    cgrs = compose_all_route_cgrs(fixture_routes_dict)
    assert cgrs, "Route fixture should compose at least one RouteCGR"
    return cgrs


@pytest.fixture(scope="module")
def fixture_sb_cgrs(fixture_route_cgrs):
    """Compose strategic-bond CGRs from fixed RouteCGR fixtures."""
    sb_cgrs = compose_all_sb_cgrs(fixture_route_cgrs)
    assert sb_cgrs, "Route fixture should compose at least one SB-CGR"
    return sb_cgrs


@pytest.fixture(scope="module")
def fixture_clusters(fixture_sb_cgrs):
    """Cluster fixed SB-CGR fixtures."""
    clusters = cluster_routes(fixture_sb_cgrs, use_strat=False)
    assert clusters, "Route fixture should produce at least one cluster"
    return clusters


@pytest.fixture(scope="module")
def fixture_subclusters(fixture_clusters, fixture_sb_cgrs, fixture_route_cgrs):
    """Subcluster fixed route fixtures."""
    subclusters = subcluster_all_clusters(
        fixture_clusters, fixture_sb_cgrs, fixture_route_cgrs
    )
    assert subclusters, "Route fixture should produce at least one subcluster"
    return subclusters


def calc_num_routes_subclusters(subclusters):
    """Calculate the total number of routes in subclusters."""
    count = 0
    for cluster in subclusters.values():
        for subcluster in cluster.values():
            count += len(subcluster["routes_data"])
    return count


def build_route_fixture_clustering(route_fixture_name):
    """Build clustering data from one generated route fixture."""
    fixture_path = ROUTE_FIXTURES[route_fixture_name]
    assert fixture_path.exists(), f"Test data missing: {fixture_path}"

    routes = read_routes_json(fixture_path, to_dict=True)
    assert routes, "Route fixture should contain at least one route"

    route_cgrs = compose_all_route_cgrs(routes)
    assert route_cgrs, "Route fixture should compose at least one RouteCGR"

    sb_cgrs = compose_all_sb_cgrs(route_cgrs)
    assert sb_cgrs, "Route fixture should compose at least one SB-CGR"

    clusters = cluster_routes(sb_cgrs, use_strat=False)
    assert clusters, "Route fixture should produce at least one cluster"

    subclusters = subcluster_all_clusters(clusters, sb_cgrs, route_cgrs)
    assert subclusters, "Route fixture should produce at least one subcluster"

    return route_cgrs, sb_cgrs, clusters, subclusters


def assert_route_fixture_clustering(
    route_fixture_name, route_cgrs, sb_cgrs, clusters, subclusters
):
    """Assert the clustering invariants for one generated route fixture."""
    assert set(sb_cgrs).issubset(set(route_cgrs))

    total_routes = sum(cluster["group_size"] for cluster in clusters.values())
    assert total_routes == len(sb_cgrs), (
        f"Every SB-CGR route should cluster for {route_fixture_name}"
    )

    total_subclusters = calc_num_routes_subclusters(subclusters)
    assert total_subclusters == total_routes, (
        f"Total subclusters should match total routes for {route_fixture_name}"
    )
    assert sorted(subclusters.keys()) == sorted(clusters.keys()), (
        f"Subcluster keys should match cluster keys for {route_fixture_name}"
    )


@pytest.mark.integration
def test_route_fixture_clustering_pipeline(
    route_fixture_name,
    fixture_route_cgrs,
    fixture_sb_cgrs,
    fixture_clusters,
    fixture_subclusters,
):
    """Cluster and subcluster fixed routes without running MCTS planning."""
    assert_route_fixture_clustering(
        route_fixture_name,
        fixture_route_cgrs,
        fixture_sb_cgrs,
        fixture_clusters,
        fixture_subclusters,
    )


@pytest.mark.integration
def test_complex_molecule_clustering():
    """Regression: complex target clustering uses a fixed route fixture."""
    assert_route_fixture_clustering(
        "complex", *build_route_fixture_clustering("complex")
    )


@pytest.mark.integration
def test_route_fixture_clustering_with_strategic_bonds(
    route_fixture_name, fixture_sb_cgrs
):
    """The fixed route fixtures also cluster when strategic bonds are explicit."""
    clusters = cluster_routes(fixture_sb_cgrs, use_strat=True)

    assert clusters, f"Should have a strategic-bond cluster for {route_fixture_name}"
    total_routes = sum(cluster["group_size"] for cluster in clusters.values())
    assert total_routes == len(fixture_sb_cgrs)
