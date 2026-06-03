import pickle

import pytest
from chython.containers import CGRContainer, ReactionContainer

from synplan.chem.reaction_routes.clustering import (
    cluster_routes,
    subcluster_all_clusters,
    subcluster_one_cluster,
)


@pytest.fixture(scope="module")
def sb_cgrs_dict():
    """Load precomputed SB-CGRs from pickle."""
    with open("tests/data/sb_cgrs_1_1.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def routes_cgrs_dict():
    """Load precomputed RouteCGRs from pickle."""
    with open("tests/data/route_cgrs_1_1.pkl", "rb") as f:
        return pickle.load(f)


def test_cluster_routes_empty():
    """cluster_routes returns empty dict when given no data."""
    assert cluster_routes({}, use_strat=False) == {}


@pytest.mark.parametrize("use_strat", [False, True])
def test_cluster_routes_valid(sb_cgrs_dict, use_strat):
    """cluster_routes groups all routes and includes every entry."""
    clusters = cluster_routes(sb_cgrs_dict, use_strat=use_strat)

    assert isinstance(clusters, dict)
    # Every original route ID must appear in exactly one cluster

    total = sum(len(cluster["route_ids"]) for cluster in clusters.values())
    assert total == len(sb_cgrs_dict)

    expected_keys = ["2.1", "3.1", "3.2", "4.1", "4.2", "4.3"]
    assert list(clusters.keys()) == expected_keys
    for route_id, value in clusters.items():
        assert isinstance(route_id, str)
        assert isinstance(value, dict)
        assert "route_ids" in value
        assert isinstance(value["route_ids"], list)
        assert len(value["route_ids"]) > 0
        assert "sb_cgr" in value
        assert "strat_bonds" in value
        assert "group_size" in value


def test_subcluster_one_cluster_valid(sb_cgrs_dict, routes_cgrs_dict):
    """subcluster_one_cluster returns expected structure for a valid group."""
    clusters = cluster_routes(sb_cgrs_dict, use_strat=False)
    group = next(iter(clusters.values()))
    result = subcluster_one_cluster(group, sb_cgrs_dict, routes_cgrs_dict)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(group["route_ids"])
    for route_id, value in result.items():
        assert isinstance(route_id, int)
        assert isinstance(value, tuple)
        assert len(value) == 7
        (
            sb_cgr,
            unlabeled_rxn,
            synthon_cgr,
            new_rxn,
            lg_groups,
            lg_sizes,
            supporting_groups,
        ) = value

        # The SB-CGR should be exactly what we passed in
        assert sb_cgr is sb_cgrs_dict[route_id]

        # The “unlabeled” reaction comes from the original route-CGR
        assert isinstance(unlabeled_rxn, ReactionContainer)

        # The synthon CGR is a proper CGRContainer
        assert isinstance(synthon_cgr, CGRContainer)

        # The new “re-labeled” reaction is also a ReactionContainer
        assert isinstance(new_rxn, ReactionContainer)
        assert new_rxn is not unlabeled_rxn  # it should be a distinct object

        # Leaving-group info is a dict of (CGRContainer, int) tuples
        assert isinstance(lg_groups, dict)
        assert lg_sizes == len(lg_groups)
        assert isinstance(supporting_groups, dict)
        for key, (lg_cgr, idx) in lg_groups.items():
            assert isinstance(key, int)
            assert isinstance(lg_cgr, CGRContainer)
            assert isinstance(idx, int)


def test_subcluster_one_cluster_empty():
    """subcluster_one_cluster returns empty dict for empty group."""
    result = subcluster_one_cluster({"route_ids": []}, {}, {})
    assert result == {}


def test_subcluster_one_cluster_invalid_route():
    """subcluster_one_cluster returns None if route_id is missing in dicts."""
    group = {"route_ids": ["nonexistent"]}
    with pytest.raises(KeyError):
        result = subcluster_one_cluster(group, {}, {})
        assert result is None


def test_subcluster_all_clusters(sb_cgrs_dict, routes_cgrs_dict):
    """Test that subcluster_all_clusters returns the expected results."""

    # Call the function with the mock data
    clusters = cluster_routes(sb_cgrs_dict, use_strat=False)
    subclusters = subcluster_all_clusters(clusters, sb_cgrs_dict, routes_cgrs_dict)
    # Check that the result is as expected
    assert isinstance(subclusters, dict)
    total_clusters = sum(len(cluster["route_ids"]) for cluster in clusters.values())
    total_subclusters = 0
    for cluster in subclusters.values():
        for subcluster in cluster.values():
            print(subcluster)
            total_subclusters += len(subcluster["routes_data"])

    assert len(sb_cgrs_dict) == total_clusters == total_subclusters
