from chython import smiles

from synplan.chem.reaction.routes.analysis import (
    collect_bb_usage_stats,
    compare_sb_cgr_clusters,
    flatten_route_id_groups,
    route_cgr_overlap_rows,
    route_cgr_subset,
    route_ids_with_exact_bb,
    sb_cgr_identity_to_cluster_id,
)
from synplan.chem.reaction.routes.notebook_plots import top_bb_usage_rows
from synplan.chem.reaction.routes.representation import compose_route_cgr


def _route_cgr():
    routes = {
        1: {
            0: smiles("[CH3:1].[CH3:2][Cl:3]>>[CH3:1][CH3:2].[ClH:3]"),
        }
    }
    return compose_route_cgr(routes, 1, preserve_transient_bonds=True)["cgr"]


def test_route_ids_with_exact_bb_matches_exact_pseudo_reactants():
    route_cgr = _route_cgr()

    assert route_ids_with_exact_bb("ClC", {7: route_cgr}, kind="real") == [7]
    assert route_ids_with_exact_bb("CC", {7: route_cgr}, kind="real") == []
    assert route_ids_with_exact_bb("ClC", {7: route_cgr}, kind="supporting") == []


def test_collect_bb_usage_stats_classifies_target_atom_overlap():
    route_cgr = _route_cgr()

    stats = collect_bb_usage_stats({7: route_cgr})

    assert "ClC" in stats["real_bb"]
    assert stats["real_bb"]["ClC"]["route_ids"] == [7]
    assert stats["real_bb"]["ClC"]["route_count"] == 1
    assert stats["supporting"] == {}
    assert "ClC" in stats["by_route"][7]["real_bb"]


def test_route_id_helpers():
    route_cgr = _route_cgr()

    assert flatten_route_id_groups({"a": [3, 1], "b": [2]}) == [1, 2, 3]
    assert route_cgr_subset({7: route_cgr, 8: None}, [7]) == {7: route_cgr}


def test_route_cgr_overlap_rows():
    result = {
        "route_ids_overlap": {
            "hash-a": {
                "route_cgr_dict_1": [33, 36],
                "route_cgr_dict_2": [101],
            },
            "hash-b": {
                "route_cgr_dict_1": [44],
                "route_cgr_dict_2": [201, 202],
            },
        }
    }

    assert route_cgr_overlap_rows(result) == [
        {
            "exact_hash": "hash-a",
            "route_id_1": 33,
            "route_id_2": 101,
            "route_ids_1": [33, 36],
            "route_ids_2": [101],
        },
        {
            "exact_hash": "hash-b",
            "route_id_1": 44,
            "route_id_2": 201,
            "route_ids_1": [44],
            "route_ids_2": [201, 202],
        },
    ]


def test_compare_sb_cgr_clusters():
    clusters_1 = {
        "1.1": {"sb_cgr": "A"},
        "1.2": {"sb_cgr": "B"},
    }
    clusters_2 = {
        "2.1": {"sb_cgr": "B"},
        "2.2": {"sb_cgr": "C"},
    }

    result = compare_sb_cgr_clusters(clusters_1, clusters_2)

    assert result["unique_cluster_ids_1"] == ["1.1"]
    assert result["overlap_cluster_ids"] == [("1.2", "2.1")]
    assert result["unique_cluster_ids_2"] == ["2.2"]


def test_top_bb_usage_rows_filters_and_sorts():
    stats = {
        "CC": {"route_count": 99, "occurrences": 99},
        "CCCC": {"route_count": 2, "occurrences": 4},
        "CCCO": {"route_count": 3, "occurrences": 3},
    }

    assert top_bb_usage_rows(stats, top_n=2, min_mol_size=4) == [
        ("CCCO", 3, 3),
        ("CCCC", 2, 4),
    ]


def test_analysis_accepts_legacy_route_cgr_wrappers_and_molecule_inputs():
    route_cgr = _route_cgr()
    wrapped = {"cgr": route_cgr}

    assert route_ids_with_exact_bb(smiles("ClC"), {7: wrapped}) == [7]
    stats = collect_bb_usage_stats({7: wrapped})
    assert stats["real_bb"]["ClC"]["route_ids"] == [7]


def test_cluster_identity_accepts_missing_keys_and_attribute_records():
    class Cluster:
        sb_cgr = "A"

    assert sb_cgr_identity_to_cluster_id(
        {"missing": {}, "object": Cluster()}
    ) == {"A": "object"}
