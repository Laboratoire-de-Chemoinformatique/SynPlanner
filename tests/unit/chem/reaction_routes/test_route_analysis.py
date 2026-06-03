from synplan.chem.reaction_routes.analysis import (
    compare_sb_cgr_clusters,
    flatten_route_id_groups,
    route_cgr_overlap_rows,
    route_cgr_subset,
)


def test_route_id_helpers():
    route_cgr = object()

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
