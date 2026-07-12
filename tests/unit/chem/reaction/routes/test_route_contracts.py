from __future__ import annotations

import csv
import json
import warnings

import pytest
from chython import smiles

from synplan.chem.reaction.routes.clustering.subclustering import (
    post_process_subcluster,
)
from synplan.chem.reaction.routes.contracts import (
    RouteExportError,
    SubclusterRouteData,
)
from synplan.chem.reaction.routes.io import (
    build_route_trees,
    make_dict,
    make_json,
    read_routes_csv,
    write_routes_csv,
    write_routes_json,
)
from synplan.chem.reaction.routes.representation import (
    build_route_cgr,
    compose_route_cgr,
)
from synplan.chem.reaction.routes.representation.state import bond_key


def _routes():
    return {1: {0: smiles("[CH3:1].[CH3:2][Cl:3]>>[CH3:1][CH3:2].[ClH:3]")}}


def test_build_route_cgr_exposes_typed_result_and_legacy_adapter():
    result = build_route_cgr(_routes(), 1, include_reactions=True)

    assert result.ok
    assert result.cgr is not None
    assert result.reactions_dict is not None
    assert result.as_legacy_dict(include_reactions=True)["cgr"] == result.cgr
    assert compose_route_cgr(_routes(), 1)["cgr"] == result.cgr


def test_build_route_cgr_reports_tree_composition_failure():
    class BrokenTree:
        def synthesis_route(self, route_id):
            raise ValueError(f"missing route {route_id}")

    result = build_route_cgr(BrokenTree(), 3)

    assert not result.ok
    assert result.diagnostic is not None
    assert result.diagnostic.route_id == 3
    assert result.diagnostic.stage == "route_cgr_composition"


def test_route_tree_diagnostics_and_strict_mode(monkeypatch, tmp_path):
    import synplan.chem.reaction.routes.io.json as route_io

    monkeypatch.setattr(route_io, "_make_json_v1", lambda *args, **kwargs: {})
    routes = _routes()

    result = build_route_trees(routes)
    assert result.routes == {}
    assert result.diagnostics[0].route_id == 1

    output = tmp_path / "routes.json"
    with pytest.raises(RouteExportError):
        write_routes_json(routes, output, strict=True)
    assert not output.exists()


def test_route_json_and_csv_preserve_reaction_metadata(tmp_path):
    reaction = _routes()[1][0]
    reaction.meta["source"] = "unit-test"
    routes = {1: {0: reaction}}

    route_tree = make_json(routes)
    reaction_node = route_tree[1]["children"][0]
    assert reaction_node["meta"] == {"source": "unit-test"}
    restored = make_dict(route_tree)
    assert restored[1][0].meta["source"] == "unit-test"

    csv_path = tmp_path / "routes.csv"
    write_routes_csv(routes, csv_path)
    with csv_path.open(newline="") as file:
        row = next(csv.DictReader(file))
    assert json.loads(row["meta"]) == {"source": "unit-test"}
    assert read_routes_csv(csv_path)[1][0].meta["source"] == "unit-test"


def test_route_csv_preserves_legacy_non_json_metadata(tmp_path):
    csv_path = tmp_path / "legacy.csv"
    csv_path.write_text(
        'route_id,step_id,smiles,meta\n1,0,"[CH3:1]>>[CH4:1]",legacy-note\n'
    )

    reaction = read_routes_csv(csv_path)[1][0]
    assert reaction.meta["legacy_csv_meta"] == "legacy-note"


def test_bond_key_is_order_independent_and_shared():
    assert bond_key(7, 3) == (3, 7)
    assert bond_key(3, 7) == (3, 7)


def test_subcluster_route_data_keeps_legacy_tuple_shape():
    data = SubclusterRouteData(
        sb_cgr="sb",
        unlabeled_reaction="original",
        synthon_cgr="synthon",
        synthon_reaction="reaction",
        leaving_groups={1: ("group", 2)},
        leaving_group_count=1,
        supporting_groups={1: ("support", None)},
    )

    assert data.as_legacy_tuple() == (
        "sb",
        "original",
        "synthon",
        "reaction",
        {1: ("group", 2)},
        1,
        {1: ("support", None)},
    )
    assert callable(post_process_subcluster)


def test_root_legacy_helper_warns_and_stays_available():
    import synplan.chem.reaction.routes as routes

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        helper = routes.get_clean_mapping

    assert callable(helper)
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)


def test_post_process_subcluster_does_not_mutate_completed_mapping():
    subgroup = {"post_processed": True, "routes_data": {}}

    result = post_process_subcluster(subgroup)

    assert result == subgroup
    assert result is not subgroup
