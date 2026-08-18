from __future__ import annotations

from types import SimpleNamespace

import pytest

from synplan.chem.building_blocks import BuildingBlockCatalog
from synplan.chem.building_blocks.reports import IdentityReportRow
from synplan.chem.reaction.routes.postprocess import (
    RoutePostprocessConfig,
    postprocess_routes,
)


def _deprotected_leaf():
    return {
        "type": "mol",
        "smiles": "N",
        "in_stock": True,
        "bb": {
            "records": [
                {
                    "source_index": 1,
                    "input_smiles": "CCOC(=O)N",
                    "canonical_smiles": "N",
                    "output_origin": "deprotected",
                }
            ]
        },
    }


def test_pipeline_refuses_cost_before_required_restoration():
    catalog = BuildingBlockCatalog(())
    result = postprocess_routes(
        {"route": _deprotected_leaf()},
        catalog,
        config=RoutePostprocessConfig(
            expand_deprotected=False,
            calculate_cost=True,
        ),
    )

    assert result.variants == ()
    assert result.diagnostics[0].stage == "ordering"


def test_pipeline_enforces_variant_limit_before_materialization():
    catalog = BuildingBlockCatalog(
        (
            IdentityReportRow(
                source_index=1,
                output_origin="deprotected",
                canonical_smiles="N",
                input_smiles="CCOC(=O)N",
                standard_inchi="",
                standard_inchikey="",
                inchi_return_code="",
                inchi_warnings="",
                status="written",
            ),
            IdentityReportRow(
                source_index=2,
                output_origin="deprotected",
                canonical_smiles="N",
                input_smiles="CC(=O)N",
                standard_inchi="",
                standard_inchikey="",
                inchi_return_code="",
                inchi_warnings="",
                status="duplicate_skipped",
            ),
        )
    )
    route = _deprotected_leaf()
    route.pop("bb")
    result = postprocess_routes(
        {3: route},
        catalog,
        config=RoutePostprocessConfig(
            calculate_cost=False,
            max_variants_per_route=1,
        ),
    )

    assert result.variants == ()
    diagnostic = result.diagnostics[0]
    assert diagnostic.route_id == 3
    assert diagnostic.stage == "bb_restoration"
    assert diagnostic.exception_type == "RouteExpansionLimitError"
    assert "requires 2 variants" in diagnostic.message


def test_pipeline_restores_before_batch_costing(monkeypatch):
    import synplan.chem.reaction.routes.postprocess.pipeline as pipeline

    observed = []

    def expand(route, catalog, *, max_variants):
        observed.append(("expand", max_variants))
        return [{**route, "restored": True}, {**route, "restored": True}]

    def cost(routes, catalog):
        assert all(route["restored"] for route in routes.values())
        observed.append(("cost", tuple(routes)))
        return {key: f"cost-{key[1]}" for key in routes}

    monkeypatch.setattr(pipeline, "expand_deprotected_building_blocks", expand)
    monkeypatch.setattr(pipeline, "estimate_route_costs", cost)
    result = postprocess_routes(
        {"r": {"type": "mol", "smiles": "C"}}, BuildingBlockCatalog(())
    )

    assert observed[0] == ("expand", 100)
    assert observed[1][0] == "cost"
    assert [(item.route_id, item.variant_index) for item in result.variants] == [
        ("r", 0),
        ("r", 1),
    ]
    assert [item.cost for item in result.variants] == ["cost-0", "cost-1"]
    assert result.ok


def test_route_failure_does_not_abort_following_routes(monkeypatch):
    import synplan.chem.reaction.routes.postprocess.pipeline as pipeline

    def expand(route, catalog, *, max_variants):
        if route["smiles"] == "bad":
            raise ValueError("broken route")
        return [dict(route)]

    monkeypatch.setattr(pipeline, "expand_deprotected_building_blocks", expand)
    result = postprocess_routes(
        {
            "bad": {"type": "mol", "smiles": "bad"},
            "good": {"type": "mol", "smiles": "C"},
        },
        SimpleNamespace(),
        config=RoutePostprocessConfig(calculate_cost=False),
    )

    assert [(item.route_id, item.variant_index) for item in result.variants] == [
        ("good", 0)
    ]
    assert result.diagnostics[0].route_id == "bad"


def test_target_and_bb_variants_form_one_ordered_cartesian_product(monkeypatch):
    import synplan.chem.reaction.routes.postprocess.pipeline as pipeline

    observed = []

    def restore(route, target, *, sequence_mode, max_variants):
        observed.append(("target", sequence_mode, max_variants, target))
        return [
            {**route, "target_variant": 0},
            {**route, "target_variant": 1},
        ]

    def expand(route, catalog, *, max_variants):
        observed.append(("bb", route["target_variant"], max_variants))
        return [{**route, "bb_variant": bb_variant} for bb_variant in range(3)]

    def stereo(route, target, *, catalog):
        observed.append(
            ("stereo", route["target_variant"], route["bb_variant"], target)
        )
        return {**route, "stereo_restored": True}

    def cost(routes, catalog):
        assert all(route["stereo_restored"] for route in routes.values())
        observed.append(("cost", len(routes)))
        return {key: f"cost-{key[1]}" for key in routes}

    monkeypatch.setattr(pipeline, "restore_protected_target", restore)
    monkeypatch.setattr(pipeline, "expand_deprotected_building_blocks", expand)
    monkeypatch.setattr(pipeline, "restore_route_stereo", stereo)
    monkeypatch.setattr(pipeline, "estimate_route_costs", cost)

    result = postprocess_routes(
        {"route": {"type": "mol", "smiles": "C"}},
        BuildingBlockCatalog(()),
        protected_targets={"route": "protected-target"},
        config=RoutePostprocessConfig(max_variants_per_route=6),
    )

    assert [
        (item.route["target_variant"], item.route["bb_variant"])
        for item in result.variants
    ] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert [item.variant_index for item in result.variants] == list(range(6))
    assert [item.cost for item in result.variants] == [
        f"cost-{index}" for index in range(6)
    ]
    assert observed[0] == ("target", "enumerate", 6, "protected-target")
    assert observed[-1] == ("cost", 6)
    assert result.ok


def test_pipeline_rejects_target_bb_cross_product_above_shared_cap(monkeypatch):
    import synplan.chem.reaction.routes.postprocess.pipeline as pipeline

    def restore(route, target, *, sequence_mode, max_variants):
        return [
            {**route, "target_variant": 0},
            {**route, "target_variant": 1},
        ]

    def expand(route, catalog, *, max_variants):
        return [{**route, "bb_variant": bb_variant} for bb_variant in range(3)]

    monkeypatch.setattr(pipeline, "restore_protected_target", restore)
    monkeypatch.setattr(pipeline, "expand_deprotected_building_blocks", expand)

    result = postprocess_routes(
        {"route": {"type": "mol", "smiles": "C"}},
        BuildingBlockCatalog(()),
        protected_targets={"route": "protected-target"},
        config=RoutePostprocessConfig(
            calculate_cost=False,
            max_variants_per_route=5,
        ),
    )

    assert result.variants == ()
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].stage == "bb_restoration"
    assert "requires 6 variants; limit is 5" in result.diagnostics[0].message


def test_pipeline_config_validates_target_sequence_mode() -> None:
    with pytest.raises(ValueError, match="target_protection_sequence_mode"):
        RoutePostprocessConfig(target_protection_sequence_mode="first")


def test_pipeline_forwards_optional_deterministic_mode(monkeypatch) -> None:
    import synplan.chem.reaction.routes.postprocess.pipeline as pipeline

    observed = []

    def restore(route, target, *, sequence_mode, max_variants):
        observed.append((sequence_mode, max_variants))
        return [dict(route)]

    def stereo(route, target, *, catalog):
        return dict(route)

    monkeypatch.setattr(pipeline, "restore_protected_target", restore)
    monkeypatch.setattr(pipeline, "restore_route_stereo", stereo)

    result = postprocess_routes(
        {"route": {"type": "mol", "smiles": "C"}},
        BuildingBlockCatalog(()),
        protected_targets={"route": "protected-target"},
        config=RoutePostprocessConfig(
            expand_deprotected=False,
            calculate_cost=False,
            target_protection_sequence_mode="deterministic",
        ),
    )

    assert observed == [("deterministic", 100)]
    assert len(result.variants) == 1
    assert result.ok
