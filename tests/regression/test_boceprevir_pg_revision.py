"""Replay a real bounded Boceprevir search without models or planning in CI."""

import json
from pathlib import Path

import pytest

from synplan.chem.reaction.routes import Route
from synplan.chem.reaction.routes.quality.protection import (
    ProtectionRevisionConfig,
    ProtectionRouteReviser,
)
from synplan.chem.reaction.routes.quality.protection.revision import mapped_cgr_state
from synplan.chem.reaction.routes.representation import (
    build_route_cgr,
    compose_sb_cgr,
    reactions_from_route_cgr,
    route_cgr_graph,
)


def test_recorded_boceprevir_route_carries_pg_between_transformations():
    example = Path(__file__).parents[1] / "data/boceprevir_pg_example.json"
    recorded = json.loads(example.read_text())
    config = recorded["run_metadata"]["config"]
    assert (config["max_iterations"], config["max_time"], config["max_depth"]) == (
        300,
        120,
        7,
    )
    assert recorded["run_metadata"]["seed"] == 42
    assert recorded["selected_route_id"] is not None
    route = Route.from_json(recorded["routes"][0]["route"])
    original = route.to_json()
    before = route.route_cgr()
    result = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=1)
    ).revise_route(route)
    assert result.accepted, result.diagnostics
    assert len(route) == 5
    assert len(result.route) == 7
    assert result.original_score == pytest.approx(0.7)
    assert result.revised_score == pytest.approx(13 / 14)
    assert result.original_to_final == {0: 0, 1: 2, 2: 3, 3: 5, 4: 6}
    action = result.actions[0]
    assert action.anchor_atom == 33
    assert action.new_step_ids == (1, 2, 4)
    assert action.protected_step_ids == (2, 3)
    after = result.route.route_cgr()
    assert mapped_cgr_state(after.substructure(set(before))) == mapped_cgr_state(before)
    assert mapped_cgr_state(compose_sb_cgr(before)) == mapped_cgr_state(
        compose_sb_cgr(after)
    )
    assert min(set(after) - set(before)) > max(before)
    assert result.route.target == route.target
    for step in action.protected_step_ids:
        assert set(action.added_atoms) & set(result.route.steps[step].product)
    assert not set(action.added_atoms) & set(
        result.route.steps[action.new_step_ids[2]].product
    )
    assert route.to_json() == original
    restored = Route.from_json(result.route.to_json(reconcile_atom_mapping=True))
    assert len(restored) == len(result.route)
    assert restored.target == route.target
    assert mapped_cgr_state(restored.route_cgr()) == mapped_cgr_state(after)
    for cgr in (before, after):
        rebuilt = build_route_cgr({0: reactions_from_route_cgr(cgr)}, 0).cgr
        # Compare all mapped graph labels, including depths and step orders.
        # Canonical hashing can exceed its existing symmetry limit on this route.
        if route_cgr_graph(cgr) != route_cgr_graph(rebuilt):
            pytest.fail("Boceprevir RouteCGR graph or depth annotations changed")
