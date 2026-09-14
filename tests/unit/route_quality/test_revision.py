"""Contracts for graph-linked protecting-group revision."""

from copy import deepcopy

import pytest
from chython import smiles
from chython.containers import ReactionContainer

from synplan.chem.reaction.routes import Route, Step
from synplan.chem.reaction.routes.quality.protection import (
    ProtectionRevisionConfig,
    ProtectionRouteReviser,
)
from synplan.chem.reaction.routes.quality.protection.revision import mapped_cgr_state
from synplan.chem.reaction.routes.representation import (
    build_route_cgr,
    compose_sb_cgr,
    reactions_from_route_cgr,
    route_cgr_hash,
    route_cgrs_equal,
)
from synplan.chem.utils import molecule_key


def esterification(offset=0):
    reaction = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH2:7][OH:8]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[OH2:4]"
    )
    if offset:
        for molecule in reaction.molecules():
            molecule.remap({n: n + offset for n in molecule})
    return reaction


def chain(oxidize=False):
    first = esterification()
    second = smiles(
        "[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH2:7][OH:8].[Cl:9]>>"
        "[CH3:1][C:2](=[O:3])[O:5][CH:6]([Cl:9])[CH2:7][OH:8]"
    )
    second = ReactionContainer(
        [first.products[0], second.reactants[1]], second.products
    )
    steps = [Step(first, first.products[0]), Step(second, second.products[0])]
    if oxidize:
        last = smiles(
            "[CH3:1][C:2](=[O:3])[O:5][CH:6]([Cl:9])[CH2:7][OH:8]>>"
            "[CH3:1][C:2](=[O:3])[O:5][CH:6]([Cl:9])[CH:7]=[O:8]"
        )
        last = ReactionContainer([second.products[0]], last.products)
        steps.append(Step(last, last.products[0]))
    return Route(tuple(steps))


@pytest.fixture(scope="module")
def reviser():
    return ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=1)
    )


def assert_invariants(original, result):
    assert result.accepted, result.diagnostics
    before, after = original.route_cgr(), result.route.route_cgr()
    assert mapped_cgr_state(after.substructure(set(before))) == mapped_cgr_state(before)
    assert min(set(after) - set(before)) > max(before)
    assert mapped_cgr_state(compose_sb_cgr(before)) == mapped_cgr_state(
        compose_sb_cgr(after)
    )
    assert result.route.target == original.target
    assert result.revised_score > result.original_score
    made = {id(step.product) for step in result.route}
    consumed = {id(m) for step in result.route for m in step.reaction.reactants}
    assert made - consumed == {id(result.route.target)}
    for old, new in result.original_to_final.items():
        original_cgr = ~original.steps[old].reaction
        revised_cgr = ~result.route.steps[new].reaction
        assert mapped_cgr_state(
            revised_cgr.substructure(set(original_cgr))
        ) == mapped_cgr_state(original_cgr)


def test_single_step_pair_preserves_input_and_reagent_verdict(reviser):
    reaction = esterification()
    route = Route((Step(reaction, reaction.products[0]),))
    before = deepcopy(route.to_json())
    result = reviser.revise_route(route)
    assert_invariants(route, result)
    assert result.actions[0].anchor_atom == 8
    assert result.actions[0].new_step_ids == (0, 1, 2)
    assert result.original_to_final == {0: 1}
    assert route.to_json() == before
    added = [m for m in result.route.leaves() if m.meta.get("added_protecting_group")]
    assert added
    assert all(molecule_key(m) in result.route.unresolved for m in added)


@pytest.mark.parametrize("oxidize", [False, True])
def test_carry_across_two_transformations_and_deprotect(reviser, oxidize):
    route = chain(oxidize)
    result = reviser.revise_route(route)
    assert_invariants(route, result)
    action = result.actions[0]
    assert action.new_step_ids == (0, 1, 3)
    assert action.protected_step_ids == (1, 2)
    assert result.original_to_final == ({0: 1, 1: 2, 2: 4} if oxidize else {0: 1, 1: 2})
    for index in action.protected_step_ids:
        assert set(action.added_atoms) & set(result.route.steps[index].product)
    assert not set(action.added_atoms) & set(result.route.steps[3].product)
    if oxidize:
        assert (
            result.route.steps[4].reaction.reactants[0] is result.route.steps[3].product
        )


def test_roundtrips_and_revision_metadata(reviser):
    result = reviser.revise_route(chain(True))
    assert result.accepted
    restored = Route.from_json(result.route.to_json())
    assert len(restored) == 5
    assert (
        restored.steps[0].reaction.meta["protection_revision"][0]["role"]
        == "protection"
    )
    assert (
        restored.steps[3].reaction.meta["protection_revision"][0]["role"]
        == "deprotection"
    )
    assert restored.unresolved == result.route.unresolved
    cgr = result.route.route_cgr()
    rebuilt = build_route_cgr({0: reactions_from_route_cgr(cgr)}, 0).cgr
    if not route_cgrs_equal(cgr, rebuilt) or route_cgr_hash(cgr) != route_cgr_hash(
        rebuilt
    ):
        pytest.fail("Protected RouteCGR lost exact identity on recomposition")
    assert mapped_cgr_state(rebuilt) == mapped_cgr_state(cgr)
    for n in cgr:
        assert (
            rebuilt.atom(n).route_atom_step_states == cgr.atom(n).route_atom_step_states
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_convergent_identical_branches_and_multiple_actions(reverse):
    left, right = esterification(), esterification(10)
    p1, p2 = left.products[0], right.products[0]
    product = p1.union(p2)
    product.add_bond(6, 16, 1)
    joining = ReactionContainer([p2, p1] if reverse else [p1, p2], [product])
    route = Route((Step(left, p1), Step(right, p2), Step(joining, product)))
    result = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=2)
    ).revise_route(route)
    assert_invariants(route, result)
    assert len(result.actions) == 2
    assert {a.anchor_atom for a in result.actions} == {8, 18}
    assert {a.original_step_id for a in result.actions} == {0, 1}
    assert set(result.actions[0].added_atoms).isdisjoint(result.actions[1].added_atoms)
    for action in result.actions:
        p, t, d = action.new_step_ids
        assert p < t < d
        assert t == result.original_to_final[action.original_step_id]
        assert (
            result.route.steps[p].reaction.meta["protection_revision"][-1]["role"]
            == "protection"
        )
        assert any(
            e["role"] == "deprotection"
            for e in result.route.steps[d].reaction.meta["protection_revision"]
        )
        other = result.original_to_final[1 - action.original_step_id]
        assert not set(action.added_atoms) & set(result.route.steps[other].product)


def test_selected_product_need_not_be_first(reviser):
    reaction = esterification()
    selected = reaction.products[0]
    reaction = ReactionContainer(reaction.reactants, tuple(reversed(reaction.products)))
    route = Route((Step(reaction, selected),))
    result = reviser.revise_route(route)
    assert_invariants(route, result)
    assert result.actions[0].anchor_atom == 8


def test_no_supported_group_is_explicit(reviser):
    r = smiles(
        "[O:20]=[C:1]1[CH2:2][CH2:3][C:4](=[O:5])[CH2:6][CH2:7]1>>"
        "[OH:20][CH:1]1[CH2:2][CH2:3][C:4](=[O:5])[CH2:6][CH2:7]1"
    )
    route = Route((Step(r, r.products[0]),))
    result = reviser.revise_route(route)
    assert not result.accepted
    assert result.route is route
    assert any(d.reason == "no_supported_protecting_group" for d in result.diagnostics)


def test_threshold_and_limit():
    route = chain()
    result = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(min_score_improvement=1.0)
    ).revise_route(route)
    assert not result.accepted
    assert any(d.reason == "score_not_improved" for d in result.diagnostics)
    limited = ProtectionRouteReviser.from_config(
        ProtectionRevisionConfig(max_revisions_per_route=0)
    ).revise_route(route)
    assert limited.route is route
    assert any(d.reason == "revision_limit_reached" for d in limited.diagnostics)


@pytest.mark.parametrize("reported", [False, True])
def test_invalid_candidate_is_rejected(reviser, monkeypatch, reported):
    def reject(*args):
        if reported:
            return "sb_cgr_changed"
        raise ValueError("invalid composition")

    monkeypatch.setattr(reviser, "_validate", reject)
    route = chain()
    result = reviser.revise_route(route)
    assert result.route is route
    assert not result.accepted
    reason = "sb_cgr_changed" if reported else "candidate_invalid"
    assert any(d.reason == reason for d in result.diagnostics)


def test_scanner_cache_keeps_actual_atom_handles(reviser):
    first, second = esterification(), esterification(100)
    scans = [
        reviser.scanner.scan_route(Route((Step(r, r.products[0]),)), detailed=True)
        for r in (first, second)
    ]
    assert scans[0].interactions[0].anchor_atom == 8
    assert scans[1].interactions[0].anchor_atom == 108
    assert scans[1].interactions[0].fg_atoms == (108,)


def test_unknown_interactions_remain_unknown(reviser, monkeypatch):
    monkeypatch.setattr(
        reviser.scanner._incompatibility, "lookup", lambda *args: "unknown"
    )
    result = reviser.revise_route(chain())
    assert not result.accepted
    assert result.remaining_interactions
    assert all(i.severity == "unknown" for i in result.remaining_interactions)
    assert any(d.reason == "interaction_unknown" for d in result.diagnostics)


def test_route_ids(reviser):
    results = reviser.revise_routes({7: chain()})
    assert list(results) == [7]
    assert results[7].accepted


def test_stereo_and_metadata_survive_revision(reviser):
    from synplan.chem.reaction.routes import RouteProvenance, StepOrigin
    from synplan.chem.stereo import stereo_requirements, stereo_sign

    reaction = smiles(
        "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][C@H:6]([CH3:9])[CH2:7][OH:8]>>"
        "[CH3:1][C:2](=[O:3])[O:5][C@H:6]([CH3:9])[CH2:7][OH:8].[OH2:4]",
        ignore_stereo=False,
    )
    reaction.meta["custom"] = {"nested": ["kept"]}
    origin = StepOrigin(rule_key="original-rule")
    provenance = RouteProvenance(search_score=0.75, tree_node_id=19)
    route = Route(
        (Step(reaction, reaction.products[0], origin=origin),), provenance=provenance
    )
    requirements = stereo_requirements(route.target)
    assert requirements
    result = reviser.revise_route(route)
    assert_invariants(route, result)
    assert result.route.provenance == provenance
    transformed = result.route.steps[result.original_to_final[0]]
    assert transformed.origin == origin
    assert transformed.reaction.meta["custom"] == {"nested": ["kept"]}
    for requirement in requirements:
        assert stereo_sign(transformed.product, requirement) == requirement.sign
    transformed.reaction.meta["custom"]["nested"].append("new")
    assert reaction.meta["custom"]["nested"] == ["kept"]


def test_composition_failure_returns_original(reviser, monkeypatch):
    from synplan.chem.reaction.routes.contracts import (
        RouteCGRBuildResult,
        RouteDiagnostic,
    )
    from synplan.chem.reaction.routes.quality.protection import revision

    monkeypatch.setattr(
        revision,
        "build_route_cgr",
        lambda *a, **k: RouteCGRBuildResult(
            route_id=None,
            diagnostic=RouteDiagnostic(
                None, "route_cgr_composition", "invalid composition"
            ),
        ),
    )
    route = chain()
    result = reviser.revise_route(route)
    assert result.route is route
    assert not result.accepted
    assert result.diagnostics[0].reason == "baseline_invalid"


def test_incomplete_scan_does_not_accept_revision(reviser, monkeypatch):
    from synplan.chem.reaction.routes.quality.protection import CompetingScanResult

    scan = reviser.scanner.scan_route

    def incomplete(route, *, detailed=False):
        if not detailed:
            return scan(route)
        return CompetingScanResult(
            interactions=[], halogen_count=0, processed_steps=(), skipped_steps=(0,)
        )

    monkeypatch.setattr(reviser.scanner, "scan_route", incomplete)
    route = chain()
    result = reviser.revise_route(route)
    assert result.route is route
    assert not result.accepted
    assert any(d.reason == "scan_step_skipped" for d in result.diagnostics)


def test_original_non_target_atom_conflict_is_normalized(reviser):
    # Atom 4 is oxygen in step 0 but an unrelated chlorine in step 1.
    # The baseline RouteCGR allocates that oxygen a fresh ID. Revision must
    # preserve the baseline's ID, not reuse the local reaction's ambiguous 4.
    route = chain()
    r0, r1 = [s.reaction for s in route]
    reactants = [m.copy() for m in r1.reactants]
    products = [m.copy() for m in r1.products]
    for m in [*reactants, *products]:
        if m.has_atom(9):
            m.remap({9: 4})
    reactants[0] = r0.products[0]
    second = ReactionContainer(reactants, products)
    route = Route((route.steps[0], Step(second, products[0])))
    before = route.route_cgr()
    result = reviser.revise_route(route)
    assert result.accepted, result.diagnostics
    after = result.route.route_cgr()
    assert mapped_cgr_state(after.substructure(set(before))) == mapped_cgr_state(before)
    assert min(set(after) - set(before)) > max(before)


def test_clean_route_is_unchanged(reviser):
    reaction = smiles("[CH3:1][CH3:2]>>[CH3:1][CH3:2]")
    route = Route((Step(reaction, reaction.products[0]),))
    result = reviser.revise_route(route)
    assert result.route is route
    assert result.original_score == result.revised_score == 1.0
    assert not result.actions
    assert not result.remaining_interactions
