"""Whole-route mapping must preserve occurrence identity and JSON connectivity."""

from copy import deepcopy

import pytest
from chython import smiles
from chython.containers import ReactionContainer

from synplan.chem.reaction.routes import Route, Step
from synplan.chem.reaction.routes.representation import (
    build_route_cgr,
    compose_sb_cgr,
    reactions_from_route_cgr,
    route_cgr_hash,
    route_cgrs_equal,
)
from synplan.chem.reaction.routes.representation.stereo import graph_digest


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def collision_route():
    first = smiles("[CH3:1][O:2][CH3:9]>>[CH3:1][OH:2].[CH4:9]")
    last = smiles("[CH3:1][OH:2].[CH3:9][Br:10]>>[CH3:1][O:2][CH3:9].[BrH:10]")
    last = ReactionContainer([first.products[0], last.reactants[1]], last.products)
    return Route((Step(first, first.products[0]), Step(last, last.products[0])))


def test_unrelated_carbons_with_equal_local_ids_stay_distinct():
    route = collision_route()
    original = deepcopy(route.to_json())
    result = build_route_cgr(route, include_reactions=True)
    check(result.ok, result.diagnostic)
    check(len(result.cgr) == 5, "two unrelated carbon atoms were merged")
    check(result.atom_mapping[0][9] > 10, "leaving carbon needs a fresh global ID")
    check(result.atom_mapping[1][9] == 9, "target carbon must keep its ID")
    check(route.to_json() == original, "composition edited its input")
    mapped = result.mapped_route
    check(
        mapped.steps[0].product is mapped.steps[1].reaction.reactants[0],
        "mapping broke the producer/consumer identity",
    )
    repeated = build_route_cgr(mapped, include_reactions=True)
    check(
        all(
            n == m
            for mapping in repeated.atom_mapping.values()
            for n, m in mapping.items()
        ),
        "already reconciled atoms were renumbered",
    )
    restored = Route.from_json(route.to_json(reconcile_atom_mapping=True))
    check(
        graph_digest(restored.route_cgr()) == graph_digest(result.cgr),
        "JSON round trip changed mapped chemistry",
    )
    check(
        graph_digest(route.route_cgr()) == graph_digest(result.cgr),
        "Route and exporter use different numbering",
    )
    rebuilt = build_route_cgr({0: reactions_from_route_cgr(result.cgr)}, 0)
    check(
        graph_digest(rebuilt.cgr) == graph_digest(result.cgr),
        "deconvolution/recomposition changed mapped chemistry",
    )


def test_reagent_with_partial_overlap_is_not_linked_to_an_earlier_product():
    first = smiles("[CH3:1][OH:2]>>[CH3:1][OH:2]")
    second = smiles("[CH3:1][OH:2]>>[CH4:1].[OH2:2]")
    second = ReactionContainer([first.products[0]], second.products)
    last = smiles("[CH4:1].[OH2:2]>>[CH3:1][OH:2]")
    last = ReactionContainer([second.products[0], last.reactants[1]], last.products)
    route = Route(
        (
            Step(first, first.products[0]),
            Step(second, second.products[0]),
            Step(last, last.products[0]),
        )
    )
    restored = Route.from_json(route.to_json(reconcile_atom_mapping=True))
    check(len(restored) == 3, "export inserted a spurious dependency")
    check(len(restored.leaves()) == len(route.leaves()), "export lost a reagent")
    check(
        graph_digest(restored.route_cgr()) == graph_digest(route.route_cgr()),
        "reagent collision corrupted chemistry",
    )


def branched_route(reverse):
    steps, intermediates = [], []
    for offset in (0, 10):
        reaction = smiles("[CH3:1][Br:2]>>[CH3:1][OH:3]")
        for molecule in reaction.molecules():
            molecule.remap({n: n + offset for n in molecule})
        steps.append(Step(reaction, reaction.products[0]))
        intermediates.append(reaction.products[0])
    target = smiles("[CH3:1][O:3][CH2:11][OH:13]")
    joining = ReactionContainer(
        intermediates[::-1] if reverse else intermediates, [target]
    )
    return Route((*steps, Step(joining, target)))


@pytest.mark.parametrize("reverse", [False, True])
def test_identical_molecules_on_separate_branches_roundtrip(reverse):
    route = branched_route(reverse)
    restored = Route.from_json(route.to_json(reconcile_atom_mapping=True))
    check(len(restored) == 3, "identical branches were merged")
    check(
        graph_digest(restored.route_cgr()) == graph_digest(route.route_cgr()),
        "JSON swapped the mapped branch occurrences",
    )
    cgr = route.route_cgr()
    rebuilt = build_route_cgr({0: reactions_from_route_cgr(cgr)}, 0).cgr
    check(route_cgrs_equal(cgr, rebuilt), "branched RouteCGR identity changed")
    check(route_cgr_hash(cgr) == route_cgr_hash(rebuilt), "branched CGR hash changed")
    for reactant in restored.steps[-1].reaction.reactants:
        check(
            sum(reactant is step.product for step in restored.steps[:-1]) == 1,
            "each reactant must reference its own producer",
        )


def test_selected_product_and_stereo_survive_reconciliation():
    reaction = smiles(
        "[CH3:1][C@@H:2]([OH:3])[CH2:4][Br:5]>>"
        "[BrH:5].[CH3:1][C@@H:2]([OH:3])[CH2:4][OH:6]"
    )
    chosen = next(p for p in reaction.products if p.has_atom(1))
    route = Route(
        (Step(reaction, chosen),), unresolved=frozenset({str(reaction.reactants[0])})
    )
    result = build_route_cgr(route, include_reactions=True)
    restored = Route.from_json(route.to_json(reconcile_atom_mapping=True))
    check(result.mapped_route.target == route.target, "wrong multi-product target")
    check(restored.target == route.target, "target stereo changed")
    check(restored.unresolved == route.unresolved, "stock verdict changed")
    check(
        graph_digest(compose_sb_cgr(restored.route_cgr()))
        == graph_digest(compose_sb_cgr(route.route_cgr())),
        "SB-CGR chemistry changed",
    )


def test_real_boceprevir_export_previously_failed_import():
    import json
    from pathlib import Path

    fixture = Path(__file__).parents[4] / "data/boceprevir_route_mapping.json"
    source = Route.from_json(json.loads(fixture.read_text())["route"])
    snapshot = deepcopy(source.to_json())
    restored = Route.from_json(source.to_json(reconcile_atom_mapping=True))
    check(len(restored) == len(source), "real export changed the synthesis steps")
    check(len(restored.leaves()) == len(source.leaves()), "real export lost a reagent")
    check(restored.target == source.target, "real export changed target stereo")
    check(
        graph_digest(restored.route_cgr()) == graph_digest(source.route_cgr()),
        "real export changed mapped chemistry",
    )
    check(source.to_json() == snapshot, "export modified the source Route")
