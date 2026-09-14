"""Exact depth identity survives deconvolution without trusting edited steps."""

import json
import pickle
from copy import deepcopy

import pytest
from chython import smiles
from chython.containers import ReactionContainer

from synplan.chem.reaction.routes import Route, Step
from synplan.chem.reaction.routes.representation import (
    build_route_cgr,
    reactions_from_route_cgr,
    route_cgr_hash,
    route_cgrs_equal,
)


def linear_route():
    reactions = [
        smiles("[CH3:1][Br:2]>>[CH3:1][OH:3]"),
        smiles("[CH3:1][OH:3]>>[CH2:1]=[O:3]"),
        smiles("[CH2:1]=[O:3]>>[CH:1](=[O:3])[OH:4]"),
    ]
    for index in range(1, len(reactions)):
        reactions[index] = ReactionContainer(
            [reactions[index - 1].products[0]], reactions[index].products
        )
    return Route(tuple(Step(r, r.products[0]) for r in reactions))


def check_equal(before, after):
    if not route_cgrs_equal(before, after):
        pytest.fail("Exact RouteCGR identity changed")
    if route_cgr_hash(before) != route_cgr_hash(after):
        pytest.fail("RouteCGR hashes changed")


@pytest.mark.parametrize(
    "copy_kind", ["original", "copy", "deepcopy", "pickle", "flush"]
)
def test_exact_depth_roundtrip_survives_container_lifecycle(copy_kind):
    route = linear_route()
    source_json = deepcopy(route.to_json())
    original = route.route_cgr()
    cgr = original
    if copy_kind == "copy":
        cgr = cgr.copy()
    elif copy_kind == "deepcopy":
        cgr = deepcopy(cgr)
    elif copy_kind == "pickle":
        cgr = pickle.loads(pickle.dumps(cgr))
    elif copy_kind == "flush":
        cgr.flush_cache()
    for _ in range(2):
        reactions = reactions_from_route_cgr(cgr)
        for reaction in reactions.values():
            # Metadata remains valid through the JSON-compatible representation.
            reaction.meta.update(json.loads(json.dumps(reaction.meta)))
        cgr = build_route_cgr({0: reactions}, 0).cgr
        check_equal(original, cgr)
    if route.to_json() != source_json:
        pytest.fail("Composition changed the input Route")


@pytest.mark.parametrize(
    "edit", ["chemistry", "replace", "reorder", "remove", "append", "missing"]
)
def test_edited_dictionary_does_not_reuse_recorded_depths(edit):
    reactions = list(reactions_from_route_cgr(linear_route().route_cgr()).values())
    if edit == "chemistry":
        first = reactions[0]
        reactions[0] = ReactionContainer(
            [*first.reactants, smiles("[Na+:90]")], first.products, meta=first.meta
        )
    elif edit == "replace":
        # A replacement may carry valid metadata from a different source route.
        alternate = linear_route()
        first = alternate.steps[0]
        changed = ReactionContainer(
            [*first.reaction.reactants, smiles("[Na+:90]")], first.reaction.products
        )
        alternate = Route((Step(changed, first.product), *alternate.steps[1:]))
        reactions[0] = reactions_from_route_cgr(alternate.route_cgr())[0]
    elif edit == "reorder":
        reactions[0], reactions[1] = reactions[1], reactions[0]
    elif edit == "remove":
        reactions.pop()
    elif edit == "append":
        reactions.append(reactions[-1].copy())
    elif edit == "missing":
        reactions[0].meta.pop("route_cgr_depth")
    edited = build_route_cgr({0: dict(enumerate(reactions))}, 0).cgr
    for reaction in reactions:
        reaction.meta.pop("route_cgr_depth", None)
    inferred = build_route_cgr({0: dict(enumerate(reactions))}, 0).cgr
    check_equal(edited, inferred)


def test_older_cgr_without_depth_records_keeps_legacy_fallback():
    cgr = linear_route().route_cgr()
    del cgr.route_step_depths
    reactions = reactions_from_route_cgr(cgr)
    if any("route_cgr_depth" in r.meta for r in reactions.values()):
        pytest.fail("Deconvolution invented source depths for an older CGR")
    rebuilt = build_route_cgr({0: reactions}, 0)
    if not rebuilt.ok:
        pytest.fail(str(rebuilt.diagnostic))


@pytest.mark.parametrize("copy", [False, True])
def test_remapped_cgr_keeps_depths(copy):
    cgr = linear_route().route_cgr()
    remapped = cgr.remap({n: n + 100 for n in cgr}, copy=copy)
    if copy:
        cgr = remapped
    rebuilt = build_route_cgr({0: reactions_from_route_cgr(cgr)}, 0).cgr
    check_equal(cgr, rebuilt)
