"""Origin-verified direct RouteCGR composition."""

from chython import smiles

from synplan.chem.reaction.reactor import ATOM_ORIGINS_META
from synplan.chem.reaction.routes.representation import (
    route_cgr,
    route_cgrs_equal,
    routes_dict_from_route_cgrs,
)
from synplan.chem.reaction.routes.representation.container import RouteCGRContainer
from synplan.chem.reaction.routes.representation.route_cgr import build_route_cgr


def _native_routes(*, collision: bool = False):
    chlorine_map = 2 if collision else 3
    bromine_map = 3 if collision else 4
    routes = {
        1: {
            0: smiles(f"[CH3:2][Br:{bromine_map}]>>[CH4:2]"),
            1: smiles(f"[CH3:1][Cl:{chlorine_map}]>>[CH4:1]"),
            2: smiles("[CH4:1].[CH4:2]>>[CH3:1][CH3:2]"),
        }
    }
    for reaction in routes[1].values():
        for molecule in (*reaction.reactants, *reaction.products):
            origins = {}
            for atom_map, atom in molecule.atoms():
                if collision and atom_map == 2 and atom.atomic_number == 17:
                    origins[atom_map] = 3
                else:
                    origins[atom_map] = atom_map
            molecule.meta[ATOM_ORIGINS_META] = origins
    return routes


def test_direct_fold_matches_legacy_and_deconvolves():
    routes = _native_routes()
    direct = build_route_cgr(
        routes,
        1,
        include_reactions=True,
        reconcile_atom_mapping=False,
    )
    legacy = build_route_cgr(
        routes,
        1,
        include_reactions=True,
        reconcile_atom_mapping=True,
    )

    assert direct.ok and legacy.ok
    assert route_cgrs_equal(direct.cgr, legacy.cgr)
    restored = routes_dict_from_route_cgrs({1: direct.cgr})[1]
    assert list(restored) == [0, 1, 2]


def test_direct_fold_rejects_same_element_or_element_agnostic_map_collision():
    routes = {
        1: {
            0: smiles("[CH3:2][CH3:4]>>[CH4:2]"),
            # This branch introduces a different carbon as map 2. Element-only
            # CGR checks cannot distinguish it from the map-2 carbon above.
            1: smiles("[CH3:1][CH3:2]>>[CH4:1]"),
            2: smiles("[CH4:1].[CH4:2]>>[CH3:1][CH3:2]"),
        }
    }
    for step, reaction in routes[1].items():
        for molecule in (*reaction.reactants, *reaction.products):
            origins = {atom_map: atom_map for atom_map in molecule}
            if step == 1 and molecule in reaction.reactants and 2 in set(molecule):
                origins[2] = 3
            molecule.meta[ATOM_ORIGINS_META] = origins
    result = build_route_cgr(routes, 1, reconcile_atom_mapping=False)

    assert not result.ok
    assert result.diagnostic.exception_type == "ValueError"
    assert "atom map 2 refers to origins" in result.diagnostic.message


def test_strict_direct_fold_rejects_missing_origins():
    routes = _native_routes()
    routes[1][1].reactants[0].meta.pop(ATOM_ORIGINS_META)

    result = build_route_cgr(routes, 1, reconcile_atom_mapping=False)

    assert not result.ok
    assert "requires atom-origin metadata" in result.diagnostic.message


def test_automatic_native_route_never_calls_legacy_reconciliation(monkeypatch):
    routes = _native_routes()

    def legacy_should_not_run(*args, **kwargs):
        raise AssertionError("native route entered legacy reconciliation")

    monkeypatch.setattr(
        "synplan.chem.reaction.routes.representation.route_cgr."
        "_compose_route_cgr_legacy",
        legacy_should_not_run,
    )
    result = build_route_cgr(routes, 1)

    assert result.ok
    assert isinstance(result.cgr, RouteCGRContainer)


def test_automatic_native_route_extracts_and_validates_once(monkeypatch):
    routes = _native_routes()
    calls = {"extract": 0, "validate": 0}

    original_extract = route_cgr._route_reactions
    original_validate = route_cgr._validate_atom_origins

    def counted_extract(*args, **kwargs):
        calls["extract"] += 1
        return original_extract(*args, **kwargs)

    def counted_validate(*args, **kwargs):
        calls["validate"] += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(route_cgr, "_route_reactions", counted_extract)
    monkeypatch.setattr(route_cgr, "_validate_atom_origins", counted_validate)

    result = build_route_cgr(routes, 1)

    assert result.ok
    assert calls == {"extract": 1, "validate": 1}


def test_automatic_direct_failure_warns_and_falls_back_once(caplog):
    routes = _native_routes(collision=True)

    with caplog.at_level(
        "WARNING",
        logger="synplan.chem.reaction.routes.representation.route_cgr",
    ):
        result = build_route_cgr(routes, 1)

    assert result.ok
    fallback_records = [
        record for record in caplog.records if "falling back" in record.message
    ]
    assert len(fallback_records) == 1
