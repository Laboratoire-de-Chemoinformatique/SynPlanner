from __future__ import annotations


def test_moved_protection_converter_is_importable():
    from scripts.convert_protection_source_data import main

    assert callable(main)


def test_main_branch_leaving_groups_path_reexports_pseudo_atoms():
    from synplan.chem.reaction.routes.clustering.pseudo_atoms import (
        DynamicX as CanonicalDynamicX,
    )
    from synplan.chem.reaction_routes.leaving_groups import DynamicX as LegacyDynamicX

    assert LegacyDynamicX is CanonicalDynamicX


def test_main_branch_visualisation_path_reexports_helpers():
    from synplan.chem.reaction.routes.visualisation import (
        WideBondDepictCGR as CanonicalWideBondDepictCGR,
    )
    from synplan.chem.reaction.routes.visualisation import (
        cgr_display as canonical_cgr_display,
    )
    from synplan.chem.reaction_routes.visualisation import (
        WideBondDepictCGR as LegacyWideBondDepictCGR,
    )
    from synplan.chem.reaction_routes.visualisation import (
        cgr_display as legacy_cgr_display,
    )

    assert LegacyWideBondDepictCGR is CanonicalWideBondDepictCGR
    assert legacy_cgr_display is canonical_cgr_display


def test_visualisation_accepts_legacy_cgr_wrapper():
    from chython import smiles

    from synplan.chem.reaction.routes.representation import compose_route_cgr
    from synplan.chem.reaction.routes.visualisation import cgr_display

    routes = {
        1: {
            0: smiles("[CH3:1].[CH3:2][Cl:3]>>[CH3:1][CH3:2].[ClH:3]"),
        }
    }
    cgr = compose_route_cgr(routes, 1)["cgr"]

    svg = cgr_display({"cgr": cgr})
    assert "<svg" in svg
