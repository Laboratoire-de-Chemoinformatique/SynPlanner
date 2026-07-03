"""Reconstruct route reaction dictionaries from RouteCGR metadata."""

from __future__ import annotations

from typing import Any

from chython import smiles as read_smiles
from chython.containers import CGRContainer, ReactionContainer

from synplan.chem.reaction.routes.io import make_json

ROUTE_RECONSTRUCTION_SCHEMA = "route-cgr-reactions-v1"


def attach_route_reconstruction_metadata(
    route_cgr: CGRContainer,
    reactions_dict: dict[int, ReactionContainer],
    *,
    route_metadata: dict[int, dict[str, Any]] | None = None,
    route_json: dict[str, Any] | None = None,
) -> CGRContainer:
    """Attach exact route-step reactions to a RouteCGR.

    Route-order and step-order labels encode where dynamics occurred in the
    composed graph, but they are not sufficient to recover every full per-step
    molecule context losslessly. The exact mapped reactions produced during
    composition are therefore stored on the RouteCGR itself so a downstream
    analysis can reconstruct the same tree JSON without keeping the source
    ``Tree`` object.
    """

    route_cgr.route_reconstruction_schema = ROUTE_RECONSTRUCTION_SCHEMA
    route_cgr.route_reaction_smiles = {
        int(step_id): format(reaction, "m")
        for step_id, reaction in sorted(reactions_dict.items())
    }
    route_cgr.route_reaction_metadata = route_metadata or {}
    route_cgr.route_json = route_json
    return route_cgr


def prepare_route_cgr_reconstruction(
    route_cgr: CGRContainer,
    reactions_dict: dict[int, ReactionContainer],
    route_id: int,
    *,
    tree: Any | None = None,
    route_metadata: dict[int, dict[str, Any]] | None = None,
) -> CGRContainer:
    """Attach reconstruction metadata to an already composed RouteCGR.

    This keeps RouteCGR composition independent from JSON round-trip
    bookkeeping. Call this only in workflows that need to deconvolute a
    composed RouteCGR back into its exact route representation.
    """

    routes_json = make_json({int(route_id): reactions_dict}, tree=tree)
    return attach_route_reconstruction_metadata(
        route_cgr,
        reactions_dict,
        route_metadata=route_metadata,
        route_json=routes_json[int(route_id)],
    )


def reactions_from_route_cgr(route_cgr: CGRContainer) -> dict[int, ReactionContainer]:
    """Return the mapped reaction sequence embedded in a RouteCGR.

    Raises
    ------
    ValueError
        If the RouteCGR was not produced by a composer that stores exact
        reconstruction metadata.
    """

    schema = getattr(route_cgr, "route_reconstruction_schema", None)
    if schema != ROUTE_RECONSTRUCTION_SCHEMA:
        raise ValueError(
            "RouteCGR does not carry exact reconstruction metadata. "
            "Recompose it with compose_route_cgr(..., preserve_transient_bonds=True) "
            "from a SynPlanner version that stores route reconstruction metadata."
        )

    reaction_smiles = getattr(route_cgr, "route_reaction_smiles", None)
    if not isinstance(reaction_smiles, dict) or not reaction_smiles:
        raise ValueError("RouteCGR reconstruction metadata has no reactions")

    return {
        int(step_id): read_smiles(smiles)
        for step_id, smiles in sorted(reaction_smiles.items(), key=lambda item: int(item[0]))
    }


def routes_dict_from_route_cgrs(
    route_cgrs: dict[int, CGRContainer],
) -> dict[int, dict[int, ReactionContainer]]:
    """Convert ``route_id -> RouteCGR`` into ``route_id -> step_id -> Reaction``."""

    return {
        int(route_id): reactions_from_route_cgr(route_cgr)
        for route_id, route_cgr in sorted(route_cgrs.items())
    }

def route_json_from_route_cgrs(route_cgrs: dict[int, CGRContainer]) -> dict[int, Any]:
    """Return exact route JSON trees embedded during RouteCGR composition.

    This preserves sibling order and route metadata for exact JSON round-trips.
    Use :func:`routes_dict_from_route_cgrs` when ReactionContainer objects are
    needed instead.
    """

    routes_json = {}
    for route_id, route_cgr in sorted(route_cgrs.items()):
        schema = getattr(route_cgr, "route_reconstruction_schema", None)
        if schema != ROUTE_RECONSTRUCTION_SCHEMA:
            raise ValueError(
                "RouteCGR does not carry exact reconstruction metadata. "
                "Recompose it with compose_route_cgr(..., preserve_transient_bonds=True)."
            )
        route_json = getattr(route_cgr, "route_json", None)
        if route_json is None:
            raise ValueError("RouteCGR reconstruction metadata has no route JSON")
        routes_json[int(route_id)] = route_json
    return routes_json
