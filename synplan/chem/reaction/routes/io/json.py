import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from chython import smiles as read_smiles
from chython.exceptions import InvalidAromaticRing

from synplan.chem.reaction.routes.contracts import (
    RouteDiagnostic,
    RouteExportError,
    RouteExportResult,
)
from synplan.chem.reaction.routes.io.metadata import (
    reaction_metadata as _reaction_metadata,
)
from synplan.chem.reaction.routes.io.metadata import (
    restore_reaction_metadata as _restore_reaction_metadata,
)

logger = logging.getLogger(__name__)
MoleculeInStock = Callable[[Any], bool | Mapping[str, Any]]


def _route_tree_has_null_node(node) -> bool:
    """Return True if the assembled route tree contains a ``None`` node.

    ``build_mol_node`` returns ``None`` when a route is malformed (e.g. a node
    holding more than one molecule). Such a ``None`` ends up either as the route
    root or nested in a ``children`` list, which would serialize to a JSON
    ``null`` child and corrupt the route.
    """
    return node is None or any(
        _route_tree_has_null_node(child) for child in node.get("children", ())
    )


def _route_molecule_smiles(mol) -> str:
    """Return the route-IO molecule string using the existing preparation flow."""
    try:
        mol.kekule()
        mol.implicify_hydrogens()
        mol.thiele()
    except InvalidAromaticRing:
        # Keep serializing the original molecule string when aromatic
        # preparation fails; route export should remain best-effort.
        pass
    return str(mol)


def _collect_reactions(tree):
    """
    Traverse a reaction tree in post-order and collect all ReactionContainers.
    Returns a dict mapping each reaction's new step ID (0, 1, …) to its container.
    """
    rxn_list = []

    def recurse(node):
        if not isinstance(node, dict):
            return
        for child in node.get("children", []) or []:
            recurse(child)
        if node.get("type") == "reaction":
            reaction = read_smiles(node["smiles"])
            _restore_reaction_metadata(reaction, node.get("meta"))
            rxn_list.append(reaction)

    recurse(tree)
    return {i: rxn for i, rxn in enumerate(rxn_list)}


def make_dict(routes_json):
    """
    routes_json : dict or list of tree-dicts as produced by make_json()

    Returns a dict mapping each route index (0, 1, …) to a sub-dict
    of {new_step_id: ReactionContainer}, where the step IDs run
    from the earliest reaction (0) up to the final (max).
    """
    routes_dict = {}

    # Normalize to iterable of (route_idx, tree)
    if isinstance(routes_json, dict):
        items = ((int(k), v) for k, v in routes_json.items())
    else:
        items = enumerate(routes_json)

    for route_idx, tree in items:
        try:
            routes_dict[int(route_idx)] = _collect_reactions(tree)
        except Exception as e:
            logger.warning("Error processing route %s: %s", route_idx, e)

    return routes_dict


def read_routes_json(file_path="routes.json", to_dict=False):
    with open(file_path) as file:
        routes_json = json.load(file)
    if to_dict:
        return make_dict(routes_json)
    return routes_json


def _make_json_v1(
    routes_dict,
    keep_ids=True,
    molecule_in_stock: MoleculeInStock | None = None,
    route_metadata: dict[int, dict[int, dict[str, Any]]] | None = None,
):
    """
    Convert routes into a nested JSON tree of reaction and molecule nodes.

    Args:
        routes_dict (dict[int, dict[int, Reaction]]): Mapping route IDs to steps (step_id -> Reaction).
        keep_ids (bool): If True, returns a dict mapping route IDs to trees; otherwise returns a list.
        route_metadata (dict | None): Optional per-route metadata mapping
            ``route_id -> step_id -> metadata``.
        molecule_in_stock (callable | None): Optional molecule membership callback.

    Returns:
        list or dict: JSON-like tree(s) of routes.
    """
    # Prepare output
    all_routes = {} if keep_ids else []

    def molecule_fields(molecule, *, fallback: bool) -> dict[str, Any]:
        if molecule_in_stock is None:
            return {"in_stock": fallback}
        result = molecule_in_stock(molecule)
        if isinstance(result, Mapping):
            return {"in_stock": True, "bb": dict(result)}
        return {"in_stock": bool(result)}

    for route_id, steps in routes_dict.items():
        if not steps:
            continue
        route_step_metadata = (
            route_metadata.get(route_id) if route_metadata is not None else None
        )
        try:
            # Determine target molecule atoms from the final step of this route
            final_step = max(steps)
            target = steps[final_step].products[0]
            atom_nums = set(target._atoms.keys())

            # Precompute canonical SMILES and producer mapping for all products
            prod_map = {}  # smiles -> list of step_ids
            for sid, rxn in steps.items():
                for prod in rxn.products:
                    s = _route_molecule_smiles(prod)
                    prod_map.setdefault(s, []).append(sid)
        except Exception as e:
            logger.warning("Error processing route %s: %s", route_id, e)
            continue

        def build_mol_node(sid, want_react=None, _steps=steps, _atom_nums=atom_nums):
            """Select the product fragment of step ``sid`` and recurse into its reaction.

            ``want_react`` is the consuming reactant of the next step (set by
            ``build_reaction_node``). Selection is structural first: pick the
            product fragment that *is* the consuming reactant by chython
            structural equality. This recovers routes whose per-step atom-number
            chaining leaves the relevant fragment numbered disjoint from the
            target. When no reactant context is available (the route root), or
            no fragment matches structurally, fall back to atom-number overlap
            with the target.
            """
            rxn = _steps[sid]
            product = None
            if want_react is not None:
                product = next((p for p in rxn.products if p == want_react), None)
            if product is None:
                product = next(
                    (p for p in rxn.products if _atom_nums & p._atoms.keys()), None
                )
            if product is not None:
                return {
                    "type": "mol",
                    "smiles": _route_molecule_smiles(product),
                    "children": [build_reaction_node(sid)],
                    **molecule_fields(product, fallback=False),
                }
            # Neither structural identity nor atom-number overlap matched: route
            # is genuinely unrecoverable; the drop guard in make_json handles it.
            return None

        def build_reaction_node(
            sid,
            _steps=steps,
            _route_step_metadata=route_step_metadata,
            _prod_map=prod_map,
        ):
            """Build reaction node and recurse into reactant molecule nodes."""
            rxn = _steps[sid]
            node = {"type": "reaction", "smiles": format(rxn, "m"), "children": []}
            reaction_metadata = _reaction_metadata(rxn)
            if reaction_metadata:
                node["meta"] = reaction_metadata
            if _route_step_metadata and sid in _route_step_metadata:
                node.update(_route_step_metadata[sid])

            for react in rxn.reactants:
                r_smi = _route_molecule_smiles(react)
                # Look up any prior step producing this reactant
                prior = [ps for ps in _prod_map.get(r_smi, []) if ps < sid]
                if prior:
                    node["children"].append(
                        build_mol_node(max(prior), want_react=react)
                    )
                else:
                    node["children"].append(
                        {
                            "type": "mol",
                            "smiles": r_smi,
                            **molecule_fields(react, fallback=True),
                        }
                    )

            return node

        # Build route tree and store
        route_tree = build_mol_node(final_step)
        if _route_tree_has_null_node(route_tree):
            logger.warning(
                "Dropping malformed route %s from export: route tree contains a "
                "null node (multiple molecules in one node / malformed route node).",
                route_id,
            )
            continue
        if keep_ids:
            all_routes[int(route_id)] = route_tree
        else:
            all_routes.append(route_tree)

    return all_routes


def build_route_trees(
    routes_dict,
    keep_ids: bool = True,
    molecule_in_stock: MoleculeInStock | None = None,
    route_metadata: dict[int, dict[int, dict[str, Any]]] | None = None,
    *,
    strict: bool = False,
) -> RouteExportResult:
    """Build v1 route trees with explicit diagnostics for skipped routes."""

    route_trees = _make_json_v1(
        routes_dict,
        keep_ids=True,
        molecule_in_stock=molecule_in_stock,
        route_metadata=route_metadata,
    )
    diagnostics = tuple(
        RouteDiagnostic(
            route_id=route_id,
            stage="route_tree_export",
            message="Route could not be represented as a valid v1 route tree",
        )
        for route_id, steps in routes_dict.items()
        if steps and int(route_id) not in route_trees
    )
    if strict and diagnostics:
        raise RouteExportError(diagnostics)
    routes = route_trees if keep_ids else list(route_trees.values())
    return RouteExportResult(routes=routes, diagnostics=diagnostics)


def make_json(
    routes_dict,
    keep_ids: bool = True,
    molecule_in_stock: MoleculeInStock | None = None,
    route_metadata: dict[int, dict[int, dict[str, Any]]] | None = None,
    *,
    strict: bool = False,
):
    """Convert routes into v1 JSON trees.

    ``keep_ids=True`` returns a route-id mapping; ``False`` returns a list.
    Use :func:`build_route_trees` when callers need skipped-route diagnostics.
    """

    return build_route_trees(
        routes_dict,
        keep_ids=keep_ids,
        molecule_in_stock=molecule_in_stock,
        route_metadata=route_metadata,
        strict=strict,
    ).routes


def write_routes_json(
    routes_dict,
    file_path,
    molecule_in_stock: MoleculeInStock | None = None,
    route_metadata: dict[int, dict[int, dict[str, Any]]] | None = None,
    *,
    strict: bool = False,
) -> RouteExportResult:
    """Serialize v1 route trees and return export diagnostics."""

    result = build_route_trees(
        routes_dict,
        molecule_in_stock=molecule_in_stock,
        route_metadata=route_metadata,
        strict=strict,
    )
    with open(file_path, "w") as f:
        json.dump(result.routes, f, indent=2)
    return result


__all__ = [
    "build_route_trees",
    "make_dict",
    "make_json",
    "read_routes_json",
    "write_routes_json",
]
