import json
import logging
from functools import lru_cache
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, NamedTuple

from chython import smiles as read_smiles

from synplan.chem.building_blocks import BuildingBlockCatalogue
from synplan.chem.precursor import is_purchasable
from synplan.chem.reaction.routes.contracts import (
    RouteDiagnostic,
    RouteExportError,
    RouteExportResult,
)
from synplan.chem.reaction.routes.io.metadata import (
    reaction_metadata,
    restore_reaction_metadata,
)
from synplan.chem.utils import mapped_smiles, molecule_key, normalise

if TYPE_CHECKING:
    from chython.containers import MoleculeContainer, ReactionContainer

    from synplan.chem.reaction.routes.contracts import RouteNode
    from synplan.mcts.tree import Tree

logger = logging.getLogger(__name__)


def route_tree_has_null_node(node) -> bool:
    """Return True if the assembled route tree contains a ``None`` node.

    ``build_mol_node`` returns ``None`` when a route is malformed (e.g. a node
    holding more than one molecule). Such a ``None`` ends up either as the route
    root or nested in a ``children`` list, which would serialize to a JSON
    ``null`` child and corrupt the route.
    """
    return node is None or any(
        route_tree_has_null_node(child) for child in node.get("children", ())
    )


def _purchasable(
    smiles: str,
    molecule,
    stock,
    min_mol_size: int,
    fallback: bool,
):
    """``Precursor.is_building_block`` on an already-serialised molecule.

    Without a stock the caller gets the old positional answer, so an export that never
    had a catalogue to consult keeps meaning what it used to.
    """
    if stock is None:
        return fallback
    return is_purchasable(
        molecule,
        stock,
        min_mol_size,
        key=smiles,
    )


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
            restore_reaction_metadata(reaction, node.get("meta"))
            rxn_list.append(reaction)

    recurse(tree)
    return {i: rxn for i, rxn in enumerate(rxn_list)}


#: Per-step rule metadata keys ``_make_json_v1`` writes onto a reaction node.
_STEP_META_KEYS = ("step_id", "tree_node_id", "rule_id", "rule_source", "rule_key")


@lru_cache(maxsize=4096)
def _file_key(smiles: str) -> str:
    """The canonical SMILES of a molecule that is already in a file.

    Files outlive canonicalisers: a string an older one wrote has to be read and
    written again, or the same molecule ends up under two names. A node whose
    SMILES will not parse keeps its own string -- it then matches nothing, which
    is what an unreadable node deserves.

    Memoised: one file writes the same building block into hundreds of routes,
    and re-reading a string to arrive at the same string is the definition of a
    no-op. String in, string out -- nothing shared across the cache is mutable.
    """
    try:
        return molecule_key(read_smiles(smiles))
    except Exception as error:  # unparseable node: keep the file's own string
        logger.warning("Route molecule %s could not be re-read: %s", smiles, error)
        return smiles


def _canonicalise(reaction) -> int:
    """Rewrite ``reaction``'s molecules the way a file writes them, counting failures.

    The reader normalises with the same function the writer keys by, so a route
    written and read back comes out spelled as it went in. A molecule chython
    cannot prepare survives exactly as the file wrote it -- readable, drawable,
    and counted.
    """
    kept = 0
    for attribute in ("_reactants", "_products"):
        normalised = []
        for mol in getattr(reaction, attribute):
            standard = normalise(mol)
            if standard is None:
                kept += 1
                standard = mol
            normalised.append(standard)
        setattr(reaction, attribute, tuple(normalised))
    return kept


class RouteRead(NamedTuple):
    """What one v1 route tree holds, as objects.

    :param steps: One ``(reaction, product, metadata)`` per reaction node,
        deepest first.
    :param unresolved: The leaves the file does not call purchasable, by
        identity -- each one a reactant of a step in ``steps``.
    :param uncanonical: How many molecules were kept as the file wrote them
        because chython could not canonicalise them.
    """

    steps: tuple[tuple["ReactionContainer", "MoleculeContainer", dict[str, Any]], ...]
    unresolved: tuple["MoleculeContainer", ...]
    uncanonical: int


def read_route_tree(route_json) -> RouteRead:
    """Read one v1 route tree into molecules, steps and stock verdicts.

    A step's product is the very object the consuming reaction holds as a
    reactant, so a route read from a file is the same object graph the search
    hands out and the two behave alike. Unlike :func:`make_dict` this keeps the
    molecule nodes, so the exported ``in_stock`` verdicts survive.

    Every molecule is canonicalised on the way in: the file may come from an
    older SynPlanner or another tool, and one spelling per molecule is what makes
    the rest of this -- and of :class:`~synplan.chem.reaction.routes.route.Route`
    -- a lookup rather than a guess. Canonicalisation keeps atom mapping.
    """
    steps = []
    unresolved = []
    uncanonical = 0

    def link(reaction, child_nodes):
        """Give every reactant its file verdict and, if a step made it, that step.

        Neither side is in a trustworthy order -- a reaction SMILES does not keep
        its container's molecule order, and an older file's children were never
        written in one -- so both are sorted by canonical SMILES and paired off.
        Reactants that share that key are the same molecule, so pairing equals in
        an arbitrary order is a swap of interchangeable subtrees, not a mismatch.
        """
        reactants = list(reaction.reactants)
        slots = sorted(range(len(reactants)), key=lambda index: str(reactants[index]))
        nodes = sorted(
            (node for node in child_nodes if isinstance(node, dict)),
            key=lambda node: _file_key(node["smiles"]),
        )
        for slot, node in zip_longest(slots, nodes):
            if node is None:  # a reactant the file wrote no node for: no verdict
                unresolved.append(reactants[slot])
                continue
            made = molecule(node)
            if slot is None:  # a node no reactant claims: keep its steps, drop it
                continue
            if made is not None:
                reactants[slot] = made
            elif not node.get("in_stock"):
                unresolved.append(reactants[slot])
        # splice, rather than rebuild, to keep the parsed reaction's metadata
        reaction._reactants = tuple(reactants)

    def product_of(reaction, node):
        """The fragment of ``reaction`` this molecule node stands for."""
        if len(reaction.products) == 1:
            return reaction.products[0]
        smiles = _file_key(node["smiles"])
        product = next((p for p in reaction.products if str(p) == smiles), None)
        if product is None:
            logger.warning(
                "Route molecule %s is not a product of %s", node["smiles"], reaction
            )
            return reaction.products[0]
        return product

    def molecule(node):
        """The molecule this node stands for, or None when the file made none."""
        nonlocal uncanonical
        if not isinstance(node, dict):
            return None
        children = node.get("children") or ()
        source = next(
            (
                c
                for c in children
                if isinstance(c, dict) and c.get("type") == "reaction"
            ),
            None,
        )
        if source is None:  # a leaf: its molecule is the reactant the parent holds
            return None

        reaction = read_smiles(source["smiles"])
        restore_reaction_metadata(reaction, source.get("meta"))
        uncanonical += _canonicalise(reaction)
        link(reaction, source.get("children") or ())

        product = product_of(reaction, node)
        steps.append(
            (reaction, product, {k: source[k] for k in _STEP_META_KEYS if k in source})
        )
        return product

    molecule(route_json)
    if uncanonical:
        logger.warning(
            "%d route molecule(s) could not be canonicalised and were read as the "
            "file wrote them",
            uncanonical,
        )
    return RouteRead(tuple(steps), tuple(unresolved), uncanonical)


def route_tree(target, source_of, purchasable, step_fields) -> "RouteNode | None":
    """Unfold a route into one v1 JSON tree, starting from the molecule it makes.

    ``source_of(molecule, consumer)`` answers the only question the callers
    answer differently: which step makes this molecule, and which of that step's
    products it is. ``None`` means nothing makes it -- a starting material. A
    step with no usable product gives ``None`` back in its place, which the drop
    guard in :func:`make_json` reports.

    :param purchasable: ``(molecule, key, is_leaf) -> bool``.
    :param step_fields: ``step_id -> (mapped_smiles, extra node keys)``.
    """

    def mol_node(molecule, consumer):
        source = source_of(molecule, consumer)
        if source is None:
            key = molecule_key(molecule)
            return {
                "type": "mol",
                "smiles": key,
                "in_stock": purchasable(molecule, key, True),
            }
        step_id, reaction, product = source
        if product is None:
            return None
        key = molecule_key(product)
        return {
            "type": "mol",
            "smiles": key,
            "children": [reaction_node(step_id, reaction)],
            "in_stock": purchasable(product, key, False),
        }

    def reaction_node(step_id, reaction):
        smiles, extra = step_fields(step_id)
        return {
            "type": "reaction",
            "smiles": smiles,
            # a reaction SMILES does not keep its container's molecule order, so
            # the reader pairs children with reactants by key rather than by
            # position; writing them in that order is what makes the file stable
            "children": sorted(
                (mol_node(molecule, step_id) for molecule in reaction.reactants),
                key=lambda child: (child or {}).get("smiles", ""),
            ),
            **extra,
        }

    return mol_node(target, None)


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


def read_routes_json(file_path="routes.json", to_dict=False, *, as_routes=False):
    """Read a v1 routes file: raw trees, ``{route: {step: Reaction}}``, or routes.

    :param to_dict: Hand back the legacy ``{route_id: {step_id: Reaction}}``.
    :param as_routes: Hand back :class:`~synplan.chem.reaction.routes.route.Route`
        objects -- the file's own steps and stock verdicts, with no search behind
        them.
    """
    if to_dict and as_routes:
        raise ValueError("read_routes_json returns one shape: to_dict or as_routes")
    with open(file_path) as file:
        routes_json = json.load(file)
    if to_dict:
        return make_dict(routes_json)
    if as_routes:
        # local: Route is built on this module, so it cannot be imported at the top
        from synplan.chem.reaction.routes.route import Route

        trees = routes_json.values() if isinstance(routes_json, dict) else routes_json
        return [Route.from_json(route_tree) for route_tree in trees]
    return routes_json


def _make_json_v1(
    routes_dict,
    keep_ids=True,
    tree: "Tree | None" = None,
    route_metadata: dict[int, dict[int, dict[str, Any]]] | None = None,
    building_blocks: frozenset[str] | set[str] | BuildingBlockCatalogue | None = None,
    min_mol_size: int = 6,
):
    """
    Convert routes into a nested JSON tree of reaction and molecule nodes.

    Args:
        routes_dict (dict[int, dict[int, Reaction]]): Mapping route IDs to steps (step_id -> Reaction).
        keep_ids (bool): If True, returns a dict mapping route IDs to trees; otherwise returns a list.
        tree (Tree | None): Optional source tree used to attach rule metadata to
            reaction nodes.
        route_metadata (dict | None): Optional per-route metadata mapping
            ``route_id -> step_id -> metadata``. This overrides metadata derived
            from ``tree`` when provided.

    Returns:
        list or dict: JSON-like tree(s) of routes.
    """
    if tree is not None and building_blocks is None:
        building_blocks = getattr(tree, "building_blocks", None)
        tree_config = getattr(tree, "config", None)
        if tree_config is not None:
            min_mol_size = tree_config.min_mol_size

    # Prepare output
    all_routes = {} if keep_ids else []

    for route_id, steps in routes_dict.items():
        if not steps:
            continue
        route_step_metadata = (
            route_metadata.get(route_id) if route_metadata is not None else None
        )
        if route_step_metadata is None and tree is not None:
            route_step_metadata = tree.step_metadata(route_id)
        try:
            # Determine target molecule atoms from the final step of this route
            final_step = max(steps)
            target = steps[final_step].products[0]
            atom_nums = set(target._atoms.keys())

            # Precompute canonical SMILES and producer mapping for all products
            prod_map = {}  # smiles -> list of step_ids
            for sid, rxn in steps.items():
                for prod in rxn.products:
                    s = molecule_key(prod)
                    prod_map.setdefault(s, []).append(sid)
        except Exception as e:
            logger.warning("Error processing route %s: %s", route_id, e)
            continue

        def source_of(
            molecule, consumer, _steps=steps, _prod_map=prod_map, _atom_nums=atom_nums
        ):
            """The step that makes ``molecule``, matched on spelling.

            A bare ``{step: reaction}`` carries no link between a step's product
            and the next step's reactant, so the producer is looked up by
            canonical SMILES among the steps that ran earlier, and which product
            fragment it is comes from structural identity with the consuming
            reactant -- falling back to sharing atom numbers with the target,
            which is all there is to go on at the root.
            """
            prior = [
                producer
                for producer in _prod_map.get(molecule_key(molecule), ())
                if consumer is None or producer < consumer
            ]
            if not prior:
                return None
            step_id = max(prior)
            reaction = _steps[step_id]
            product = next((p for p in reaction.products if p == molecule), None)
            if product is None:
                product = next(
                    (p for p in reaction.products if _atom_nums & p._atoms.keys()), None
                )
            return step_id, reaction, product

        def purchasable(
            molecule,
            key,
            leaf,
            _bb=building_blocks,
            _size=min_mol_size,
        ):
            return _purchasable(key, molecule, _bb, _size, leaf)

        def step_fields(step_id, _steps=steps, _meta=route_step_metadata):
            reaction = _steps[step_id]
            extra = {}
            metadata = reaction_metadata(reaction)
            if metadata:
                extra["meta"] = metadata
            if _meta and step_id in _meta:
                extra.update(_meta[step_id])
            return mapped_smiles(reaction), extra

        # Build route tree and store
        tree_node = route_tree(target, source_of, purchasable, step_fields)
        if route_tree_has_null_node(tree_node):
            logger.warning(
                "Dropping malformed route %s from export: route tree contains a "
                "null node (multiple molecules in one node / malformed route node).",
                route_id,
            )
            continue
        if keep_ids:
            all_routes[int(route_id)] = tree_node
        else:
            all_routes.append(tree_node)

    return all_routes


def build_route_trees(
    routes_dict,
    keep_ids: bool = True,
    tree: "Tree | None" = None,
    route_metadata: dict[int, dict[int, dict[str, Any]]] | None = None,
    *,
    strict: bool = False,
    building_blocks: frozenset[str] | set[str] | BuildingBlockCatalogue | None = None,
    min_mol_size: int = 6,
) -> RouteExportResult:
    """Build v1 route trees with explicit diagnostics for skipped routes."""

    route_trees = _make_json_v1(
        routes_dict,
        keep_ids=True,
        tree=tree,
        route_metadata=route_metadata,
        building_blocks=building_blocks,
        min_mol_size=min_mol_size,
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
    tree: "Tree | None" = None,
    route_metadata: dict[int, dict[int, dict[str, Any]]] | None = None,
    *,
    strict: bool = False,
    building_blocks: frozenset[str] | set[str] | BuildingBlockCatalogue | None = None,
    min_mol_size: int = 6,
):
    """Convert routes into v1 JSON trees.

    ``keep_ids=True`` returns a route-id mapping; ``False`` returns a list.
    Use :func:`build_route_trees` when callers need skipped-route diagnostics.
    """

    return build_route_trees(
        routes_dict,
        keep_ids=keep_ids,
        tree=tree,
        route_metadata=route_metadata,
        strict=strict,
        building_blocks=building_blocks,
        min_mol_size=min_mol_size,
    ).routes


def write_routes_json(
    routes,
    file_path,
    tree: "Tree | None" = None,
    route_metadata: dict[int, dict[int, dict[str, Any]]] | None = None,
    *,
    strict: bool = False,
) -> RouteExportResult:
    """Serialize v1 route trees and return export diagnostics.

    :param routes: An iterable of :class:`~synplan.chem.reaction.routes.route.Route`
        -- each writes itself, and the file keys them by position -- or the
        legacy ``{route_id: {step_id: Reaction}}`` mapping, whose keys are tree
        node ids when ``tree`` is given.
    :param tree: The search behind a *mapping* of routes, for per-step rule
        metadata. A ``Route`` already carries its own.
    """

    if isinstance(routes, dict):
        result = build_route_trees(
            routes,
            tree=tree,
            route_metadata=route_metadata,
            strict=strict,
        )
    else:
        if tree is not None or route_metadata is not None:
            raise TypeError(
                "tree= and route_metadata= describe the {route_id: {step_id: "
                "Reaction}} form; a Route already carries its step origins"
            )
        result = RouteExportResult(
            routes={index: route.to_json() for index, route in enumerate(routes)},
            diagnostics=(),
        )
    with open(file_path, "w") as f:
        json.dump(result.routes, f, indent=2)
    return result


__all__ = [
    "RouteRead",
    "build_route_trees",
    "make_dict",
    "make_json",
    "read_route_tree",
    "read_routes_json",
    "write_routes_json",
]
