"""Tree-to-route export adapters kept separate from JSON/CSV codecs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from synplan.chem.precursor import Precursor

if TYPE_CHECKING:
    from synplan.mcts.tree import Tree


def _route_step_metadata_from_tree(
    tree: Tree, route_id: int
) -> dict[int, dict[str, object]]:
    """Map extracted route step ids to their source Tree metadata."""
    details = tree.route_details(route_id)
    steps = details.get("steps", [])
    total_steps = len(steps)
    metadata_by_step_id = {}

    for step_index, step in enumerate(steps):
        step_id = total_steps - 1 - step_index
        metadata_by_step_id[step_id] = {
            "step_id": step_id,
            "tree_node_id": step.get("node_id"),
            "rule_id": step.get("rule_id"),
            "rule_source": step.get("rule_source"),
            "rule_key": step.get("rule_key"),
        }

    return metadata_by_step_id


def _molecule_in_stock_from_tree(tree: Tree):
    stock = getattr(tree, "building_blocks", None)
    config = getattr(tree, "config", None)
    if stock is None or config is None:
        return None

    def molecule_in_stock(molecule) -> bool:
        return Precursor(molecule, canonicalize=False).is_building_block(
            stock, config.min_mol_size
        )

    return molecule_in_stock


def build_tree_route_trees(
    tree: Tree,
    *,
    reactions: dict | None = None,
    route_id=None,
    keep_ids: bool = True,
    preserve_transient_bonds: bool = True,
    reconcile_atom_mapping: bool = False,
    strict: bool = False,
):
    """Build v1 route trees from a synthesis Tree.

    ``reactions`` may be supplied when a caller has already extracted the
    route reactions, avoiding a second traversal of the search tree.
    """
    from synplan.chem.reaction.routes.io.json import build_route_trees
    from synplan.chem.reaction.routes.representation import extract_reactions

    if reactions is not None and route_id is not None:
        raise ValueError("route_id cannot be combined with precomputed reactions")
    if reactions is None:
        reactions = extract_reactions(
            tree,
            route_id,
            preserve_transient_bonds=preserve_transient_bonds,
            reconcile_atom_mapping=reconcile_atom_mapping,
        )
    if reactions is None:
        raise ValueError("Failed to extract reactions for the specified route_id.")

    route_metadata = {
        current_route_id: _route_step_metadata_from_tree(tree, current_route_id)
        for current_route_id in reactions
    }
    return build_route_trees(
        reactions,
        keep_ids=keep_ids,
        molecule_in_stock=_molecule_in_stock_from_tree(tree),
        route_metadata=route_metadata,
        strict=strict,
    )


def make_tree_json(
    tree: Tree,
    *,
    reactions: dict | None = None,
    route_id=None,
    keep_ids: bool = True,
    preserve_transient_bonds: bool = True,
    reconcile_atom_mapping: bool = False,
    strict: bool = False,
):
    """Convert a synthesis Tree directly into v1 JSON route trees."""
    return build_tree_route_trees(
        tree,
        reactions=reactions,
        route_id=route_id,
        keep_ids=keep_ids,
        preserve_transient_bonds=preserve_transient_bonds,
        reconcile_atom_mapping=reconcile_atom_mapping,
        strict=strict,
    ).routes


def export_tree_to_json(
    tree: Tree,
    file_path: str,
    route_id=None,
    *,
    reactions: dict | None = None,
    preserve_transient_bonds: bool = True,
    reconcile_atom_mapping: bool = False,
    strict: bool = False,
):
    """Export a retrosynthetic search tree directly to route JSON."""
    result = build_tree_route_trees(
        tree,
        reactions=reactions,
        route_id=route_id,
        preserve_transient_bonds=preserve_transient_bonds,
        reconcile_atom_mapping=reconcile_atom_mapping,
        strict=strict,
    )
    with open(file_path, "w") as stream:
        json.dump(result.routes, stream, indent=2)
    return result


def export_tree_to_csv(tree: Tree, file_path: str = "routes.csv", route_id=None):
    """Export a retrosynthetic search tree directly to route CSV."""

    from synplan.chem.reaction.routes.io import write_routes_csv
    from synplan.chem.reaction.routes.representation import extract_reactions

    routes_dict = extract_reactions(tree, route_id)
    if routes_dict is None:
        raise ValueError("Failed to extract reactions for the specified route_id.")
    return write_routes_csv(routes_dict, file_path)


__all__ = [
    "build_tree_route_trees",
    "export_tree_to_csv",
    "export_tree_to_json",
    "make_tree_json",
]
