"""Tree-to-route export adapters kept separate from JSON/CSV codecs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synplan.mcts.tree import Tree


def export_tree_to_json(tree: Tree, file_path: str, route_id=None):
    """Export a retrosynthetic search tree directly to route JSON."""

    from synplan.chem.reaction.routes.io import write_routes_json
    from synplan.chem.reaction.routes.representation import extract_reactions

    routes_dict = extract_reactions(tree, route_id)
    if routes_dict is None:
        raise ValueError("Failed to extract reactions for the specified route_id.")
    return write_routes_json(routes_dict, file_path, tree=tree)


def export_tree_to_csv(tree: Tree, file_path: str = "routes.csv", route_id=None):
    """Export a retrosynthetic search tree directly to route CSV."""

    from synplan.chem.reaction.routes.io import write_routes_csv
    from synplan.chem.reaction.routes.representation import extract_reactions

    routes_dict = extract_reactions(tree, route_id)
    if routes_dict is None:
        raise ValueError("Failed to extract reactions for the specified route_id.")
    return write_routes_csv(routes_dict, file_path)
