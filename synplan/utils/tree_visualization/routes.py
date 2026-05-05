"""Route metadata helpers shared by tree visualization renderers."""

from __future__ import annotations

from collections.abc import Iterable

from synplan.mcts.tree import Tree


def route_nodes_by_route(
    tree: Tree, route_ids: Iterable[int] | None = None
) -> dict[str, list[int]]:
    route_ids = tree.winning_nodes if route_ids is None else route_ids
    node_ids_by_object = {id(node): node_id for node_id, node in tree.nodes.items()}

    return {
        str(route_id): [
            node_ids_by_object[id(node)]
            for node in reversed(tree.route_to_node(int(route_id)))
        ]
        for route_id in sorted(int(route_id) for route_id in route_ids)
    }
