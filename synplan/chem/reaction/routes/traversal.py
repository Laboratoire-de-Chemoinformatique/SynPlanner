"""Internal helpers for traversing Tree-like retrosynthetic routes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import pairwise
from typing import Any


def route_node_ids(parents: Mapping[int, int], node_id: int) -> tuple[int, ...]:
    """Return route node IDs from the root to ``node_id``."""

    path = []
    current_id = node_id
    while current_id:
        path.append(current_id)
        current_id = parents[current_id]
    path.reverse()
    return tuple(path)


def iter_route_nodes(tree: Any, node_id: int) -> Iterator[Any]:
    """Yield Tree-like route nodes from the root to ``node_id``."""

    for route_node_id in route_node_ids(tree.parents, node_id):
        yield tree.nodes[route_node_id]


def iter_route_steps(tree: Any, node_id: int) -> Iterator[tuple[Any, Any]]:
    """Yield chronological ``(before_node, after_node)`` route steps."""

    yield from pairwise(iter_route_nodes(tree, node_id))


__all__ = ["iter_route_nodes", "iter_route_steps", "route_node_ids"]
