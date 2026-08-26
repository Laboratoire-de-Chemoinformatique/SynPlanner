"""Internal helpers for traversing Tree-like retrosynthetic routes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from itertools import pairwise
from typing import Any


def linearise(steps: Sequence[Any]) -> tuple[int, ...]:
    """Order the steps so that each branch of a convergent route is contiguous.

    The search enumerates precursors breadth-first, which interleaves the branches;
    a depth-first post-order walk from the target puts every linear stretch back
    together. The result is still topological: a step comes after the steps feeding
    it.

    :param steps: The route's reactions, in any topological order.
    :return: Indices into ``steps``, deepest step first and the target's
        disconnection last.
    """

    by_product = {str(step.products[0]): index for index, step in enumerate(steps)}
    order: list[int] = []
    seen: set[int] = set()

    def visit(index: int) -> None:
        seen.add(index)
        for mol in steps[index].reactants:
            feeder = by_product.get(str(mol))
            if feeder is not None and feeder not in seen:
                visit(feeder)
        order.append(index)

    if steps:
        visit(len(steps) - 1)
    order.extend(index for index in range(len(steps)) if index not in seen)
    return tuple(order)


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


__all__ = ["iter_route_nodes", "iter_route_steps", "linearise", "route_node_ids"]
