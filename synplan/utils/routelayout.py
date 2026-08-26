"""Tidy-tree layout for a retrosynthetic route. Stdlib only.

A route is a tree — every molecule is made by exactly one reaction — so layer
assignment is depth and Reingold-Tilford gives a crossing-free order. Columns are
right-aligned, which keeps every connector inside the gap between two columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Node", "edges", "layout", "walk"]


@dataclass
class Node:
    """One box. ``x``/``y`` are the top-left corner, filled in by :func:`layout`."""

    key: str
    w: float
    h: float
    children: list[Node] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    depth: int = 0


def layout(
    root: Node, col_gap: float = 74.0, row_gap: float = 18.0
) -> tuple[float, float, dict[int, float], dict[int, float]]:
    """Position every node. The target sits on the right, the leaves on the left.

    :param root: The target's node; its ``children`` are the precursors it came from.
    :param col_gap: Empty width between two columns; the connectors live in it.
    :param row_gap: Empty height between two stacked subtrees.
    :return: ``(width, height, x per column, width per column)``.
    """
    if not col_gap > 0.0 or not row_gap >= 0.0:
        raise ValueError(
            f"col_gap must be > 0 and row_gap >= 0, got {col_gap}, {row_gap}"
        )

    # This walk is the only place that sees every node once, so the input is checked
    # here: a repeat means a cycle or a shared node, a bad size means a box that
    # would land outside its own column.
    order, col_w, seen = [], {}, set()
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if id(node) in seen:
            raise ValueError(
                f"{node.key!r} is reachable twice — input is not a tree "
                "(cycle, self-ancestor, or one Node object used in two places)"
            )
        seen.add(id(node))
        if not (0.0 <= node.w < 1e9 and 0.0 <= node.h < 1e9):  # False for NaN and inf
            raise ValueError(
                f"{node.key!r} has size {node.w}x{node.h}: need finite, non-negative"
            )
        node.depth = depth
        order.append(node)
        col_w[depth] = max(col_w.get(depth, 0.0), node.w)
        stack.extend((child, depth + 1) for child in node.children)

    col_x, x = {}, 0.0
    for depth in range(max(col_w), -1, -1):  # deepest column left, target flush right
        col_x[depth] = x
        x += col_w[depth] + col_gap
    total_w = x - col_gap

    total_h = _place(root, 0.0, row_gap)
    for node in order:
        node.x = col_x[node.depth] + (col_w[node.depth] - node.w)
    return total_w, total_h, col_x, col_w


def _place(node: Node, top: float, row_gap: float) -> float:
    """Stack the subtree below ``top`` and centre each parent on its children."""
    if not node.children:
        node.y = top
        return top + node.h

    cursor = top
    for child in node.children:
        cursor = _place(child, cursor, row_gap) + row_gap
    cursor -= row_gap

    first, last = node.children[0], node.children[-1]
    node.y = (first.y + first.h / 2 + last.y + last.h / 2) / 2 - node.h / 2
    if node.y < top:  # a parent taller than its children must not enter the row above
        shift = top - node.y
        for child in walk(node):
            child.y += shift
        node.y = top
        cursor += shift
    return max(cursor, node.y + node.h)


def walk(node: Node) -> list[Node]:
    """Every descendant of ``node``, excluding ``node`` itself."""
    out, stack = [], list(node.children)
    while stack:
        child = stack.pop()
        out.append(child)
        stack.extend(child.children)
    return out


def edges(
    root: Node,
    col_x: dict[int, float],
    col_w: dict[int, float],
    lane_frac: float = 0.42,
) -> list[dict[str, Any]]:
    """One entry per parent: the lane x plus the child and parent anchor points.

    ``lane_frac`` must stay strictly inside (0, 1): at either end the lane collapses
    onto a column edge and the no-connector-over-a-box guarantee is gone.
    """
    if not 0.0 < lane_frac < 1.0:
        raise ValueError(f"lane_frac must be strictly between 0 and 1, got {lane_frac}")

    out = []
    for node in [root, *walk(root)]:
        if not node.children:
            continue
        right = col_x[node.depth + 1] + col_w[node.depth + 1]
        out.append(
            {
                "node": node,
                "lane": right + (node.x - right) * lane_frac,
                "parent": (node.x, node.y + node.h / 2),
                "children": [(c.x + c.w, c.y + c.h / 2) for c in node.children],
            }
        )
    return out
