"""Invariants the tidy-tree layout must hold for any route, however shaped."""

from __future__ import annotations

import random
import sys

import pytest

from synplan.utils.routelayout import Node, edges, layout, walk


def boxes(root: Node) -> list[tuple[float, float, float, float, Node]]:
    return [(n.x, n.y, n.x + n.w, n.y + n.h, n) for n in [root, *walk(root)]]


def overlaps(root: Node) -> list[tuple[str, str]]:
    placed, bad = boxes(root), []
    for i, a in enumerate(placed):
        for c in placed[i + 1 :]:
            if (
                a[0] < c[2] - 1e-9
                and c[0] < a[2] - 1e-9
                and a[1] < c[3] - 1e-9
                and c[1] < a[3] - 1e-9
            ):
                bad.append((a[4].key, c[4].key))
    return bad


def edge_hits(root: Node, col_x: dict, col_w: dict) -> list[tuple[str, str]]:
    """Every connector segment that crosses the interior of some box."""
    placed, bad = boxes(root), []
    for edge in edges(root, col_x, col_w):
        lane, (px, py) = edge["lane"], edge["parent"]
        segments = [((lane, py), (px, py))]
        for cx, cy in edge["children"]:
            segments += [((cx, cy), (lane, cy)), ((lane, cy), (lane, py))]
        for (x1, y1), (x2, y2) in segments:
            lo_x, hi_x = min(x1, x2), max(x1, x2)
            lo_y, hi_y = min(y1, y2), max(y1, y2)
            for bx1, by1, bx2, by2, node in placed:
                if (
                    hi_x - 1e-6 <= bx1
                    or lo_x + 1e-6 >= bx2
                    or hi_y - 1e-6 <= by1
                    or lo_y + 1e-6 >= by2
                ):
                    continue
                bad.append((edge["node"].key, node.key))
    return bad


def finite(root: Node) -> bool:
    return all(v == v and abs(v) < 1e9 for n in [root, *walk(root)] for v in (n.x, n.y))


def lanes_inside_gap(root: Node, col_x: dict, col_w: dict) -> bool:
    """Every lane sits strictly inside the empty gap between two columns.

    A degenerate picture with everything collapsed onto one point passes every
    no-overlap test vacuously; it fails this one.
    """
    for edge in edges(root, col_x, col_w):
        node = edge["node"]
        right = col_x[node.depth + 1] + col_w[node.depth + 1]
        if not right < edge["lane"] < node.x:
            return False
    return True


def bbox_inside(root: Node, width: float, height: float) -> bool:
    placed = boxes(root)
    return (
        min(x1 for x1, *_ in placed) >= -1e-6
        and min(y1 for _, y1, *_ in placed) >= -1e-6
        and max(x2 for _, _, x2, *_ in placed) <= width + 1e-6
        and max(y2 for *_, y2, _ in placed) <= height + 1e-6
    )


def assert_well_formed(root: Node, **kwargs) -> None:
    width, height, col_x, col_w = layout(root, **kwargs)
    assert overlaps(root) == []
    assert edge_hits(root, col_x, col_w) == []
    assert finite(root)
    assert bbox_inside(root, width, height)
    assert lanes_inside_gap(root, col_x, col_w)


def leaf(key: str, w: float = 100, h: float = 60) -> Node:
    return Node(key, w, h)


def chain(n: int, w: float = 90, h: float = 50) -> Node:
    root = current = Node("c0", w, h)
    for i in range(1, n):
        current.children = [Node(f"c{i}", w, h)]
        current = current.children[0]
    return root


WELL_FORMED = {
    # A lone target is a valid route; the canvas is exactly its box.
    "single node": Node("T", 200, 120),
    "chain depth 12": chain(12),
    "8 siblings": Node(
        "T", 150, 90, [leaf(f"s{i}", 80 + i * 20, 40 + i * 15) for i in range(8)]
    ),
    "30-sibling fan, sizes 5..600": Node(
        "T", 150, 90, [leaf(f"f{i}", 5 + i * 20, 5 + (i % 7) * 60) for i in range(30)]
    ),
    # A parent taller than its whole child stack is centred, and the recentring
    # never pushes it above the row it was given.
    "parent far taller than children": Node(
        "T", 200, 700, [leaf("a", 90, 40), leaf("b", 90, 40)]
    ),
    "parent taller than children combined": Node(
        "T", 120, 1000, [leaf("a", 90, 40), leaf("b", 90, 40), leaf("c", 90, 40)]
    ),
    "tiny leaf beside huge leaf": Node(
        "T", 150, 90, [leaf("tiny", 20, 14), leaf("huge", 700, 520)]
    ),
    # A zero-size box is degenerate but legal: it must not collapse its column.
    "zero-size node": Node("T", 150, 90, [leaf("z", 0, 0), leaf("b", 90, 60)]),
    "zero width, real height": Node(
        "T", 150, 90, [leaf("z", 0, 60), leaf("b", 90, 60)]
    ),
    "zero height, real width": Node(
        "T", 150, 90, [leaf("z", 100, 0), leaf("b", 90, 60)]
    ),
}


@pytest.mark.parametrize("name", sorted(WELL_FORMED))
def test_layout_is_well_formed(name):
    assert_well_formed(WELL_FORMED[name])


def _cycle() -> Node:
    node = Node("A", 100, 60)
    node.children = [Node("B", 100, 60, [node])]
    return node


def _self_child() -> Node:
    node = Node("S", 100, 60)
    node.children = [node]
    return node


def _self_ancestor() -> Node:
    node = Node("X", 100, 60)
    node.children = [Node("Y", 100, 60, [Node("Z", 100, 60, [node])])]
    return node


def _shared_sibling() -> Node:
    shared = leaf("shared", 100, 60)
    return Node("T", 150, 90, [shared, shared])


def _shared_across_branches() -> Node:
    shared = leaf("shared", 100, 60)
    return Node(
        "T", 150, 90, [Node("L", 100, 60, [shared]), Node("R", 100, 60, [shared])]
    )


# Sizes come from an outside renderer and are not trusted: anything that is not a
# finite, non-negative number must raise rather than place a box at NaN (invisible)
# or with its right edge inside the connector corridor. Anything reachable twice
# must raise too — placing it twice draws one box over another, and before the
# guard a cyclic input ate RAM until the OOM killer arrived.
REJECTED = {
    "NaN width": (Node("T", 150, 90, [leaf("nan", float("nan"), 60)]), {}),
    "negative width": (Node("T", 150, 90, [leaf("neg", -40, 60)]), {}),
    "infinite height": (Node("T", 150, 90, [leaf("inf", 90, float("inf"))]), {}),
    "same object twice as sibling": (_shared_sibling(), {}),
    "same object in two branches": (_shared_across_branches(), {}),
    "cycle A->B->A": (_cycle(), {}),
    "node is its own child": (_self_child(), {}),
    "node is its own ancestor": (_self_ancestor(), {}),
    # The whole no-connector-over-a-box property rests on a strictly positive column
    # gap; a degenerate one must raise, not report a clean canvas of collapsed edges.
    "col_gap = 0": (Node("T", 150, 90, [leaf("a", 90, 60)]), {"col_gap": 0.0}),
    "negative row_gap": (Node("T", 150, 90, [leaf("a", 90, 60)]), {"row_gap": -5.0}),
}


@pytest.mark.parametrize("name", sorted(REJECTED))
def test_layout_rejects_impossible_input(name):
    root, kwargs = REJECTED[name]
    with pytest.raises(ValueError):
        layout(root, **kwargs)


@pytest.mark.parametrize("lane_frac", [0.0, 1.0, -0.2, 1.5])
def test_edges_reject_lane_on_a_column_edge(lane_frac):
    root = Node("T", 150, 90, [leaf("a", 90, 60)])
    _, _, col_x, col_w = layout(root)
    with pytest.raises(ValueError):
        edges(root, col_x, col_w, lane_frac=lane_frac)


def test_deep_chain_does_not_blow_the_stack():
    """A route is shallow, but a stack overflow here would be a crash, not a picture."""
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    try:
        assert_well_formed(chain(1200, 60, 40))
    finally:
        sys.setrecursionlimit(limit)


def test_random_trees_are_well_formed():
    rng = random.Random(7)
    for _ in range(200):
        nodes = [Node("r", rng.uniform(5, 600), rng.uniform(5, 400))]
        for i in range(rng.randint(0, 39)):
            parent = rng.choice(nodes)
            child = Node(f"n{i}", rng.uniform(5, 600), rng.uniform(5, 400))
            parent.children.append(child)
            nodes.append(child)
        assert_well_formed(nodes[0])
