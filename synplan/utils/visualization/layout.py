"""Shared tree layout helpers for HTML visualizations."""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

from synplan.mcts.tree import Tree


def group_nodes_by_depth(nodes_depth: dict[int, int]) -> dict[int, list[int]]:
    by_depth: dict[int, list[int]] = {}
    for node_id, depth in nodes_depth.items():
        by_depth.setdefault(depth, []).append(node_id)
    for node_ids in by_depth.values():
        node_ids.sort()
    return by_depth


def build_children_map(
    tree: Tree, allowed_nodes: Optional[set[int]] = None
) -> dict[int, list[int]]:
    if allowed_nodes is None:
        allowed_nodes = set(tree.nodes.keys())
    else:
        allowed_nodes = set(allowed_nodes)
    allowed_nodes.add(1)

    children_map: dict[int, list[int]] = {node_id: [] for node_id in allowed_nodes}
    for child_id, parent_id in tree.parents.items():
        if child_id == 1 or not parent_id:
            continue
        if parent_id not in allowed_nodes or child_id not in allowed_nodes:
            continue
        children_map[parent_id].append(child_id)
    for children in children_map.values():
        children.sort()
    return children_map


def sorted_children(children_map: dict[int, list[int]], node_id: int) -> list[int]:
    return children_map.get(node_id, [])


def compute_depths(
    children_map: dict[int, list[int]], root_id: int = 1
) -> dict[int, int]:
    if root_id not in children_map:
        return {}

    depths = {root_id: 0}
    queue = deque([root_id])
    while queue:
        node_id = queue.popleft()
        for child_id in children_map.get(node_id, []):
            if child_id in depths:
                continue
            depths[child_id] = depths[node_id] + 1
            queue.append(child_id)
    return depths


def compute_subtree_leaf_counts(
    children_map: dict[int, list[int]], root_id: int = 1
) -> dict[int, int]:
    order: list[int] = []
    if root_id not in children_map:
        return {}

    stack = [root_id]
    while stack:
        node_id = stack.pop()
        order.append(node_id)
        stack.extend(sorted_children(children_map, node_id))

    leaf_counts: dict[int, int] = {}
    for node_id in reversed(order):
        children = sorted_children(children_map, node_id)
        leaf_counts[node_id] = (
            1 if not children else sum(leaf_counts[child] for child in children)
        )
    return leaf_counts


def assign_subtree_angles(
    children_map: dict[int, list[int]],
    leaf_counts: dict[int, int],
    root_id: int = 1,
    base_gap: float = 0.04,
) -> dict[int, float]:
    angles: dict[int, float] = {}
    if root_id not in children_map:
        return angles

    stack = [(root_id, 0.0, 2.0 * math.pi)]
    while stack:
        node_id, start_angle, end_angle = stack.pop()
        angles[node_id] = (start_angle + end_angle) / 2.0
        children = sorted_children(children_map, node_id)
        if not children:
            continue

        span = max(0.0, end_angle - start_angle)
        total = sum(leaf_counts.get(child, 1) for child in children)
        if total <= 0 or span <= 0.0:
            continue

        if len(children) > 1:
            max_gap = span * 0.15 / (len(children) - 1)
            gap = min(base_gap, max_gap)
        else:
            gap = 0.0

        span_for_children = span - gap * (len(children) - 1)
        if span_for_children <= 0.0:
            gap = 0.0
            span_for_children = span

        cursor = start_angle
        for child in children:
            frac = leaf_counts.get(child, 1) / total
            child_span = span_for_children * frac
            child_start = cursor
            child_end = cursor + child_span
            stack.append((child, child_start, child_end))
            cursor = child_end + gap
    return angles


def compute_radius_scale(
    by_depth: dict[int, list[int]],
    angles: dict[int, float],
    radius_step: float,
    node_radius: float,
    spacing_factor: float = 2.2,
    root_gap_factor: float = 2.8,
) -> float:
    min_distance = node_radius * spacing_factor
    epsilon = 1e-6
    scale = max(1.0, min_distance / max(radius_step, epsilon))
    root_gap = node_radius * root_gap_factor
    scale = max(scale, root_gap / max(radius_step, epsilon))

    for depth, node_ids in by_depth.items():
        if depth == 0 or len(node_ids) < 2:
            continue
        radius = depth * radius_step
        depth_angles = sorted(angles.get(node_id, 0.0) for node_id in node_ids)
        deltas = [
            right - left for left, right in zip(depth_angles, depth_angles[1:])
        ]
        deltas.append(2.0 * math.pi - depth_angles[-1] + depth_angles[0])
        min_delta = max(min(deltas), epsilon)
        required = min_distance / (radius * min_delta)
        scale = max(scale, required)
    return max(scale, 1.0)


def radial_layout(
    nodes_depth: dict[int, int],
    children_map: dict[int, list[int]],
    radius_step: float,
    node_radius: float,
    spacing_factor: float = 2.2,
    root_gap_factor: float = 2.8,
) -> dict[int, tuple[float, float]]:
    if not nodes_depth:
        return {}

    by_depth = group_nodes_by_depth(nodes_depth)
    leaf_counts = compute_subtree_leaf_counts(children_map)
    angles = assign_subtree_angles(children_map, leaf_counts)
    scale = compute_radius_scale(
        by_depth,
        angles,
        radius_step,
        node_radius,
        spacing_factor=spacing_factor,
        root_gap_factor=root_gap_factor,
    )
    radius_step *= scale

    positions: dict[int, tuple[float, float]] = {}
    for node_id, depth in nodes_depth.items():
        if depth == 0:
            positions[node_id] = (0.0, 0.0)
            continue
        angle = angles.get(node_id, 0.0)
        radius = depth * radius_step
        positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))
    return positions


def scale_positions(
    positions: dict[int, tuple[float, float]], scale: float
) -> dict[int, tuple[float, float]]:
    scale = max(0.01, float(scale))
    return {node_id: (x * scale, y * scale) for node_id, (x, y) in positions.items()}


def edge_key(parent_id: int, child_id: int) -> str:
    return f"{int(parent_id)}->{int(child_id)}"


def bounds_with_pad(
    positions: dict[int, tuple[float, float]],
    base_radius: float,
    bubble_radius: float,
    pad_scale: float = 6.0,
) -> tuple[float, float, float, float]:
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    # Ensure large bubbles are not clipped while keeping padding reasonable.
    pad = max(base_radius * pad_scale, bubble_radius * 1.05, 0.1)
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    return min_x, min_y, max_x - min_x, max_y - min_y
