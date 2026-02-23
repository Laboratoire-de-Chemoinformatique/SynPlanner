#!/usr/bin/env python3
"""
Expansion timeline visualization for a SynPlanner MCTS tree.

This script reuses the layout helpers from tutorials/visualize_tree.py and
approximates expansion order by node creation order (node_id). In the current
Tree implementation, node IDs are assigned sequentially in Tree._add_node.

python expansion_tree.py --tree tree.pkl --output expansion_evol.html
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import visualize_tree as base_vis
from synplan.chem.precursor import Precursor
from synplan.mcts.tree import Tree


def _auto_node_radius(num_nodes: int) -> float:
    if num_nodes <= 0:
        return 0.06
    # Shrink nodes as the tree grows, but keep a reasonable minimum size.
    return max(0.01, min(0.07, 1.8 / math.sqrt(num_nodes)))


def _node_status(tree: Tree, node_id: int) -> str:
    if node_id == 1:
        return "target"
    node = tree.nodes.get(node_id)
    if node is None:
        return "unknown"
    try:
        try:
            if getattr(tree, "children", {}).get(node_id):
                return "intermediate"
        except Exception:
            pass
        precursor = getattr(node, "curr_precursor", None)
        if precursor is None:
            precursor = None
        elif isinstance(precursor, (tuple, list, dict)) and len(precursor) == 0:
            precursor = None
        if precursor is not None and precursor.is_building_block(
            tree.building_blocks, tree.config.min_mol_size
        ):
            return "leaf"

        precursors_to_expand = getattr(node, "precursors_to_expand", None)
        if precursors_to_expand:
            try:
                candidate = precursors_to_expand[0]
            except Exception:
                candidate = None
            if candidate is not None and candidate.is_building_block(
                tree.building_blocks, tree.config.min_mol_size
            ):
                return "leaf"
        else:
            new_precursors = getattr(node, "new_precursors", None) or ()
            if new_precursors and all(
                p.is_building_block(tree.building_blocks, tree.config.min_mol_size)
                for p in new_precursors
            ):
                return "leaf"
    except Exception:
        pass
    return "intermediate"


def _node_primary_smiles_and_svg(node, with_svg: bool) -> Tuple[Optional[str], Optional[str]]:
    molecule = None
    # Prefer the first newly created precursor, matching the requested behavior.
    if getattr(node, "new_precursors", None):
        try:
            precursor = node.new_precursors[0]
            molecule = getattr(precursor, "molecule", None)
        except Exception:
            molecule = None
    if molecule is None:
        molecule = base_vis._node_primary_molecule(node)
    if molecule is None:
        return None, None
    try:
        smiles = str(molecule)
    except Exception:
        smiles = None
    if not with_svg:
        return smiles, None
    try:
        svg = base_vis._depict_molecule_svg(molecule)
    except Exception:
        svg = None
    return smiles, svg


def _node_product_molecules(node) -> List[object]:
    """Return product molecules for a node, falling back to a single primary molecule."""
    molecules: List[object] = []
    new_precursors = getattr(node, "new_precursors", None) or ()
    for precursor in new_precursors:
        mol = getattr(precursor, "molecule", None)
        if mol is not None:
            molecules.append(mol)
    if molecules:
        return molecules
    primary = base_vis._node_primary_molecule(node)
    return [primary] if primary is not None else []


def _molecule_smiles_and_svg(
    molecule: Optional[object], *, with_svg: bool
) -> Tuple[Optional[str], Optional[str]]:
    if molecule is None:
        return None, None
    try:
        smiles = str(molecule)
    except Exception:
        smiles = None
    if not with_svg:
        return smiles, None
    try:
        svg = base_vis._depict_molecule_svg(molecule)
    except Exception:
        svg = None
    return smiles, svg


def _molecule_key(molecule: Optional[object]) -> Optional[str]:
    if molecule is None:
        return None
    try:
        return str(molecule)
    except Exception:
        return None


def _curr_precursor_key(node) -> Optional[str]:
    curr = getattr(node, "curr_precursor", None)
    if curr is None:
        return None
    molecule = getattr(curr, "molecule", curr)
    key = _molecule_key(molecule)
    if key:
        return key
    try:
        return str(curr)
    except Exception:
        return None


def _spread_product_positions(
    base_x: float,
    base_y: float,
    count: int,
    bubble_radius: float,
    *,
    parent_pos: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    """Spread multiple products around the base position to avoid overlap."""
    if count <= 1 or bubble_radius <= 0:
        return [(base_x, base_y)]

    if parent_pos is not None:
        dx = base_x - float(parent_pos[0])
        dy = base_y - float(parent_pos[1])
    else:
        dx, dy = base_x, base_y

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        base_angle = 0.0
    else:
        base_angle = math.atan2(dy, dx)

    # Keep same-tree bubbles overlapping by half of their size.
    ring_radius = bubble_radius

    # Keep the primary (anchor) product at the base position to preserve layout.
    positions: List[Tuple[float, float]] = [(base_x, base_y)]
    remaining = count - 1
    start_angle = base_angle - math.pi / 2.0
    step = (2.0 * math.pi) / float(max(1, remaining))

    for idx in range(remaining):
        angle = start_angle + step * idx
        positions.append(
            (
                base_x + ring_radius * math.cos(angle),
                base_y + ring_radius * math.sin(angle),
            )
        )
    return positions


def _select_node_ids(tree: Tree, max_nodes: Optional[int]) -> List[int]:
    node_ids = sorted(int(nid) for nid in tree.nodes.keys())
    if max_nodes and max_nodes > 0:
        node_ids = node_ids[: max_nodes]
    if 1 in tree.nodes and 1 not in node_ids:
        node_ids = [1, *node_ids]
    return node_ids


def _creation_steps(node_ids: Iterable[int], sample_step: int) -> Dict[int, int]:
    sample_step = 1
    ordered = sorted(set(int(nid) for nid in node_ids))
    steps: Dict[int, int] = {}
    for idx, node_id in enumerate(ordered):
        steps[node_id] = idx // sample_step
    return steps


def _has_expandable_nodes(tree: Tree, node_ids: Set[int]) -> bool:
    for node_id in node_ids:
        node = tree.nodes.get(node_id)
        if node is None:
            continue
        precursors = getattr(node, "precursors_to_expand", None)
        if not precursors:
            continue
        try:
            if len(precursors) > 0:
                return True
        except Exception:
            return True
    return False


def _min_angular_separation(radius: float, bubble_radius: float) -> float:
    """Approximate the angular gap needed to avoid overlap on a given radius."""
    epsilon = 1e-6
    if radius <= epsilon or bubble_radius <= 0:
        return 0.0
    # chord length ~= 2 * radius * sin(theta / 2) >= 2 * bubble_radius
    ratio = max(0.0, min(0.999, bubble_radius / radius))
    try:
        theta = 2.0 * math.asin(ratio)
    except ValueError:
        theta = math.pi
    return max(0.0, min(theta * 1.08, 2.0 * math.pi))


def _separate_angles_by_depth(
    nodes_depth: Dict[int, int],
    children_map: Dict[int, List[int]],
    *,
    radius_step: float,
    node_radius: float,
    bubble_scale: float,
    separation_boost: float,
) -> Dict[int, float]:
    """Spread angles locally to reduce collisions without global rescaling."""
    if not nodes_depth:
        return {}
    by_depth = base_vis._group_nodes_by_depth(nodes_depth)
    leaf_counts = base_vis._compute_subtree_leaf_counts(children_map)
    angles = base_vis._assign_subtree_angles(children_map, leaf_counts)

    bubble_radius = node_radius * max(0.1, float(bubble_scale))
    boost = max(1.0, float(separation_boost))
    full_turn = 2.0 * math.pi

    for depth, node_ids in by_depth.items():
        if depth <= 0 or len(node_ids) < 2:
            continue
        radius = max(1e-6, depth * radius_step)
        min_delta = _min_angular_separation(radius, bubble_radius) * boost
        max_delta = (full_turn - 1e-6) / len(node_ids)
        min_delta = min(min_delta, max_delta)
        if min_delta <= 0:
            continue

        sorted_nodes = sorted(node_ids, key=lambda nid: angles.get(nid, 0.0))
        original_angles = [angles.get(nid, 0.0) for nid in sorted_nodes]
        adjusted = [original_angles[0]]
        for idx in range(1, len(sorted_nodes)):
            desired = original_angles[idx]
            adjusted.append(max(desired, adjusted[-1] + min_delta))

        span = adjusted[-1] - adjusted[0]
        if span > full_turn:
            # Fall back to evenly spaced angles around the circle.
            adjusted = [original_angles[0] + i * max_delta for i in range(len(sorted_nodes))]

        # Re-center to preserve the overall orientation of this depth layer.
        orig_center = (original_angles[0] + original_angles[-1]) / 2.0
        new_center = (adjusted[0] + adjusted[-1]) / 2.0
        shift = new_center - orig_center
        for nid, ang in zip(sorted_nodes, adjusted):
            angles[nid] = ang - shift

    return angles


def _radial_layout_with_separation(
    nodes_depth: Dict[int, int],
    children_map: Dict[int, List[int]],
    *,
    radius_step: float,
    node_radius: float,
    bubble_scale: float,
    separation_boost: float,
) -> Dict[int, Tuple[float, float]]:
    angles = _separate_angles_by_depth(
        nodes_depth,
        children_map,
        radius_step=radius_step,
        node_radius=node_radius,
        bubble_scale=bubble_scale,
        separation_boost=separation_boost,
    )
    if not angles:
        return {}

    positions: Dict[int, Tuple[float, float]] = {}
    for node_id, depth in nodes_depth.items():
        if depth == 0:
            positions[node_id] = (0.0, 0.0)
            continue
        angle = angles.get(node_id, 0.0)
        radius = depth * radius_step
        positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))
    return positions


def _edge_key(parent_id: int, child_id: int) -> str:
    return f"{int(parent_id)}->{int(child_id)}"


def _winning_route_paths(
    tree: Tree,
    *,
    allowed_tree_nodes: Set[int],
    tree_steps: Dict[int, int],
    anchor_by_tree: Dict[int, int],
    render_edge_keys_by_tree_edge: Dict[str, List[str]],
) -> Tuple[Set[str], List[Dict[str, object]], Optional[int], Optional[int]]:
    winning_nodes = list(getattr(tree, "winning_nodes", []) or [])
    winning_edge_keys: Set[str] = set()
    paths: List[Dict[str, object]] = []
    win_steps: List[int] = []

    parents = getattr(tree, "parents", {}) or {}
    for raw_node_id in winning_nodes:
        try:
            node_id = int(raw_node_id)
        except Exception:
            continue
        if node_id not in allowed_tree_nodes:
            continue

        node_step = int(tree_steps.get(node_id, 0))
        win_steps.append(node_step)

        edge_keys: List[str] = []
        current = node_id
        seen: Set[int] = set()
        while current and current not in seen:
            seen.add(current)
            parent_id = int(parents.get(current, 0) or 0)
            if not parent_id:
                break
            if parent_id not in allowed_tree_nodes or current not in allowed_tree_nodes:
                break
            tree_edge_key = _edge_key(parent_id, current)
            render_keys = render_edge_keys_by_tree_edge.get(tree_edge_key)
            if not render_keys:
                parent_anchor = int(anchor_by_tree.get(parent_id, parent_id))
                child_anchor = int(anchor_by_tree.get(current, current))
                render_keys = [_edge_key(parent_anchor, child_anchor)]
            edge_keys.extend(render_keys)
            for key in render_keys:
                winning_edge_keys.add(key)
            current = parent_id

        render_node_id = int(anchor_by_tree.get(node_id, node_id))
        paths.append(
            {
                "nodeId": render_node_id,
                "treeNodeId": node_id,
                "nodeStep": node_step,
                "edgeKeys": edge_keys,
            }
        )

    first_step = min(win_steps) if win_steps else None
    last_step = max(win_steps) if win_steps else None
    return winning_edge_keys, paths, first_step, last_step


def _scale_positions(
    positions: Dict[int, Tuple[float, float]], scale: float
) -> Dict[int, Tuple[float, float]]:
    scale = max(0.01, float(scale))
    return {node_id: (x * scale, y * scale) for node_id, (x, y) in positions.items()}


def _reorder_children_for_layout(
    children_map: Dict[int, List[int]],
    leaf_counts: Dict[int, int],
) -> Dict[int, List[int]]:
    """Ensure the first non-extended child sits next to the extended child."""
    reordered: Dict[int, List[int]] = {}
    for parent_id, children in children_map.items():
        if len(children) < 2:
            reordered[parent_id] = list(children)
            continue
        expanded = [c for c in children if children_map.get(c)]
        if not expanded:
            reordered[parent_id] = list(children)
            continue
        unexpanded = [c for c in children if c not in expanded]
        if not unexpanded:
            reordered[parent_id] = list(children)
            continue
        pivot = max(expanded, key=lambda cid: leaf_counts.get(cid, 1))
        first_unexpanded = unexpanded[0]
        new_order: List[int] = []
        for candidate in (pivot, first_unexpanded):
            if candidate in children and candidate not in new_order:
                new_order.append(candidate)
        for child_id in children:
            if child_id not in new_order:
                new_order.append(child_id)
        reordered[parent_id] = new_order
    return reordered


def _petal_layout_from_depth1(
    children_map: Dict[int, List[int]],
    nodes_depth: Dict[int, int],
    parents: Dict[int, int],
    *,
    step: float,
    bubble_radius: float,
    root_id: int = 1,
    arc_span: float = math.pi,
    depth1_angles: Optional[Dict[int, float]] = None,
) -> Dict[int, Tuple[float, float]]:
    """Root uses a full circle; depth>=2 uses parent-centric 180° petals."""
    if not children_map or root_id not in children_map:
        return {}
    if step <= 0:
        return {}

    positions: Dict[int, Tuple[float, float]] = {root_id: (0.0, 0.0)}
    headings: Dict[int, float] = {}
    bubble_size = max(0.0, float(bubble_radius) * 2.0)
    radius_growth = float(step)  # grow radius_step by +step/2 per depth
    min_dist_sq = (bubble_size) ** 2 if bubble_size > 0 else 0.0

    # Depth 1: regular circle around the target.
    depth1_children = [c for c in children_map.get(root_id, []) if nodes_depth.get(c) == 1]
    if depth1_children:
        if depth1_angles:
            ordered = sorted(depth1_children, key=lambda cid: depth1_angles.get(cid, 0.0))
        else:
            ordered = list(depth1_children)
        count = len(ordered)
        step_angle = (2.0 * math.pi) / max(1, count)
        start = -math.pi / 2.0
        for idx, child_id in enumerate(ordered):
            angle = depth1_angles.get(child_id, start + step_angle * idx) if depth1_angles else (start + step_angle * idx)
            depth = int(nodes_depth.get(child_id, 1))
            radius = float(step) + radius_growth * max(0, depth - 1)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[child_id] = (x, y)
            headings[child_id] = angle

    def _span_for_count(count: int) -> float:
        # min_span = 2.0 * math.pi / 3.0  # 120 degrees
        min_span = math.pi / 2.0  # 90 degrees
        max_span = float(arc_span) if arc_span > 0 else math.pi
        if count <= 25:
            return min_span
        if count >= 50:
            return max_span
        ratio = (count - 25) / 25.0
        return min_span + (max_span - min_span) * ratio

    queue: List[int] = [c for c in depth1_children if c in positions]
    while queue:
        parent_id = queue.pop(0)
        children = children_map.get(parent_id, [])
        if not children:
            continue
        parent_pos = positions.get(parent_id)
        if parent_pos is None:
            continue
        px, py = parent_pos
        grandparent_id = int(parents.get(parent_id, 0) or 0)
        if grandparent_id in positions:
            gx, gy = positions[grandparent_id]
            center_angle = math.atan2(py - gy, px - gx)
        else:
            center_angle = headings.get(parent_id, 0.0)

        span = _span_for_count(len(children))
        count = len(children)
        step_angle = span / max(1, count)
        start = center_angle - span / 2.0 + step_angle / 2.0

        sibling_positions: List[Tuple[float, float]] = []
        for idx, child_id in enumerate(children):
            depth = int(nodes_depth.get(child_id, nodes_depth.get(parent_id, 0) + 1))
            angle = start + step_angle * idx
            radius = float(step) + radius_growth * max(0, depth - 1)
            attempts = 0
            while bubble_size > 0 and attempts < 6:
                x = px + radius * math.cos(angle)
                y = py + radius * math.sin(angle)
                overlap = False
                for sx, sy in sibling_positions:
                    dx = x - sx
                    dy = y - sy
                    if (dx * dx + dy * dy) < min_dist_sq:
                        overlap = True
                        break
                if not overlap:
                    break
                radius += bubble_size
                attempts += 1
            x = px + radius * math.cos(angle)
            y = py + radius * math.sin(angle)
            positions[child_id] = (x, y)
            headings[child_id] = angle
            sibling_positions.append((x, y))
            queue.append(child_id)

    return positions


def _collect_subtree_nodes(children_map: Dict[int, List[int]], root_id: int) -> Set[int]:
    if root_id not in children_map:
        return set()
    collected: Set[int] = set()
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        if node_id in collected:
            continue
        collected.add(node_id)
        stack.extend(children_map.get(node_id, []))
    return collected


def _apply_depth2_parent_arc_gap(
    positions: Dict[int, Tuple[float, float]],
    children_map: Dict[int, List[int]],
    nodes_depth: Dict[int, int],
    *,
    bubble_radius: float,
    gap_diameters: float = 2.0,
    root_id: int = 1,
) -> Dict[int, Tuple[float, float]]:
    """Insert an angular gap between depth-2 groups from different depth-1 parents."""
    if not positions or not children_map or not nodes_depth:
        return positions
    if bubble_radius <= 0:
        return positions

    depth1_nodes = [n for n in children_map.get(root_id, []) if nodes_depth.get(n) == 1]
    if len(depth1_nodes) < 2:
        return positions

    full_turn = 2.0 * math.pi

    def _norm(angle: float) -> float:
        return angle % full_turn

    def _group_bounds(angles: List[float]) -> Tuple[float, float, float]:
        if not angles:
            return 0.0, 0.0, 0.0
        if len(angles) == 1:
            ang = _norm(angles[0])
            return ang, ang, 0.0
        angles = sorted(_norm(a) for a in angles)
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append(angles[0] + full_turn - angles[-1])
        max_gap = max(gaps)
        idx = gaps.index(max_gap)
        start = angles[(idx + 1) % len(angles)]
        end = angles[idx]
        span = (end - start) % full_turn
        return start, end, span

    groups: List[Dict[str, object]] = []
    for parent_id in depth1_nodes:
        depth2_children = [
            c
            for c in children_map.get(parent_id, [])
            if nodes_depth.get(c) == 2 and c in positions
        ]
        if not depth2_children:
            continue
        angles = [math.atan2(positions[c][1], positions[c][0]) for c in depth2_children]
        start, end, span = _group_bounds(angles)
        radius = sum(math.hypot(positions[c][0], positions[c][1]) for c in depth2_children) / len(
            depth2_children
        )
        groups.append(
            {
                "parent": parent_id,
                "depth2": depth2_children,
                "start": start,
                "end": end,
                "span": span,
                "radius": radius,
                "order": min(depth2_children),
            }
        )

    if len(groups) < 2:
        return positions

    groups.sort(key=lambda g: int(g["order"]))
    bubble_size = bubble_radius * 2.0

    updated = dict(positions)
    cumulative_shift = 0.0
    prev_end_unwrapped: Optional[float] = None

    for group in groups:
        start = float(group["start"])
        span = float(group["span"])
        radius = max(1e-6, float(group["radius"]))
        gap_angle = (gap_diameters * bubble_size) / radius

        base_start = start + cumulative_shift
        if prev_end_unwrapped is not None:
            while base_start < prev_end_unwrapped:
                base_start += full_turn
            min_start = prev_end_unwrapped + gap_angle
            if base_start < min_start:
                shift_needed = min_start - base_start
                cumulative_shift += shift_needed
                base_start += shift_needed
        base_end = base_start + span
        prev_end_unwrapped = base_end

        if abs(cumulative_shift) < 1e-9:
            continue

        rotate_nodes: Set[int] = set()
        for depth2_id in group["depth2"]:
            rotate_nodes.update(_collect_subtree_nodes(children_map, int(depth2_id)))

        if not rotate_nodes:
            continue

        delta = cumulative_shift % full_turn
        cos_a = math.cos(delta)
        sin_a = math.sin(delta)
        for node_id in rotate_nodes:
            if node_id not in updated:
                continue
            x, y = updated[node_id]
            updated[node_id] = (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

    return updated


def _apply_sibling_ladder_to_render_positions(
    render_positions: Dict[int, Tuple[float, float]],
    nodes_payload: List[Dict[str, object]],
    *,
    bubble_radius: float,
) -> Dict[int, Tuple[float, float]]:
    
    num_layers = 4
    """Apply a ladder shift for siblings that overlap."""
    if not render_positions or not nodes_payload:
        return render_positions
    if bubble_radius <= 0:
        return render_positions

    updated = dict(render_positions)
    bubble_size = bubble_radius * 2.0
    twin_distance = bubble_radius  # twins spaced by one radius
    min_distance_diff = bubble_radius * 2.0  # different tree ids at least one diameter apart
    min_distance_diff_sq = min_distance_diff * min_distance_diff
    arc_shift = bubble_size / 1.5

    meta_by_id: Dict[int, Dict[str, object]] = {
        int(node["id"]): node for node in nodes_payload if "id" in node
    }
    siblings_by_parent: Dict[int, List[int]] = {}
    same_tree_groups: Dict[int, List[int]] = {}
    for node_meta in nodes_payload:
        try:
            rid = int(node_meta.get("id", 0))
            parent_id = int(node_meta.get("parent", 0))
            tree_id = int(node_meta.get("tree_id", 0))
        except Exception:
            continue
        siblings_by_parent.setdefault(parent_id, []).append(rid)
        same_tree_groups.setdefault(tree_id, []).append(rid)

    locked_same_tree: Set[int] = set()
    for tree_id, group in same_tree_groups.items():
        if len(group) < 2:
            continue
        anchor_id = tree_id if tree_id in group else group[0]
        if anchor_id not in updated:
            continue
        ax, ay = updated[anchor_id]
        r = math.hypot(ax, ay)
        if r <= 1e-6:
            continue
        base_angle = math.atan2(ay, ax)
        step_angle = 2.0 * math.asin(min(1.0, twin_distance / (2.0 * r)))
        ordered = sorted(
            group,
            key=lambda rid: int(meta_by_id.get(rid, {}).get("product_index", 0)),
        )
        idx_offset = 0
        for rid in ordered:
            locked_same_tree.add(rid)
            if rid == anchor_id:
                continue
            idx_offset += 1
            angle = base_angle + step_angle * idx_offset
            updated[rid] = (r * math.cos(angle), r * math.sin(angle))

    for parent_id, render_ids in siblings_by_parent.items():
        if len(render_ids) < 2:
            continue
        ordered_all = sorted(
            [rid for rid in render_ids if rid in updated],
            key=lambda rid: math.atan2(updated[rid][1], updated[rid][0]),
        )
        placed: List[int] = [rid for rid in ordered_all if rid in locked_same_tree]
        ordered = [rid for rid in ordered_all if rid not in locked_same_tree]
        layer_index = 0
        arc_steps = 0

        for rid in ordered:
            x, y = updated[rid]
            tree_id = None
            try:
                tree_id = int(meta_by_id.get(rid, {}).get("tree_id"))
            except Exception:
                tree_id = None
            base_r = math.hypot(x, y)
            base_angle = math.atan2(y, x)
            attempts = 0
            max_attempts = 12

            while True:
                too_close_diff = False
                for placed_id in placed:
                    ox, oy = updated[placed_id]
                    dx = x - ox
                    dy = y - oy
                    if (dx * dx + dy * dy) <= min_distance_diff_sq:
                        try:
                            placed_tree_id = int(meta_by_id.get(placed_id, {}).get("tree_id"))
                        except Exception:
                            placed_tree_id = None
                        if placed_tree_id is None or tree_id is None or placed_tree_id != tree_id:
                            too_close_diff = True
                            break

                if not too_close_diff or attempts >= max_attempts:
                    break

                layer_index = (layer_index + 1) % num_layers
                arc_steps += 1
                
                new_r = base_r + (layer_index * bubble_size)
                angle = base_angle + ((arc_shift * arc_steps) / max(new_r, 1e-6))
                x, y = (new_r * math.cos(angle), new_r * math.sin(angle))
                attempts += 1

            updated[rid] = (x, y)
            if not too_close_diff:
                layer_index = 0
                arc_steps = 0

            # Final on-the-fly arc shift to avoid any overlap at the same radius.
            min_dist_sq = bubble_size * bubble_size
            if base_r > 1e-9:
                angle = math.atan2(y, x)
                arc_step = bubble_size / max(base_r, 1e-6)
                arc_attempts = 0
                while arc_attempts < 24:
                    overlap_any = False
                    for placed_id in placed:
                        ox, oy = updated[placed_id]
                        dx = x - ox
                        dy = y - oy
                        try:
                            placed_tree_id = int(meta_by_id.get(placed_id, {}).get("tree_id"))
                        except Exception:
                            placed_tree_id = None
                        local_min_sq = min_dist_sq
                        if placed_tree_id is not None and tree_id is not None and placed_tree_id == tree_id:
                            local_min_sq = twin_distance * twin_distance
                        if (dx * dx + dy * dy) <= local_min_sq:
                            overlap_any = True
                            break
                    if not overlap_any:
                        break
                    angle += arc_step
                    x, y = (base_r * math.cos(angle), base_r * math.sin(angle))
                    arc_attempts += 1
                updated[rid] = (x, y)
            placed.append(rid)

    for node_meta in nodes_payload:
        try:
            rid = int(node_meta.get("id", 0))
        except Exception:
            continue
        if rid in updated:
            node_meta["x"], node_meta["y"] = updated[rid]

    return updated



def _bounds_with_pad(
    positions: Dict[int, Tuple[float, float]],
    base_radius: float,
    bubble_radius: float,
    pad_scale: float = 6.0,
) -> Tuple[float, float, float, float]:
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    # Ensure large bubbles are not clipped while keeping padding reasonable.
    pad = max(base_radius * pad_scale, bubble_radius * 1.05, 0.1)
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    return min_x, min_y, max_x - min_x, max_y - min_y


def _build_payload(
    tree: Tree,
    *,
    max_nodes: Optional[int],
    sample_step: int,
    with_svg: bool,
    radius_step: float,
    render_scale: float,
    node_radius: Optional[float],
    bubble_scale: float,
) -> Dict[str, object]:
    selected_ids = _select_node_ids(tree, max_nodes=max_nodes)
    allowed_tree_nodes: Set[int] = set(selected_ids)
    allowed_tree_nodes.add(1)

    children_map = base_vis._build_children_map(tree, allowed_nodes=allowed_tree_nodes)
    nodes_depth_tree = base_vis._compute_depths(children_map)
    if not nodes_depth_tree:
        raise ValueError("Tree has no nodes to render.")
    leaf_counts = base_vis._compute_subtree_leaf_counts(children_map)
    layout_children_map = _reorder_children_for_layout(children_map, leaf_counts)

    num_nodes = len(nodes_depth_tree)
    base_radius = node_radius if node_radius is not None else _auto_node_radius(num_nodes)
    bubble_scale = max(0.1, float(bubble_scale))
    has_expandable_nodes = _has_expandable_nodes(tree, allowed_tree_nodes)
    base_scaled_radius = base_radius * max(0.01, float(render_scale))
    bubble_radius = base_scaled_radius * bubble_scale
    step_distance = float(radius_step) * max(0.01, float(render_scale))
    angles = base_vis._assign_subtree_angles(layout_children_map, leaf_counts)
    positions_tree = _petal_layout_from_depth1(
        layout_children_map,
        nodes_depth_tree,
        getattr(tree, "parents", {}) or {},
        step=step_distance,
        bubble_radius=bubble_radius,
        root_id=1,
        arc_span= math.pi/2,
        depth1_angles=angles,
    )
    positions_tree = _apply_depth2_parent_arc_gap(
        positions_tree,
        children_map,
        nodes_depth_tree,
        bubble_radius=bubble_radius,
        gap_diameters=2.0,
        root_id=1,
    )

    winning_nodes = list(getattr(tree, "winning_nodes", []) or [])
    route_index_by_tree = {int(node_id): idx for idx, node_id in enumerate(winning_nodes)}

    tree_steps = _creation_steps(nodes_depth_tree.keys(), sample_step=sample_step)
    total_steps = max(tree_steps.values()) if tree_steps else 0

    nodes_visit = getattr(tree, "nodes_visit", {}) or {}
    nodes_rules = getattr(tree, "nodes_rules", {}) or {}
    nodes_rule_label = getattr(tree, "nodes_rule_label", {}) or {}
    children = getattr(tree, "children", {}) or {}

    # Molecule-level render nodes: one bubble per product molecule.
    tree_to_render_ids: Dict[int, List[int]] = {}
    anchor_by_tree: Dict[int, int] = {}
    render_positions: Dict[int, Tuple[float, float]] = {}
    render_steps: Dict[int, int] = {}

    nodes_payload: List[Dict[str, object]] = []
    next_render_id = (max(allowed_tree_nodes) + 1) if allowed_tree_nodes else 2

    for node_id in sorted(nodes_depth_tree.keys()):
        node = tree.nodes.get(node_id)
        if node is None:
            continue

        base_x, base_y = positions_tree.get(node_id, (0.0, 0.0))
        parent_tree_id = int(tree.parents.get(node_id, 0)) if hasattr(tree, "parents") else 0
        parent_pos = positions_tree.get(parent_tree_id) if parent_tree_id in positions_tree else None

        molecules = _node_product_molecules(node)
        if not molecules:
            molecules = [None]
        product_count = len(molecules)

        # Anchor children to the product that matches the precursor expanded at this node.
        molecule_keys = [_molecule_key(mol) for mol in molecules]
        curr_key = _curr_precursor_key(node)
        anchor_index = 0
        if curr_key:
            for idx, key in enumerate(molecule_keys):
                if key and key == curr_key:
                    anchor_index = idx
                    break

        product_positions = _spread_product_positions(
            float(base_x),
            float(base_y),
            product_count,
            float(bubble_radius),
            parent_pos=parent_pos,
        )
        if anchor_index != 0 and anchor_index < len(product_positions):
            product_positions[0], product_positions[anchor_index] = (
                product_positions[anchor_index],
                product_positions[0],
            )

        render_ids: List[int] = []
        anchor_render_id = int(node_id)
        extra_counter = 0
        node_status = _node_status(tree, node_id)
        if node_status == "leaf" and product_count > 1:
            product_order = [i for i in range(product_count) if i != anchor_index] + [
                anchor_index
            ]
        else:
            product_order = list(range(product_count))

        for product_index in product_order:
            molecule = molecules[product_index]
            is_anchor = product_index == anchor_index
            if is_anchor:
                render_id = int(node_id)
                extra_rank = 0
            else:
                render_id = int(next_render_id)
                next_render_id += 1
                extra_counter += 1
                extra_rank = int(extra_counter)

            render_ids.append(render_id)
            if is_anchor:
                anchor_render_id = render_id

            px, py = product_positions[min(product_index, len(product_positions) - 1)]
            render_positions[render_id] = (float(px), float(py))
            render_steps[render_id] = int(tree_steps.get(node_id, 0))

            smiles, svg = _molecule_smiles_and_svg(molecule, with_svg=with_svg)
            product_status = node_status
            if molecule is not None:
                try:
                    if Precursor(molecule).is_building_block(
                        tree.building_blocks, tree.config.min_mol_size
                    ):
                        product_status = "leaf"
                except Exception:
                    pass
            num_children = int(len(children.get(node_id, []))) if is_anchor else 0
            display_id = str(node_id) if is_anchor else f"{node_id}.{extra_rank}"

            nodes_payload.append(
                {
                    "id": render_id,
                    "tree_id": int(node_id),
                    "product_index": int(product_index),
                    "product_count": int(product_count),
                    "anchor_index": int(anchor_index),
                    "is_extra_product": bool(not is_anchor),
                    "extra_rank": int(extra_rank),
                    "display_id": display_id,
                    "tree_display_id": str(node_id),
                    "x": float(px),
                    "y": float(py),
                    "depth": int(nodes_depth_tree.get(node_id, 0)),
                    "parent": int(parent_tree_id) if is_anchor else int(node_id),
                    "visits": int(nodes_visit.get(node_id, 0)),
                    "num_children": num_children,
                    "rule_id": nodes_rules.get(node_id),
                    "rule_label": nodes_rule_label.get(node_id),
                    "status": product_status,
                    "is_solved": bool(getattr(node, "is_solved", lambda: False)()),
                    "step": int(render_steps.get(render_id, 0)),
                    "route_index": route_index_by_tree.get(node_id),
                    "smiles": smiles,
                    "svg": svg,
                }
            )

        tree_to_render_ids[node_id] = render_ids
        anchor_by_tree[node_id] = anchor_render_id

    render_positions = _apply_sibling_ladder_to_render_positions(
        render_positions, nodes_payload, bubble_radius=bubble_radius
    )

    # Render edges: connect each parent anchor to every product bubble.
    edges_payload: List[Dict[str, object]] = []
    render_edge_keys_by_tree_edge: Dict[str, List[str]] = {}
    for child_id, parent_id in getattr(tree, "parents", {}).items():
        if not parent_id or child_id == 1:
            continue
        if child_id not in nodes_depth_tree or parent_id not in nodes_depth_tree:
            continue

        parent_anchor = int(anchor_by_tree.get(parent_id, parent_id))
        child_render_ids = tree_to_render_ids.get(int(child_id)) or [int(child_id)]
        child_anchor = int(anchor_by_tree.get(child_id, child_id))
        tree_edge_key = _edge_key(parent_id, child_id)

        # Main tree edge: parent anchor -> child anchor.
        main_key = _edge_key(parent_anchor, child_anchor)
        edges_payload.append(
            {
                "parent": int(parent_anchor),
                "child": int(child_anchor),
                "step": int(tree_steps.get(child_id, 0)),
                "key": main_key,
                "tree_key": tree_edge_key,
                "winning": False,
            }
        )

        # Extra product bubbles: connect from child anchor to each extra product.
        for render_child_id in child_render_ids:
            if int(render_child_id) == int(child_anchor):
                continue
            render_key = _edge_key(child_anchor, render_child_id)
            edges_payload.append(
                {
                    "parent": int(child_anchor),
                    "child": int(render_child_id),
                    "step": int(tree_steps.get(child_id, 0)),
                    "key": render_key,
                    "tree_key": tree_edge_key,
                    "winning": False,
                }
            )

        render_edge_keys_by_tree_edge[tree_edge_key] = [main_key]

    winning_edge_keys, winning_paths, win_first_step, win_last_step = _winning_route_paths(
        tree,
        allowed_tree_nodes=allowed_tree_nodes,
        tree_steps=tree_steps,
        anchor_by_tree=anchor_by_tree,
        render_edge_keys_by_tree_edge=render_edge_keys_by_tree_edge,
    )

    if winning_edge_keys:
        for edge in edges_payload:
            if edge.get("key") in winning_edge_keys:
                edge["winning"] = True

    # Recolor extra product bubbles that touch winning edges to green.
    winning_edge_connected_ids: Set[int] = set()
    for edge in edges_payload:
        if not edge.get("winning"):
            continue
        try:
            winning_edge_connected_ids.add(int(edge.get("parent", 0)))
            winning_edge_connected_ids.add(int(edge.get("child", 0)))
        except Exception:
            continue
    if winning_edge_connected_ids:
        for node_meta in nodes_payload:
            try:
                node_id_val = int(node_meta.get("id", 0))
            except Exception:
                continue
            if (
                node_meta.get("is_extra_product")
                and int(node_meta.get("num_children", 0)) == 0
                and node_meta.get("status") == "intermediate"
                and node_id_val in winning_edge_connected_ids
            ):
                node_meta["status"] = "leaf"

    positions = render_positions if render_positions else positions_tree
    max_depth = max(nodes_depth_tree.values()) if nodes_depth_tree else 0
    min_x, min_y, view_w, view_h = _bounds_with_pad(
        positions, base_scaled_radius, bubble_radius
    )

    step_counts = [0 for _ in range(total_steps + 1)]
    for step in render_steps.values():
        step_counts[step] += 1
    cumulative_counts: List[int] = []
    running = 0
    for count in step_counts:
        running += count
        cumulative_counts.append(running)

    winning_node_ids: Set[int] = set()
    for path in winning_paths:
        try:
            winning_node_ids.add(int(path.get("nodeId")))
        except Exception:
            pass
    for edge in edges_payload:
        if edge.get("winning"):
            winning_node_ids.add(int(edge.get("parent", 0)))
            winning_node_ids.add(int(edge.get("child", 0)))

    return {
        "nodes": nodes_payload,
        "edges": edges_payload,
        "stats": {
            "totalNodesInTree": int(len(getattr(tree, "nodes", {}) or {})),
            "renderedNodes": int(len(nodes_payload)),
            "renderedEdges": int(len(edges_payload)),
            "maxDepth": int(max_depth),
            "totalSteps": int(total_steps),
            "totalIterations": int(getattr(tree, "curr_iteration", total_steps)),
            "withSvg": bool(with_svg),
            "bubbleScale": float(bubble_scale),
            "spacingScale": float(1.0),
            "hasExpandableNodes": bool(has_expandable_nodes),
            "winningNodesTotal": int(len(winning_node_ids)),
            "winningEdgesTotal": int(len(winning_edge_keys)),
            "winningFirstStep": int(win_first_step) if win_first_step is not None else None,
            "winningLastStep": int(win_last_step) if win_last_step is not None else None,
        },
        "winningPaths": winning_paths,
        "steps": {
            "counts": step_counts,
            "cumulative": cumulative_counts,
        },
        "viewBox": {
            "minX": float(min_x),
            "minY": float(min_y),
            "width": float(view_w),
            "height": float(view_h),
        },
        "nodeRadius": float(bubble_radius),
    }


def generate_expansion_html(
    tree: Tree,
    output_path: Path,
    *,
    max_nodes: Optional[int],
    sample_step: int,
    with_svg: bool,
    radius_step: float,
    render_scale: float,
    node_radius: Optional[float],
    bubble_scale: float,
) -> None:
    payload = _build_payload(
        tree,
        max_nodes=max_nodes,
        sample_step=sample_step,
        with_svg=with_svg,
        radius_step=radius_step,
        render_scale=render_scale,
        node_radius=node_radius,
        bubble_scale=bubble_scale,
    )

    nodes_json = json.dumps(payload["nodes"], ensure_ascii=True)
    edges_json = json.dumps(payload["edges"], ensure_ascii=True)
    stats_json = json.dumps(payload["stats"], ensure_ascii=True)
    winning_paths_json = json.dumps(payload.get("winningPaths", []), ensure_ascii=True)
    steps_json = json.dumps(payload["steps"], ensure_ascii=True)
    view_box_json = json.dumps(payload["viewBox"], ensure_ascii=True)
    node_radius = float(payload["nodeRadius"])

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SynPlanner Tree Expansion Timeline</title>
    <style>
      :root {{
        --bg: #0b0f14;
        --panel: #111722;
        --panel-border: #1c2432;
        --text: #e6edf3;
        --muted: #9aa7b3;
        --edge: #2b3648;
        --edge-visible: #5a6c8a;
        --edge-winning: #ffd166;
        --edge-winning-glow: rgba(255, 209, 102, 0.45);
        --target: #4e8cff;
        --leaf: #4cd97b;
        --intermediate: #ff5c5c;
        --node-stroke: #0b0f14;
      }}

      html, body {{
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        background: var(--bg);
        color: var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      }}

      #app {{
        position: relative;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
      }}

      svg {{
        width: 100%;
        height: 100%;
        display: block;
        background: radial-gradient(1200px 900px at 50% 45%, #0f1623 0%, var(--bg) 70%);
        cursor: grab;
      }}

      svg.panning {{
        cursor: grabbing;
      }}

      .edges line {{
        stroke: var(--edge);
        stroke-width: {min(max(node_radius * 0.5, 0.6), 6.0):.3f};
        opacity: 0;
        pointer-events: none;
      }}

      .edges line.visible {{
        opacity: 0.9;
        stroke: var(--edge-visible);
      }}

      .edges line.hidden-nonwinning {{
        opacity: 0 !important;
        pointer-events: none;
      }}

      .edges line.winning-route {{
        /* Marker class for winning-route edges; styling is applied when active. */
      }}

      .edges line.visible.winning-active {{
        opacity: 1;
        stroke: var(--edge-winning);
        stroke-width: {min(max(node_radius * 0.75, 1.4), 10.0):.3f};
        filter: drop-shadow(0 0 4px var(--edge-winning-glow));
      }}

      .nodes circle {{
        r: {node_radius:.6f};
        opacity: 0;
        stroke: var(--node-stroke);
        stroke-width: {min(max(node_radius * 0.35, 0.8), 8.0):.3f};
        transform-origin: center;
        cursor: pointer;
      }}

      .nodes circle.visible {{
        opacity: 1;
      }}

      .nodes circle.hidden-nonwinning {{
        opacity: 0 !important;
        pointer-events: none;
      }}

      .nodes circle.status-target {{ fill: var(--target); }}
      .nodes circle.status-leaf {{ fill: var(--leaf); }}
      .nodes circle.status-intermediate {{ fill: var(--intermediate); }}
      .nodes circle.active {{
        stroke: #ffffff;
        stroke-width: {min(max(node_radius * 0.6, 1.2), 10.0):.3f};
        filter: drop-shadow(0 0 6px rgba(255, 255, 255, 0.35));
      }}

      .panel {{
        position: absolute;
        background: color-mix(in srgb, var(--panel) 86%, transparent);
        border: 1px solid var(--panel-border);
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(6px);
      }}

      #controls {{
        top: 14px;
        left: 14px;
        padding: 0;
        width: min(420px, calc(100vw - 28px));
        overflow: hidden;
      }}

      #controls.collapsed #controls-body {{
        display: none;
      }}

      #hide-panel {{
        left: 50%;
        bottom: 14px;
        transform: translateX(-50%);
        padding: 8px 10px;
        z-index: 2;
      }}

      #recenter-panel {{
        left: 50%;
        top: 14px;
        transform: translateX(-50%);
        padding: 6px 10px;
        z-index: 2;
      }}

      #branch-btn {{
        position: absolute;
        z-index: 3;
        display: none;
        padding: 4px 8px;
        font-size: 12px;
        border-radius: 8px;
        border: 1px solid #1a2437;
        background: #101826;
        color: var(--text);
        cursor: pointer;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
      }}

      #hide-panel button.active {{
        border-color: var(--edge-winning);
        color: var(--edge-winning);
      }}

      #hide-panel button:disabled {{
        opacity: 0.6;
        cursor: not-allowed;
      }}

      #winning-panel {{
        top: 14px;
        right: 14px;
        left: auto;
        padding: 0;
        width: min(320px, calc(100vw - 28px));
        max-height: min(46vh, 360px);
        overflow: hidden;
        z-index: 2;
      }}

      #winning-panel.collapsed #winning-body {{
        display: none;
      }}

      #info {{
        right: 14px;
        bottom: 14px;
        left: auto;
        padding: 0;
        width: min(420px, calc(100vw - 28px));
        max-height: min(60vh, 520px);
        overflow: hidden;
        display: none;
      }}

      #info.open {{
        display: block;
      }}

      #info.collapsed #info-body {{
        display: none;
      }}

      #controls .panel-header,
      #winning-panel .panel-header,
      #info .panel-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 10px 12px;
        border-bottom: 1px solid #1a2437;
      }}

      #controls.collapsed .panel-header,
      #winning-panel.collapsed .panel-header,
      #info.collapsed .panel-header {{
        border-bottom: none;
      }}

      #controls .panel-header h3,
      #winning-panel .panel-header h3,
      #info .panel-header h3 {{
        margin: 0;
        font-size: 14px;
      }}

      #controls .panel-header button,
      #winning-panel .panel-header button,
      #info .panel-header button {{
        padding: 4px 8px;
        font-size: 12px;
      }}

      #controls-body {{
        padding: 10px 12px 12px 12px;
        max-height: min(60vh, 480px);
        overflow: auto;
      }}

      #winning-body {{
        padding: 10px 12px 12px 12px;
        max-height: calc(min(46vh, 360px) - 46px);
        overflow: auto;
      }}

      #info-body {{
        padding: 10px 12px 12px 12px;
        max-height: calc(min(60vh, 520px) - 46px);
        overflow: auto;
      }}

      .row {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 6px 0;
        flex-wrap: wrap;
      }}

      .row label {{
        font-size: 12px;
        color: var(--muted);
        min-width: 68px;
      }}

      input[type="range"] {{
        flex: 1;
        min-width: 120px;
      }}

      button, select {{
        background: #1a2332;
        color: var(--text);
        border: 1px solid #263247;
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 12px;
        cursor: pointer;
      }}

      button:hover, select:hover {{
        border-color: #3a4b68;
      }}

      .stat-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 6px;
        margin-top: 8px;
      }}

      .stat {{
        background: #0f1522;
        border: 1px solid #1a2437;
        border-radius: 8px;
        padding: 6px 8px;
      }}

      .stat .label {{
        font-size: 11px;
        color: var(--muted);
      }}

      .stat .value {{
        font-size: 14px;
        font-weight: 600;
        margin-top: 2px;
      }}

      #info .muted,
      #winning-panel .muted {{
        color: var(--muted);
        font-size: 12px;
      }}

      #info .kv,
      #winning-panel .kv {{
        display: grid;
        grid-template-columns: 110px minmax(0, 1fr);
        gap: 4px 8px;
        font-size: 12px;
        margin-top: 6px;
      }}

      #info .kv div:nth-child(odd),
      #winning-panel .kv div:nth-child(odd) {{
        color: var(--muted);
      }}

      #mol-svg {{
        margin-top: 8px;
        margin-bottom: 6px;
        border: 1px solid #1a2437;
        border-radius: 8px;
        padding: 6px;
        background: #ffffff;
      }}

      #mol-svg svg {{
        width: 70%;
        height: auto;
        display: block;
        margin: 0 auto;
      }}

      #legend {{
        margin-top: 8px;
        font-size: 12px;
        color: var(--muted);
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
      }}

      .legend-item {{
        display: flex;
        align-items: center;
        gap: 6px;
      }}

      .legend-dot {{
        width: 10px;
        height: 10px;
        border-radius: 999px;
        border: 1px solid #0b0f14;
      }}
    </style>
  </head>
  <body>
    <div id="app">
      <svg id="tree" aria-label="Tree expansion timeline">
        <g class="edges" id="edges"></g>
        <g class="nodes" id="nodes"></g>
      </svg>

      <div class="panel" id="controls">
        <div class="panel-header">
          <h3>Expansion Timeline</h3>
          <button id="controls-toggle" type="button" aria-expanded="true">Collapse</button>
        </div>
        <div id="controls-body">
          <div class="row">
            <label for="step">Step</label>
            <input id="step" type="range" min="0" value="0" />
            <span id="step-label">0</span>
          </div>
          <div class="row">
            <label for="iteration">Iterations</label>
            <input id="iteration" type="range" min="0" value="0" />
            <span id="iteration-label">0</span>
          </div>
          <div class="row">
            <label>Playback</label>
            <button id="play">Play</button>
            <button id="reset">Reset</button>
            <select id="speed" aria-label="Playback speed">
              <option value="40">Very Fast</option>
              <option value="80" selected>Fast</option>
              <option value="140">Medium</option>
              <option value="220">Slow</option>
            </select>
          </div>
          <div class="stat-grid" id="stats"></div>
          <div id="legend">
            <div class="legend-item"><span class="legend-dot" style="background: var(--target)"></span>Target</div>
            <div class="legend-item"><span class="legend-dot" style="background: var(--intermediate)"></span>Intermediate</div>
            <div class="legend-item"><span class="legend-dot" style="background: var(--leaf)"></span>Starting Material</div>
          </div>
        </div>
      </div>

      <div class="panel" id="winning-panel" role="status" aria-live="polite">
        <div class="panel-header">
          <h3>Winning Routes</h3>
          <button id="winning-toggle" type="button" aria-expanded="true">Collapse</button>
        </div>
        <div id="winning-body">
          <div class="muted" id="winning-summary">Winning nodes: —</div>
          <div class="kv" id="winning-kv"></div>
        </div>
      </div>

      <div class="panel" id="hide-panel">
        <button id="hide-nonwinning-btn" type="button" aria-pressed="false">Hide</button>
      </div>

      <div class="panel" id="recenter-panel">
        <button id="recenter-btn" type="button">Recenter</button>
      </div>

      <button id="branch-btn" type="button">Hide branch</button>

      <div class="panel" id="info" role="dialog" aria-live="polite">
        <div class="panel-header">
          <h3>Node Details</h3>
          <button id="info-toggle" type="button" aria-expanded="true">Collapse</button>
        </div>
        <div id="info-body">
          <div class="muted">Click a node to inspect it.</div>
          <div id="mol-svg"></div>
          <div class="kv" id="kv"></div>
        </div>
      </div>
    </div>

    <script>
      const nodes = {nodes_json};
      const edges = {edges_json};
      const stats = {stats_json};
      const winningPaths = {winning_paths_json};
      const steps = {steps_json};
      const viewBox = {view_box_json};

      const svg = document.getElementById("tree");
      const edgesGroup = document.getElementById("edges");
      const nodesGroup = document.getElementById("nodes");
      const stepInput = document.getElementById("step");
      const stepLabel = document.getElementById("step-label");
      const iterationInput = document.getElementById("iteration");
      const iterationLabel = document.getElementById("iteration-label");
      const playBtn = document.getElementById("play");
      const resetBtn = document.getElementById("reset");
      const speedSel = document.getElementById("speed");
      const controlsEl = document.getElementById("controls");
      const controlsToggleBtn = document.getElementById("controls-toggle");
      const statsEl = document.getElementById("stats");
      const hidePanelEl = document.getElementById("hide-panel");
      const hideNonWinningBtn = document.getElementById("hide-nonwinning-btn");
      const recenterBtn = document.getElementById("recenter-btn");
      const branchBtn = document.getElementById("branch-btn");
      const winningPanelEl = document.getElementById("winning-panel");
      const winningToggleBtn = document.getElementById("winning-toggle");
      const winningSummaryEl = document.getElementById("winning-summary");
      const winningKvEl = document.getElementById("winning-kv");
      const infoEl = document.getElementById("info");
      const infoToggleBtn = document.getElementById("info-toggle");
      const kvEl = document.getElementById("kv");
      const molSvgEl = document.getElementById("mol-svg");

      const totalSteps = Math.max(0, stats.totalSteps || 0);
      const totalIterations = Math.max(0, stats.totalIterations || totalSteps);
      stepInput.max = String(totalSteps);
      if (iterationInput) iterationInput.max = String(totalIterations);

      const baseViewBox = {{
        x: Number(viewBox.minX || 0),
        y: Number(viewBox.minY || 0),
        width: Number(viewBox.width || 1),
        height: Number(viewBox.height || 1),
      }};
      const viewBoxState = {{
        x: baseViewBox.x,
        y: baseViewBox.y,
        width: baseViewBox.width,
        height: baseViewBox.height,
      }};

      function applyViewBox() {{
        svg.setAttribute(
          "viewBox",
          [viewBoxState.x, viewBoxState.y, viewBoxState.width, viewBoxState.height].join(" ")
        );
        updateBranchButtonPosition();
      }}

      function getSvgPoint(evt) {{
        const point = svg.createSVGPoint();
        point.x = evt.clientX;
        point.y = evt.clientY;
        const ctm = svg.getScreenCTM();
        if (!ctm) return {{ x: 0, y: 0 }};
        const inverse = ctm.inverse();
        const svgPoint = point.matrixTransform(inverse);
        return {{ x: svgPoint.x, y: svgPoint.y }};
      }}

      function svgPointToClient(x, y) {{
        const rect = svg.getBoundingClientRect();
        const scaleX = rect.width / viewBoxState.width;
        const scaleY = rect.height / viewBoxState.height;
        return {{
          x: rect.left + (x - viewBoxState.x) * scaleX,
          y: rect.top + (y - viewBoxState.y) * scaleY,
        }};
      }}

      const zoomMin = 0.1;
      const zoomMax = 10.0;

      function zoomAt(point, factor) {{
        if (!Number.isFinite(factor) || factor <= 0) return;

        const minWidth = baseViewBox.width * zoomMin;
        const maxWidth = baseViewBox.width * zoomMax;
        const nextWidth = viewBoxState.width * factor;
        const clampedWidth = Math.max(minWidth, Math.min(maxWidth, nextWidth));
        const actualFactor = clampedWidth / viewBoxState.width;
        const clampedHeight = viewBoxState.height * actualFactor;

        viewBoxState.x = point.x - (point.x - viewBoxState.x) * actualFactor;
        viewBoxState.y = point.y - (point.y - viewBoxState.y) * actualFactor;
        viewBoxState.width = clampedWidth;
        viewBoxState.height = clampedHeight;
        applyViewBox();
      }}

      function resetViewBox() {{
        viewBoxState.x = baseViewBox.x;
        viewBoxState.y = baseViewBox.y;
        viewBoxState.width = baseViewBox.width;
        viewBoxState.height = baseViewBox.height;
        applyViewBox();
      }}

      function onWheel(evt) {{
        evt.preventDefault();
        const point = getSvgPoint(evt);
        const direction = evt.deltaY > 0 ? 1 : -1;
        const factor = direction > 0 ? 1.1 : 0.9;
        zoomAt(point, factor);
      }}

      let isPanning = false;
      let panStart = null;
      let panStartViewBox = null;

      function startPan(evt) {{
        if (evt.button !== 0) return;
        if (evt.target && evt.target.closest && evt.target.closest("circle, rect")) return;
        isPanning = true;
        svg.classList.add("panning");
        panStart = getSvgPoint(evt);
        panStartViewBox = {{ ...viewBoxState }};
      }}

      function movePan(evt) {{
        if (!isPanning || !panStart || !panStartViewBox) return;
        const point = getSvgPoint(evt);
        const dx = panStart.x - point.x;
        const dy = panStart.y - point.y;
        viewBoxState.x = panStartViewBox.x + dx;
        viewBoxState.y = panStartViewBox.y + dy;
        applyViewBox();
      }}

      function endPan() {{
        if (!isPanning) return;
        isPanning = false;
        svg.classList.remove("panning");
        panStart = null;
        panStartViewBox = null;
      }}

      svg.addEventListener("wheel", onWheel, {{ passive: false }});
      svg.addEventListener("mousedown", startPan);
      window.addEventListener("mousemove", movePan);
      window.addEventListener("mouseup", endPan);
      svg.addEventListener("mouseleave", endPan);
      svg.addEventListener("dblclick", (evt) => {{
        evt.preventDefault();
        resetViewBox();
      }});
      svg.addEventListener("click", (evt) => {{
        if (evt.target && evt.target.closest && evt.target.closest("circle")) return;
        hideBranchButton();
      }});

      const nodeMeta = new Map(nodes.map((n) => [n.id, n]));
      const treeParentById = new Map();
      const treeChildrenById = new Map();
      const renderIdsByTree = new Map();
      for (const node of nodes) {{
        const treeId = Number(node.tree_id ?? node.treeId ?? node.tree_id);
        if (!Number.isFinite(treeId)) continue;
        if (!treeParentById.has(treeId)) {{
          const parentVal = Number(node.parent || 0);
          treeParentById.set(treeId, Number.isFinite(parentVal) ? parentVal : 0);
        }}
        if (!renderIdsByTree.has(treeId)) renderIdsByTree.set(treeId, []);
        renderIdsByTree.get(treeId).push(Number(node.id));
      }}
      for (const [treeId, parentId] of treeParentById) {{
        if (!parentId) continue;
        if (!treeChildrenById.has(parentId)) treeChildrenById.set(parentId, []);
        treeChildrenById.get(parentId).push(treeId);
      }}

      const nodesByStepMeta = Array.from({{ length: totalSteps + 1 }}, () => []);
      const edgesByStepMeta = Array.from({{ length: totalSteps + 1 }}, () => []);
      const nodesByStepEls = Array.from({{ length: totalSteps + 1 }}, () => []);
      const edgesByStepEls = Array.from({{ length: totalSteps + 1 }}, () => []);

      const nodeEls = new Map();
      const edgeEls = new Map();
      const edgeMetaByKey = new Map();
      let builtStep = -1;
      const hiddenBranches = new Set();
      let branchTargetId = null;
      applyViewBox();

      function stepIndex(value) {{
        const num = Number.isFinite(value) ? value : parseInt(value, 10);
        if (!Number.isFinite(num)) return 0;
        return Math.max(0, Math.min(totalSteps, Math.trunc(num)));
      }}

      function clsForStatus(status) {{
        if (status === "target") return "status-target";
        if (status === "leaf") return "status-leaf";
        return "status-intermediate";
      }}

      function isHiddenByBranch(treeId) {{
        let current = Number(treeId);
        if (!Number.isFinite(current) || current <= 0) return false;
        const seen = new Set();
        while (current && !seen.has(current)) {{
          if (hiddenBranches.has(current)) return current !== treeId;
          seen.add(current);
          current = treeParentById.get(current) || 0;
        }}
        return false;
      }}

      function updateBranchButtonPosition() {{
        if (!branchBtn || !branchBtn.style || branchBtn.style.display === "none") return;
        if (!branchTargetId) return;
        const meta = nodeMeta.get(branchTargetId);
        if (!meta) return;
        const client = svgPointToClient(meta.x, meta.y);
        branchBtn.style.left = `${{client.x + 10}}px`;
        branchBtn.style.top = `${{client.y + 10}}px`;
      }}

      function showBranchButton(node) {{
        if (!branchBtn || !node) return;
        const treeId = Number(node.tree_id ?? node.treeId ?? node.tree_id);
        if (!Number.isFinite(treeId) || treeId <= 0) return;
        branchTargetId = Number(node.id);
        branchBtn.dataset.treeId = String(treeId);
        branchBtn.textContent = hiddenBranches.has(treeId) ? "Show branch" : "Hide branch";
        branchBtn.style.display = "block";
        updateBranchButtonPosition();
      }}

      function hideBranchButton() {{
        if (!branchBtn) return;
        branchBtn.style.display = "none";
        branchTargetId = null;
      }}

      function recenterToTarget() {{
        const target = nodeMeta.get(1);
        if (!target) return;
        const width = viewBoxState.width;
        const height = viewBoxState.height;
        viewBoxState.x = Number(target.x) - width / 2.0;
        viewBoxState.y = Number(target.y) - height / 2.0;
        applyViewBox();
      }}

      for (const node of nodes) {{
        nodesByStepMeta[stepIndex(node.step)].push(node);
      }}
      for (const edge of edges) {{
        const edgeKey = String(edge.key || `${{edge.parent}}->${{edge.child}}`);
        edgeMetaByKey.set(edgeKey, edge);
        edgesByStepMeta[stepIndex(edge.step)].push(edge);
      }}

      function buildStep(step) {{
        const target = stepIndex(step);
        if (target <= builtStep) return;

        for (let s = builtStep + 1; s <= target; s += 1) {{
          const nodeFrag = document.createDocumentFragment();
          for (const node of nodesByStepMeta[s]) {{
            const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            circle.setAttribute("cx", String(node.x));
            circle.setAttribute("cy", String(node.y));
            circle.setAttribute("class", clsForStatus(node.status));
            circle.dataset.id = String(node.id);
            circle.dataset.step = String(node.step);
            nodeFrag.appendChild(circle);
            nodeEls.set(node.id, circle);
            nodesByStepEls[s].push(circle);

            circle.addEventListener("click", () => {{
              setActiveNode(node.id);
              showBranchButton(node);
            }});
          }}
          if (nodeFrag.childNodes.length) {{
            nodesGroup.appendChild(nodeFrag);
          }}

          const edgeFrag = document.createDocumentFragment();
          for (const edge of edgesByStepMeta[s]) {{
            const parent = nodeMeta.get(edge.parent);
            const child = nodeMeta.get(edge.child);
            if (!parent || !child) continue;
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            const edgeKey = edge.key || `${{edge.parent}}->${{edge.child}}`;
            line.setAttribute("x1", String(parent.x));
            line.setAttribute("y1", String(parent.y));
            line.setAttribute("x2", String(child.x));
            line.setAttribute("y2", String(child.y));
            line.dataset.step = String(edge.step);
            line.dataset.key = edgeKey;
            if (edge.winning) {{
              line.classList.add("winning-route");
            }}
            edgeEls.set(edgeKey, line);
            edgeFrag.appendChild(line);
            edgesByStepEls[s].push(line);
          }}
          if (edgeFrag.childNodes.length) {{
            edgesGroup.appendChild(edgeFrag);
          }}
        }}

        builtStep = target;
        updateHiddenNonWinningNodes();
      }}

      function statCard(label, value) {{
        const div = document.createElement("div");
        div.className = "stat";
        div.innerHTML = `<div class="label">${{label}}</div><div class="value">${{value}}</div>`;
        return div;
      }}

      statsEl.appendChild(statCard("Tree Nodes", stats.totalNodesInTree));
      statsEl.appendChild(statCard("Rendered", stats.renderedNodes));
      statsEl.appendChild(statCard("Edges", stats.renderedEdges));
      statsEl.appendChild(statCard("Max Depth", stats.maxDepth));
      statsEl.appendChild(statCard("Steps", stats.totalSteps));
      statsEl.appendChild(statCard("Iterations", stats.totalIterations));
      statsEl.appendChild(statCard("Bubble", stats.bubbleScale));

      let currentStep = -1;
      let playing = false;
      let timer = null;
      let activeNodeId = null;
      let infoCollapsed = false;
      let infoOpen = false;
      let winningCollapsed = false;
      let controlsCollapsed = false;

      const winningPathsData = Array.isArray(winningPaths) ? winningPaths : [];
      const parentById = new Map(nodes.map((n) => [n.id, Number(n.parent || 0)]));
      const hasWinningPaths = winningPathsData.length > 0;
      const winningConnectedNodeIds = new Set([1]);
      if (hasWinningPaths) {{
        for (const path of winningPathsData) {{
          let current = Number(path.nodeId);
          const seen = new Set();
          while (Number.isFinite(current) && current > 0 && !seen.has(current)) {{
            seen.add(current);
            winningConnectedNodeIds.add(current);
            current = parentById.get(current) || 0;
          }}
        }}
      }}
      // Keep any node that touches a winning edge visible when "Hide" is active.
      const winningEdgeConnectedNodeIds = new Set(winningConnectedNodeIds);
      for (const edge of edges) {{
        if (!edge || !edge.winning) continue;
        const parentId = Number(edge.parent || 0);
        const childId = Number(edge.child || 0);
        if (Number.isFinite(parentId) && parentId > 0) {{
          winningEdgeConnectedNodeIds.add(parentId);
        }}
        if (Number.isFinite(childId) && childId > 0) {{
          winningEdgeConnectedNodeIds.add(childId);
        }}
      }}
      const winningTreeIds = new Set();
      for (const nodeId of winningEdgeConnectedNodeIds) {{
        const meta = nodeMeta.get(nodeId);
        if (!meta) continue;
        const treeId = Number(meta.tree_id ?? meta.treeId ?? meta.tree_id);
        if (Number.isFinite(treeId) && treeId > 0) {{
          winningTreeIds.add(treeId);
        }}
      }}

      let hideNonWinning = false;

      function updateHideButton() {{
        if (!hideNonWinningBtn) return;
        hideNonWinningBtn.textContent = hideNonWinning ? "Show" : "Hide";
        hideNonWinningBtn.classList.toggle("active", hideNonWinning);
        hideNonWinningBtn.setAttribute("aria-pressed", String(hideNonWinning));
      }}

      function shouldHideNonWinningNode(nodeId) {{
        if (!hideNonWinning) return false;
        if (Number(nodeId) === 1) return false;
        if (winningEdgeConnectedNodeIds.has(nodeId)) return false;
        const meta = nodeMeta.get(nodeId);
        if (!meta) return true;
        const treeId = Number(meta.tree_id ?? meta.treeId ?? meta.tree_id);
        if (Number.isFinite(treeId) && winningTreeIds.has(treeId)) return false;
        return true;
      }}

      function shouldHideNode(nodeId) {{
        const meta = nodeMeta.get(nodeId);
        if (!meta) return false;
        if (Number(nodeId) === 1) return false;
        const treeId = Number(meta.tree_id ?? meta.treeId ?? meta.tree_id);
        if (isHiddenByBranch(treeId)) return true;
        return shouldHideNonWinningNode(nodeId);
      }}

      function shouldHideNonWinningEdge(edgeKey) {{
        const meta = edgeMetaByKey.get(String(edgeKey));
        if (!meta) return false;
        return shouldHideNode(meta.parent) || shouldHideNode(meta.child);
      }}

      function updateHiddenNonWinningEdges() {{
        for (const [edgeKey, el] of edgeEls) {{
          const hidden = shouldHideNonWinningEdge(edgeKey);
          el.classList.toggle("hidden-nonwinning", hidden);
        }}
      }}

      function updateHiddenNonWinningNodes() {{
        for (const [nodeId, el] of nodeEls) {{
          const hidden = shouldHideNode(nodeId);
          el.classList.toggle("hidden-nonwinning", hidden);
        }}
        updateHiddenNonWinningEdges();
      }}

      if (!hasWinningPaths && hideNonWinningBtn) {{
        hideNonWinningBtn.disabled = true;
        hideNonWinningBtn.title = "No winning nodes available in this tree.";
      }}
      updateHideButton();
      if (hideNonWinningBtn) {{
        hideNonWinningBtn.addEventListener("click", () => {{
          hideNonWinning = !hideNonWinning;
          updateHideButton();
          updateHiddenNonWinningNodes();
        }});
      }}

      if (recenterBtn) {{
        recenterBtn.addEventListener("click", () => {{
          recenterToTarget();
        }});
      }}

      if (branchBtn) {{
        branchBtn.addEventListener("click", (e) => {{
          e.stopPropagation();
          const treeId = Number(branchBtn.dataset.treeId || 0);
          if (!Number.isFinite(treeId) || treeId <= 0) return;
          if (hiddenBranches.has(treeId)) {{
            hiddenBranches.delete(treeId);
          }} else {{
            hiddenBranches.add(treeId);
          }}
          branchBtn.textContent = hiddenBranches.has(treeId) ? "Show branch" : "Hide branch";
          updateHiddenNonWinningNodes();
        }});
      }}

      function setControlsCollapsed(nextCollapsed) {{
        controlsCollapsed = Boolean(nextCollapsed);
        controlsEl.classList.toggle("collapsed", controlsCollapsed);
        controlsToggleBtn.textContent = controlsCollapsed ? "Expand" : "Collapse";
        controlsToggleBtn.setAttribute("aria-expanded", String(!controlsCollapsed));
      }}

      if (controlsToggleBtn) {{
        controlsToggleBtn.addEventListener("click", () => {{
          setControlsCollapsed(!controlsCollapsed);
        }});
        setControlsCollapsed(false);
      }}

      function setInfoCollapsed(nextCollapsed) {{
        infoCollapsed = Boolean(nextCollapsed);
        infoEl.classList.toggle("collapsed", infoCollapsed);
        infoToggleBtn.textContent = infoCollapsed ? "Expand" : "Collapse";
        infoToggleBtn.setAttribute("aria-expanded", String(!infoCollapsed));
      }}

      function ensureInfoOpen() {{
        if (!infoOpen) {{
          infoEl.classList.add("open");
          infoOpen = true;
        }}
        if (infoCollapsed) setInfoCollapsed(false);
      }}

      infoToggleBtn.addEventListener("click", () => {{
        if (!infoOpen) {{
          infoEl.classList.add("open");
          infoOpen = true;
        }}
        setInfoCollapsed(!infoCollapsed);
      }});

      function setWinningCollapsed(nextCollapsed) {{
        winningCollapsed = Boolean(nextCollapsed);
        winningPanelEl.classList.toggle("collapsed", winningCollapsed);
        winningToggleBtn.textContent = winningCollapsed ? "Expand" : "Collapse";
        winningToggleBtn.setAttribute("aria-expanded", String(!winningCollapsed));
      }}

      winningToggleBtn.addEventListener("click", () => {{
        setWinningCollapsed(!winningCollapsed);
      }});

      function clampStep(value) {{
        if (!Number.isFinite(value)) return 0;
        return Math.max(0, Math.min(totalSteps, Math.round(value)));
      }}

      function clampIteration(value) {{
        if (!Number.isFinite(value)) return 0;
        return Math.max(0, Math.min(totalIterations, Math.round(value)));
      }}

      function iterationToStep(iteration) {{
        if (!totalIterations) return clampStep(iteration);
        const ratio = Math.max(0, Math.min(1, iteration / totalIterations));
        return clampStep(ratio * totalSteps);
      }}

      function stepToIteration(step) {{
        if (!totalSteps) return clampIteration(step);
        const ratio = Math.max(0, Math.min(1, step / totalSteps));
        return clampIteration(ratio * totalIterations);
      }}

      function showStep(step) {{
        const idx = clampStep(step);
        buildStep(idx);
        const bins = nodesByStepEls[idx] || [];
        for (const el of bins) el.classList.add("visible");
        const edgeBins = edgesByStepEls[idx] || [];
        for (const el of edgeBins) el.classList.add("visible");
      }}

      function hideStep(step) {{
        const idx = clampStep(step);
        const bins = nodesByStepEls[idx] || [];
        for (const el of bins) el.classList.remove("visible");
        const edgeBins = edgesByStepEls[idx] || [];
        for (const el of edgeBins) el.classList.remove("visible");
      }}

      function setStep(nextStep) {{
        const target = clampStep(nextStep);
        if (target === currentStep) return;

        if (currentStep < 0) {{
          for (let s = 0; s <= target; s += 1) showStep(s);
        }} else if (target > currentStep) {{
          for (let s = currentStep + 1; s <= target; s += 1) showStep(s);
        }} else {{
          for (let s = currentStep; s > target; s -= 1) hideStep(s);
        }}

        currentStep = target;
        stepInput.value = String(target);
        const rendered = steps.cumulative?.[target] ?? "";
        stepLabel.textContent = `${{target}} / ${{totalSteps}} · nodes ${{rendered}}`;
        const iterationValue = stepToIteration(target);
        if (iterationInput) iterationInput.value = String(iterationValue);
        if (iterationLabel) iterationLabel.textContent = `${{iterationValue}} / ${{totalIterations}}`;
        updateWinningHighlights(currentStep);
        updateHiddenNonWinningNodes();
      }}

      function stopPlaying() {{
        playing = false;
        playBtn.textContent = "Play";
        if (timer) {{
          clearInterval(timer);
          timer = null;
        }}
      }}

      function startPlaying() {{
        if (playing) return;
        playing = true;
        playBtn.textContent = "Pause";
        const raw = parseInt(speedSel.value, 10);
        const intervalMs = Number.isFinite(raw) ? Math.max(20, raw) : 80;
        timer = setInterval(() => {{
          if (currentStep >= totalSteps) {{
            stopPlaying();
            return;
          }}
          setStep(currentStep + 1);
        }}, intervalMs);
      }}

      playBtn.addEventListener("click", () => {{
        if (playing) stopPlaying();
        else startPlaying();
      }});

      resetBtn.addEventListener("click", () => {{
        stopPlaying();
        setStep(0);
      }});

      stepInput.addEventListener("input", (e) => {{
        stopPlaying();
        const value = parseInt(e.target.value, 10);
        setStep(value);
      }});
      if (iterationInput) {{
        iterationInput.addEventListener("input", (e) => {{
          stopPlaying();
          const value = parseInt(e.target.value, 10);
          setStep(iterationToStep(value));
        }});
      }}

      speedSel.addEventListener("change", () => {{
        if (playing) {{
          stopPlaying();
          startPlaying();
        }}
      }});

      function setActiveNode(nodeId) {{
        if (activeNodeId !== null) {{
          const prev = nodeEls.get(activeNodeId);
          if (prev) prev.classList.remove("active");
        }}
        activeNodeId = nodeId;
        const el = nodeEls.get(nodeId);
        if (el) el.classList.add("active");
        ensureInfoOpen();
        renderNodeDetails(nodeMeta.get(nodeId));
      }}

      function safe(value) {{
        if (value === null || value === undefined || value === "") return "—";
        return String(value);
      }}

      function computeActiveWinning(step) {{
        const activeEdgeKeys = new Set();
        const activeWinningNodeIds = new Set();
        for (const path of winningPathsData) {{
          const nodeId = Number(path.nodeId);
          if (!Number.isFinite(nodeId)) continue;
          const nodeStep = stepIndex(Number(path.nodeStep ?? 0));
          if (nodeStep > step) continue;
          activeWinningNodeIds.add(nodeId);
          if (Array.isArray(path.edgeKeys)) {{
            for (const key of path.edgeKeys) activeEdgeKeys.add(String(key));
          }}
        }}
        return {{ activeEdgeKeys, activeWinningNodeIds }};
      }}

      function updateWinningPanel(step, activeWinningNodes, activeWinningEdges) {{
        const totalWinningNodes = Number(stats.winningNodesTotal || 0);
        const totalWinningEdges = Number(stats.winningEdgesTotal || 0);

        if (!totalWinningNodes) {{
          winningSummaryEl.textContent = "No winning nodes recorded in this tree.";
          winningKvEl.innerHTML = "";
          return;
        }}

        winningSummaryEl.textContent =
          `Active winning nodes: ${{activeWinningNodes}} / ${{totalWinningNodes}} · step ${{step}}`;

        const rows = [
          ["Winning Nodes (total)", totalWinningNodes],
          ["Winning Nodes (active)", activeWinningNodes],
          ["Winning Edges (active)", activeWinningEdges],
          ["Winning Edges (total)", totalWinningEdges],
          ["First Winning Step", safe(stats.winningFirstStep)],
          ["Last Winning Step", safe(stats.winningLastStep)],
        ];
        winningKvEl.innerHTML = rows
          .map(([k, v]) => `<div>${{k}}</div><div>${{safe(v)}}</div>`)
          .join("");
      }}

      function updateWinningHighlights(step) {{
        const {{ activeEdgeKeys, activeWinningNodeIds }} = computeActiveWinning(step);
        for (const [key, el] of edgeEls) {{
          const isActive = activeEdgeKeys.has(key);
          el.classList.toggle("winning-active", isActive);
        }}
        updateWinningPanel(step, activeWinningNodeIds.size, activeEdgeKeys.size);
      }}

      function ensureWhiteSvgBackground(svgMarkup) {{
        if (!svgMarkup) return "";
        try {{
          const doc = new DOMParser().parseFromString(svgMarkup, "image/svg+xml");
          const svgEl = doc.documentElement;
          if (!svgEl || svgEl.tagName.toLowerCase() !== "svg") return svgMarkup;

          const rect = doc.createElementNS("http://www.w3.org/2000/svg", "rect");
          rect.setAttribute("width", "100%");
          rect.setAttribute("height", "100%");
          rect.setAttribute("fill", "#ffffff");
          rect.setAttribute("pointer-events", "none");
          svgEl.insertBefore(rect, svgEl.firstChild);

          const style = svgEl.getAttribute("style") || "";
          svgEl.setAttribute("style", `${{style}};background:#ffffff;`);

          return new XMLSerializer().serializeToString(svgEl);
        }} catch (err) {{
          return svgMarkup;
        }}
      }}

      function renderNodeDetails(meta) {{
        if (!meta) return;
        const rows = [
          ["Node ID", safe(meta.display_id ?? meta.id)],
          ["Tree Node ID", safe(meta.tree_display_id ?? meta.tree_id)],
          ["Step", meta.step],
          ["Depth", meta.depth],
          ["Parent", meta.parent || "—"],
          ["Children", meta.num_children],
          ["Visits", meta.visits],
          ["Rule ID", safe(meta.rule_id)],
          ["SMILES", safe(meta.smiles)],
          ["Status", meta.status],
        ];
        kvEl.innerHTML = rows
          .map(([k, v]) => `<div>${{k}}</div><div>${{safe(v)}}</div>`)
          .join("");

        if (meta.svg) {{
          molSvgEl.innerHTML = ensureWhiteSvgBackground(meta.svg);
        }} else {{
          molSvgEl.innerHTML = "";
        }}
      }}

      // Initialize at step 0 and focus the root if present.
      setStep(0);
    </script>
  </body>
</html>
"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize how a SynPlanner MCTS tree expanded over time."
    )
    parser.add_argument(
        "--tree",
        type=Path,
        default=Path("tree.pkl"),
        help="Path to a pickled Tree object (default: tree.pkl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("expansion_tree.html"),
        help="Output HTML path (default: expansion_tree.html).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Render only the first N nodes by node_id (useful for large trees).",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Group node creation into steps of this size (default: 1).",
    )
    parser.add_argument(
        "--with-svg",
        dest="with_svg",
        action="store_true",
        help="Depict molecule SVGs for node details (default: enabled).",
    )
    parser.add_argument(
        "--no-svg",
        dest="with_svg",
        action="store_false",
        help="Disable molecule SVGs for node details.",
    )
    parser.set_defaults(with_svg=True)
    parser.add_argument(
        "--radius-step",
        type=float,
        default=20.0,
        help="Base radial step between depths (default: 1.0).",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=120.0,
        help="Scale factor from layout units to SVG units (default: 220).",
    )
    parser.add_argument(
        "--bubble-scale",
        type=float,
        default=10.0,
        help="Multiply node bubble radius by this factor (default: 10).",
    )
    parser.add_argument(
        "--node-radius",
        type=float,
        default=None,
        help="Override automatic node radius in layout units.",
    )

    args = parser.parse_args()

    tree = base_vis._load_tree(args.tree)
    generate_expansion_html(
        tree,
        args.output,
        max_nodes=args.max_nodes,
        sample_step=args.sample_step,
        with_svg=args.with_svg,
        radius_step=args.radius_step,
        render_scale=args.render_scale,
        node_radius=args.node_radius,
        bubble_scale=args.bubble_scale,
    )
    print(f"Wrote expansion timeline HTML to: {args.output}")


if __name__ == "__main__":
    main()
