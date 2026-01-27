#!/usr/bin/env python3
"""
Simple HTML visualization for a SynPlanner MCTS tree.

python visualize_tree.py --tree model_tree.pkl --out tree.html --clusters model_clusters.pkl
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from synplan.mcts.tree import Tree
from synplan.chem.utils import mol_from_smiles
from synplan.utils.visualisation import get_route_svg


def _node_primary_molecule(node) -> Optional[object]:
    if getattr(node, "curr_precursor", None) and hasattr(node.curr_precursor, "molecule"):
        return node.curr_precursor.molecule
    if getattr(node, "new_precursors", None):
        precursor = node.new_precursors[0]
        if hasattr(precursor, "molecule"):
            return precursor.molecule
    if getattr(node, "precursors_to_expand", None):
        precursor = node.precursors_to_expand[0]
        if hasattr(precursor, "molecule"):
            return precursor.molecule
    return None


def _depict_molecule_svg(molecule) -> Optional[str]:
    if molecule is None:
        return None
    try:
        molecule.clean2d()
        return molecule.depict()
    except Exception:
        return None


def _svg_from_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    try:
        molecule = mol_from_smiles(smiles)
    except Exception:
        return None
    return _depict_molecule_svg(molecule)


def _build_target_svg(tree: Tree) -> str:
    target_node = tree.nodes.get(1)
    if not target_node or not getattr(target_node, "curr_precursor", None):
        return ""
    molecule = target_node.curr_precursor.molecule
    try:
        molecule.clean2d()
    except Exception:
        pass

    plane = getattr(molecule, "_plane", None)
    if not plane:
        return ""

    xs = [coord[0] for coord in plane.values()]
    ys = [coord[1] for coord in plane.values()]
    if not xs or not ys:
        return ""

    pad = 0.7
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad

    min_y_svg = -max_y
    max_y_svg = -min_y
    width = max_x - min_x
    height = max_y_svg - min_y_svg

    bond_lines = []
    bond_offset = 0.18
    for a, b, _bond in molecule.bonds():
        if a not in plane or b not in plane:
            continue
        x1, y1 = plane[a]
        x2, y2 = plane[b]
        y1 = -y1
        y2 = -y2
        dx = x2 - x1
        dy = y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5
        if norm == 0:
            continue
        perp_x = -dy / norm
        perp_y = dx / norm
        bond_id = f"{a}-{b}" if a < b else f"{b}-{a}"
        order = getattr(_bond, "order", 1)
        if order == 2:
            offsets = [-bond_offset, bond_offset]
            bond_class = "target-bond bond-double"
        elif order == 3:
            offsets = [-bond_offset, 0.0, bond_offset]
            bond_class = "target-bond bond-triple"
        elif order == 4:
            offsets = [0.0]
            bond_class = "target-bond bond-aromatic"
        else:
            offsets = [0.0]
            bond_class = "target-bond bond-single"

        for offset in offsets:
            ox = perp_x * offset
            oy = perp_y * offset
            bond_lines.append(
                f'<line class="{bond_class}" data-atom1="{a}" data-atom2="{b}" '
                f'data-bond="{bond_id}" x1="{x1 + ox:.2f}" y1="{y1 + oy:.2f}" '
                f'x2="{x2 + ox:.2f}" y2="{y2 + oy:.2f}" />'
            )

    atom_marks = []
    atom_radius = 0.14
    label_size = 0.5
    atom_colors = {
        "N": "#2f6fd0",
        "O": "#e14b4b",
        "S": "#d3b338",
        "F": "#2aa84a",
        "Cl": "#2aa84a",
        "Br": "#2aa84a",
        "I": "#2aa84a",
    }
    for atom_id, atom in molecule.atoms():
        if atom_id not in plane:
            continue
        x, y = plane[atom_id]
        y = -y
        symbol = getattr(atom, "atomic_symbol", "C")
        fill = atom_colors.get(symbol, "#1f242a55")
        if symbol == "C":
            atom_marks.append(
                f'<circle class="target-atom" cx="{x:.2f}" cy="{y:.2f}" r="{atom_radius:.2f}" fill="{fill}" />'
            )
        else:
            mask_radius = atom_radius * 1.9
            atom_marks.append(
                f'<circle class="target-atom-mask" cx="{x:.2f}" cy="{y:.2f}" r="{mask_radius:.2f}" />'
            )
            atom_marks.append(
                f'<text class="target-atom-label" x="{x:.2f}" y="{y:.2f}" '
                f'font-size="{label_size:.2f}" fill="{fill}" text-anchor="middle" dominant-baseline="central">{symbol}</text>'
            )

    return f"""
    <svg id="target-svg" viewBox="{min_x:.2f} {min_y_svg:.2f} {width:.2f} {height:.2f}" preserveAspectRatio="xMidYMid meet">
      <g class="target-bonds">
        {"".join(bond_lines)}
      </g>
      <g class="target-atoms">
        {"".join(atom_marks)}
      </g>
    </svg>
    """


def _ends_with_pickle_stop(path: Path) -> bool:
    size = path.stat().st_size
    if size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        return handle.read(1) == b"."


def _load_tree(tree_pkl: Path) -> Tree:
    if not tree_pkl.exists():
        raise FileNotFoundError(f"Tree pickle not found: {tree_pkl}")
    if tree_pkl.stat().st_size == 0:
        raise ValueError(f"Tree pickle is empty: {tree_pkl}")
    if not _ends_with_pickle_stop(tree_pkl):
        raise ValueError(
            "Tree pickle appears truncated (missing STOP opcode). "
            "Re-save the tree and try again."
        )

    try:
        with tree_pkl.open("rb") as handle:
            loaded = pickle.load(handle)
    except EOFError as exc:
        raise ValueError(
            "Tree pickle is incomplete or corrupted (unexpected EOF). "
            "Re-save the tree and try again."
        ) from exc
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Tree pickle could not be unpickled. "
            "Re-save the tree and try again."
        ) from exc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency while unpickling. "
            "Run this in the same environment where the tree was saved."
        ) from exc

    if hasattr(loaded, "tree"):
        return loaded.tree
    return loaded


def _load_clusters(clusters_pkl: Optional[Path]) -> Dict[str, dict]:
    if not clusters_pkl:
        return {}
    clusters_pkl = Path(clusters_pkl)
    if not clusters_pkl.exists():
        raise FileNotFoundError(f"Clusters pickle not found: {clusters_pkl}")
    if clusters_pkl.stat().st_size == 0:
        raise ValueError(f"Clusters pickle is empty: {clusters_pkl}")
    try:
        with clusters_pkl.open("rb") as handle:
            loaded = pickle.load(handle)
    except EOFError as exc:
        raise ValueError(
            "Clusters pickle is incomplete or corrupted (unexpected EOF). "
            "Re-save the clusters and try again."
        ) from exc
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Clusters pickle could not be unpickled. "
            "Re-save the clusters and try again."
        ) from exc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency while unpickling clusters. "
            "Run this in the same environment where the clusters were saved."
        ) from exc

    if not isinstance(loaded, dict):
        raise TypeError("Clusters pickle must contain a dict.")
    return loaded


def _route_nodes_by_route(
    tree: Tree, route_ids: Optional[Iterable[int]] = None
) -> Dict[str, List[int]]:
    if route_ids is None:
        route_ids_set = set(tree.winning_nodes)
    else:
        route_ids_set = {int(rid) for rid in route_ids}
    route_nodes: Dict[str, List[int]] = {}
    for route_id in sorted(route_ids_set):
        if route_id not in tree.nodes:
            continue
        nodes: List[int] = []
        current = route_id
        seen: Set[int] = set()
        while current and current not in seen:
            seen.add(current)
            nodes.append(current)
            current = tree.parents.get(current)
        route_nodes[str(route_id)] = nodes
    return route_nodes


def _is_building_block(precursor, tree: Tree) -> bool:
    if precursor is None:
        return False
    try:
        return precursor.is_building_block(
            tree.building_blocks, getattr(tree.config, "min_mol_size", 0)
        )
    except Exception:
        try:
            return str(precursor) in tree.building_blocks
        except Exception:
            return False


def _route_extras_by_route(
    tree: Tree, route_ids: Iterable[int]
) -> Dict[str, Dict[str, object]]:
    extras: Dict[str, Dict[str, object]] = {}
    for route_id in sorted(route_ids):
        if route_id not in tree.nodes:
            continue
        path_ids: List[int] = []
        current = route_id
        seen: Set[int] = set()
        while current and current not in seen:
            seen.add(current)
            path_ids.append(current)
            current = tree.parents.get(current)
        path_ids = list(reversed(path_ids))
        if len(path_ids) < 2:
            continue
        smiles_to_node: Dict[str, int] = {}
        base_smiles: Set[str] = set()
        for node_id in path_ids:
            node = tree.nodes.get(node_id)
            molecule = _node_primary_molecule(node)
            if molecule is None:
                continue
            try:
                smiles = str(molecule)
            except Exception:
                continue
            if smiles:
                base_smiles.add(smiles)
                if smiles not in smiles_to_node:
                    smiles_to_node[smiles] = node_id

        by_parent: Dict[str, List[Dict[str, str]]] = {}
        route_seen_smiles: Set[str] = set()
        for before_id, after_id in zip(path_ids, path_ids[1:]):
            before = tree.nodes.get(before_id)
            after = tree.nodes.get(after_id)
            if before is None or after is None:
                continue
            parent_precursor = getattr(before, "curr_precursor", None)
            if parent_precursor is None:
                continue
            try:
                parent_smiles = str(parent_precursor)
            except Exception:
                continue
            if not parent_smiles:
                continue
            extra_items: List[Dict[str, str]] = []
            seen_smiles: Set[str] = set()
            for precursor in getattr(after, "new_precursors", ()) or ():
                try:
                    child_smiles = str(precursor)
                except Exception:
                    continue
                if (
                    not child_smiles
                    or child_smiles in smiles_to_node
                    or child_smiles in base_smiles
                    or child_smiles in route_seen_smiles
                ):
                    continue
                if child_smiles in seen_smiles:
                    continue
                seen_smiles.add(child_smiles)
                route_seen_smiles.add(child_smiles)
                status = "starting material" if _is_building_block(precursor, tree) else "intermediate"
                extra_items.append(
                    {
                        "smiles": child_smiles,
                        "status": status,
                        "svg": _svg_from_smiles(child_smiles),
                    }
                )
            if extra_items:
                by_parent[str(before_id)] = extra_items
        if by_parent:
            extras[str(route_id)] = {"by_parent": by_parent}
    return extras


def _normalize_strat_bonds(
    strat_bonds: Optional[Iterable[Iterable[int]]],
) -> List[List[int]]:
    if not strat_bonds:
        return []
    normalized: List[List[int]] = []
    seen: Set[Tuple[int, int]] = set()
    for bond in strat_bonds:
        if not bond or len(bond) < 2:
            continue
        try:
            a, b = bond
        except (TypeError, ValueError):
            continue
        try:
            a_int = int(a)
            b_int = int(b)
        except (TypeError, ValueError):
            continue
        pair = tuple(sorted((a_int, b_int)))
        if pair in seen:
            continue
        seen.add(pair)
        normalized.append([pair[0], pair[1]])
    normalized.sort()
    return normalized


def _build_cluster_payload(clusters: Dict[str, dict]) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for cluster_id, data in clusters.items():
        if not isinstance(data, dict):
            continue
        bonds = _normalize_strat_bonds(data.get("strat_bonds"))
        route_ids_raw = data.get("route_ids") or []
        route_ids: List[int] = []
        for rid in route_ids_raw:
            try:
                route_ids.append(int(rid))
            except (TypeError, ValueError):
                continue
        payload.append(
            {
                "id": str(cluster_id),
                "bonds": bonds,
                "route_ids": sorted(set(route_ids)),
            }
        )
    return payload




def _group_nodes_by_depth(nodes_depth: Dict[int, int]) -> Dict[int, list]:
    by_depth: Dict[int, list] = {}
    for node_id, depth in nodes_depth.items():
        by_depth.setdefault(depth, []).append(node_id)
    for node_ids in by_depth.values():
        node_ids.sort()
    return by_depth


def _build_children_map(
    tree: Tree, allowed_nodes: Optional[Set[int]] = None
) -> Dict[int, List[int]]:
    if allowed_nodes is None:
        allowed_nodes = set(tree.nodes.keys())
    else:
        allowed_nodes = set(allowed_nodes)
    allowed_nodes.add(1)

    children_map: Dict[int, List[int]] = {node_id: [] for node_id in allowed_nodes}
    for child_id, parent_id in tree.parents.items():
        if child_id == 1 or not parent_id:
            continue
        if parent_id not in allowed_nodes or child_id not in allowed_nodes:
            continue
        children_map[parent_id].append(child_id)
    for node_id, children in children_map.items():
        children.sort()
    return children_map


def _sorted_children(children_map: Dict[int, List[int]], node_id: int) -> List[int]:
    return children_map.get(node_id, [])


def _compute_depths(
    children_map: Dict[int, List[int]], root_id: int = 1
) -> Dict[int, int]:
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


def _compute_subtree_leaf_counts(
    children_map: Dict[int, List[int]], root_id: int = 1
) -> Dict[int, int]:
    order: List[int] = []
    if root_id not in children_map:
        return {}
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        order.append(node_id)
        stack.extend(_sorted_children(children_map, node_id))

    leaf_counts: Dict[int, int] = {}
    for node_id in reversed(order):
        children = _sorted_children(children_map, node_id)
        if not children:
            leaf_counts[node_id] = 1
        else:
            leaf_counts[node_id] = sum(leaf_counts[c] for c in children)
    return leaf_counts


def _assign_subtree_angles(
    children_map: Dict[int, List[int]],
    leaf_counts: Dict[int, int],
    root_id: int = 1,
    base_gap: float = 0.04,
) -> Dict[int, float]:
    angles: Dict[int, float] = {}
    if root_id not in children_map:
        return angles
    stack = [(root_id, 0.0, 2.0 * math.pi)]
    while stack:
        node_id, start_angle, end_angle = stack.pop()
        angles[node_id] = (start_angle + end_angle) / 2.0
        children = _sorted_children(children_map, node_id)
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


def _compute_radius_scale(
    by_depth: Dict[int, list],
    angles: Dict[int, float],
    radius_step: float,
    node_radius: float,
    spacing_factor: float = 2.2,
    root_gap_factor: float = 2.8,
) -> float:
    min_distance = node_radius * spacing_factor
    scale = 1.0
    epsilon = 1e-6
    scale = max(scale, min_distance / max(radius_step, epsilon))
    root_gap = node_radius * root_gap_factor
    scale = max(scale, root_gap / max(radius_step, epsilon))
    for depth, node_ids in by_depth.items():
        if depth == 0 or len(node_ids) < 2:
            continue
        radius = depth * radius_step
        depth_angles = sorted(angles.get(node_id, 0.0) for node_id in node_ids)
        deltas = []
        for left, right in zip(depth_angles, depth_angles[1:]):
            deltas.append(right - left)
        deltas.append(2.0 * math.pi - depth_angles[-1] + depth_angles[0])
        min_delta = max(min(deltas), epsilon)
        required = min_distance / (radius * min_delta)
        scale = max(scale, required)
    return max(scale, 1.0)


def _radial_layout(
    nodes_depth: Dict[int, int],
    children_map: Dict[int, List[int]],
    radius_step: float,
    node_radius: float,
    spacing_factor: float = 2.2,
    root_gap_factor: float = 2.8,
) -> Dict[int, Tuple[float, float]]:
    if not nodes_depth:
        return {}
    by_depth = _group_nodes_by_depth(nodes_depth)
    leaf_counts = _compute_subtree_leaf_counts(children_map)
    angles = _assign_subtree_angles(children_map, leaf_counts)
    scale = _compute_radius_scale(
        by_depth,
        angles,
        radius_step,
        node_radius,
        spacing_factor=spacing_factor,
        root_gap_factor=root_gap_factor,
    )
    radius_step *= scale

    positions: Dict[int, Tuple[float, float]] = {}
    for node_id, depth in nodes_depth.items():
        if depth == 0:
            positions[node_id] = (0.0, 0.0)
            continue
        angle = angles.get(node_id, 0.0)
        radius = depth * radius_step
        positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))
    return positions


def _node_status(tree: Tree, node_id: int) -> str:
    if node_id == 1:
        return "target"
    node = tree.nodes[node_id]
    if node.is_solved():
        return "starting material"
    return "intermediate"


def _node_metadata(
    tree: Tree, node_id: int, route_index: Dict[int, int]
) -> Dict[str, object]:
    node = tree.nodes[node_id]
    molecule = _node_primary_molecule(node)
    smiles = str(molecule) if molecule is not None else None
    return {
        "node_id": node_id,
        "route_id": node_id if node_id in route_index else None,
        "route_index": route_index.get(node_id),
        "depth": tree.nodes_depth.get(node_id, 0),
        "visits": tree.nodes_visit.get(node_id, 0),
        "num_children": len(tree.children.get(node_id, [])),
        "rule_id": tree.nodes_rules.get(node_id),
        "rule_label": tree.nodes_rule_label.get(node_id),
        "solved": bool(node.is_solved()),
        "smiles": smiles,
        "svg": _depict_molecule_svg(molecule),
        "pending_smiles": (
            [str(p) for p in node.precursors_to_expand]
            if node.precursors_to_expand
            else []
        ),
        "new_smiles": (
            [str(p) for p in node.new_precursors] if node.new_precursors else []
        ),
    }


def _edges_from_tree(
    tree: Tree, allowed_nodes: Optional[Set[int]] = None
) -> Iterable[Tuple[int, int]]:
    allowed = set(allowed_nodes) if allowed_nodes is not None else None
    for child_id, parent_id in tree.parents.items():
        if child_id == 1 or not parent_id:
            continue
        if allowed is not None and (child_id not in allowed or parent_id not in allowed):
            continue
        yield parent_id, child_id


def _winning_route_edges(tree: Tree) -> set:
    edges = set()
    for node_id in tree.winning_nodes:
        current = node_id
        while current and current in tree.parents:
            parent = tree.parents.get(current)
            if not parent:
                break
            edges.add((parent, current))
            current = parent
    return edges


def _scale_positions(
    positions: Dict[int, Tuple[float, float]],
    node_radius: float,
    render_scale: float,
) -> Tuple[Dict[int, Tuple[float, float]], float]:
    render_scale = max(render_scale, 0.01)
    scaled_positions = {
        node_id: (pos[0] * render_scale, pos[1] * render_scale)
        for node_id, pos in positions.items()
    }
    return scaled_positions, node_radius * render_scale


def _render_svg(
    tree: Tree,
    positions: Dict[int, Tuple[float, float]],
    edges: Iterable[Tuple[int, int]],
    winning_edges: Set[Tuple[int, int]],
    node_radius: float,
    nodes_depth: Dict[int, int],
    radius_step: Optional[float] = None,
    pad_scale: float = 4.0,
) -> str:
    if not positions:
        return ""

    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    max_radius = node_radius * 2.0
    pad = max_radius * pad_scale
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    view_w = max_x - min_x
    view_h = max_y - min_y

    line_items = []
    for parent_id, child_id in edges:
        if parent_id not in positions or child_id not in positions:
            continue
        x1, y1 = positions[parent_id]
        x2, y2 = positions[child_id]
        line_class = "edge-winning" if (parent_id, child_id) in winning_edges else ""
        line_items.append(
            f'<line class="{line_class}" data-parent-id="{parent_id}" data-child-id="{child_id}" '
            f'x1="{_format_point(x1)}" y1="{_format_point(y1)}" '
            f'x2="{_format_point(x2)}" y2="{_format_point(y2)}" />'
        )

    circle_items = []
    for node_id, (x, y) in positions.items():
        status = _node_status(tree, node_id)
        delay = min(nodes_depth.get(node_id, 0) * 0.03, 0.6)
        radius = node_radius * 2.0 if node_id == 1 else node_radius
        parent_id = tree.parents.get(node_id, 0)
        circle_items.append(
            f'<circle class="node node-{status}" '
            f'cx="{_format_point(x)}" cy="{_format_point(y)}" '
            f'r="{_format_point(radius)}" '
            f'data-node-id="{node_id}" '
            f'data-parent-id="{parent_id}" '
            f'data-node-status="{status}" '
            f'style="animation-delay: {delay:.2f}s;" />'
        )

    depth_circles = []
    if radius_step is not None:
        max_depth = max(nodes_depth.values()) if nodes_depth else 0
        for depth in range(1, max_depth + 1):
            depth_circles.append(
                f'<circle class="depth-ring" cx="0" cy="0" r="{(depth * radius_step):.2f}" />'
            )

    return f"""
    <svg class="tree-svg" viewBox="{_format_point(min_x)} {_format_point(min_y)} {_format_point(view_w)} {_format_point(view_h)}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" role="img">
      <g class="depth-rings">
        {"".join(depth_circles)}
      </g>
      <g class="edges">
        {"".join(line_items)}
      </g>
      <g class="nodes">
        {"".join(circle_items)}
      </g>
    </svg>
    """


def _format_point(value: float) -> str:
    return f"{value:.2f}"


def generate_tree_html(
    tree: Tree,
    output_path: Path,
    radius_step: float = 280.0,
    node_radius: float = 80.0,
    render_scale: float = 0.25,
    clusters_pkl: Optional[Path] = None,
) -> None:
    full_children = _build_children_map(tree)
    full_nodes_depth = _compute_depths(full_children)
    full_radius_step = radius_step * 0.4
    full_node_radius = node_radius * 10.0
    full_render_scale = min(render_scale * 1.0, 1.0)
    full_positions = _radial_layout(
        full_nodes_depth,
        full_children,
        radius_step=full_radius_step,
        node_radius=full_node_radius,
        spacing_factor=3.4,
        root_gap_factor=4.0,
    )
    full_angles = _assign_subtree_angles(
        full_children, _compute_subtree_leaf_counts(full_children)
    )
    full_layout_scale = _compute_radius_scale(
        _group_nodes_by_depth(full_nodes_depth),
        full_angles,
        full_radius_step,
        full_node_radius,
        spacing_factor=3.4,
        root_gap_factor=4.0,
    )
    full_edges = list(_edges_from_tree(tree))
    winning_edges = _winning_route_edges(tree)

    if not full_positions:
        raise ValueError("Tree has no nodes to render.")

    if winning_edges:
        solved_nodes: Set[int] = {1}
        for parent_id, child_id in winning_edges:
            solved_nodes.add(parent_id)
            solved_nodes.add(child_id)
    else:
        solved_nodes = set(tree.nodes.keys())

    solved_children = _build_children_map(tree, allowed_nodes=solved_nodes)
    solved_nodes_depth = _compute_depths(solved_children)
    solved_radius_step = radius_step * 0.5
    solved_node_radius = node_radius * 1.15
    solved_render_scale = min(render_scale * 0.4, 1.0)
    solved_positions = _radial_layout(
        solved_nodes_depth,
        solved_children,
        radius_step=solved_radius_step,
        node_radius=solved_node_radius,
    )
    solved_angles = _assign_subtree_angles(
        solved_children, _compute_subtree_leaf_counts(solved_children)
    )
    solved_layout_scale = _compute_radius_scale(
        _group_nodes_by_depth(solved_nodes_depth),
        solved_angles,
        solved_radius_step,
        solved_node_radius,
    )
    solved_edges = list(_edges_from_tree(tree, allowed_nodes=solved_nodes))

    full_positions, full_render_radius = _scale_positions(
        full_positions, full_node_radius, full_render_scale
    )
    solved_positions, solved_render_radius = _scale_positions(
        solved_positions, solved_node_radius, solved_render_scale
    )
    full_render_step = full_radius_step * full_layout_scale * full_render_scale
    solved_render_step = solved_radius_step * solved_layout_scale * solved_render_scale

    svg_full = _render_svg(
        tree,
        full_positions,
        full_edges,
        winning_edges,
        full_render_radius,
        full_nodes_depth,
        radius_step=full_render_step,
        pad_scale=1.8,
    )
    svg_solved = _render_svg(
        tree,
        solved_positions,
        solved_edges,
        winning_edges,
        solved_render_radius,
        solved_nodes_depth,
        radius_step=solved_render_step,
        pad_scale=1.6,
    )
    if not svg_solved:
        svg_solved = svg_full
        solved_positions = full_positions

    node_meta = {}
    route_index = {node_id: idx for idx, node_id in enumerate(tree.winning_nodes)}
    for node_id in tree.nodes:
        node_meta[str(node_id)] = _node_metadata(tree, node_id, route_index)
    target_svg = _build_target_svg(tree)
    target_smiles = str(tree.nodes[1].curr_precursor) if tree.nodes.get(1) else ""
    clusters_payload: List[Dict[str, object]] = []
    route_nodes: Dict[str, List[int]] = {}
    route_extras: Dict[str, Dict[str, object]] = {}
    route_svgs: Dict[str, str] = {}
    if clusters_pkl:
        clusters = _load_clusters(clusters_pkl)
        clusters_payload = _build_cluster_payload(clusters)
        cluster_route_ids = {
            route_id
            for cluster in clusters_payload
            for route_id in cluster.get("route_ids", [])
        }
        route_nodes = _route_nodes_by_route(tree, cluster_route_ids)
        route_extras = _route_extras_by_route(tree, cluster_route_ids)
        for route_id in sorted(cluster_route_ids):
            if route_id not in tree.winning_nodes:
                continue
            svg = get_route_svg(tree, route_id)
            if svg:
                route_svgs[str(route_id)] = svg

    html = f"""<!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>SynPlanner Tree Visualization</title>
      <style>
        :root {{
          --bg-1: #f4f1eb;
          --bg-2: #e1e8ef;
          --edge: #7d8793;
          --target: #4aa3ff;
          --intermediate: #f2b3a1;
          --starting material: #7bd89c;
          --node-stroke: #2b2b2b;
          --modal-bg: #ffffff;
          --modal-shadow: rgba(17, 17, 17, 0.2);
        }}

        html, body {{
          margin: 0;
          padding: 0;
          width: 100%;
          height: 100%;
          font-family: "Avenir Next", "Futura", "Gill Sans", sans-serif;
          background: linear-gradient(135deg, var(--bg-1), var(--bg-2));
          color: #1b1b1b;
          overflow: hidden;
        }}

        #canvas {{
          width: 100vw;
          height: 100vh;
          overflow: hidden;
          box-sizing: border-box;
          position: relative;
          z-index: 1;
        }}

        .tree-svg {{
          display: block;
          width: 100vw;
          height: 100vh;
        }}

        .depth-rings .depth-ring {{
          fill: none;
          stroke: rgba(80, 80, 80, 0.5);
          stroke-width: 0.8;
          stroke-dasharray: 4 6;
        }}

        #hud {{
          position: fixed;
          inset: 0;
          pointer-events: none;
          z-index: 20;
        }}

        .view-pane {{
          display: none;
        }}

        .view-pane.active {{
          display: block;
        }}

        #mode-controls {{
          position: fixed;
          top: 16px;
          right: 16px;
          z-index: 12;
          display: flex;
          gap: 10px;
          pointer-events: auto;
        }}

        #view-toggle,
        #undo-toggle {{
          padding: 12px 20px;
          border-radius: 999px;
          border: none;
          background: #1f242a;
          color: #f4f1eb;
          font-size: 18px;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          cursor: pointer;
          box-shadow: 0 14px 28px rgba(17, 17, 17, 0.25);
          box-sizing: border-box;
        }}

        #view-toggle.disabled {{
          opacity: 0.5;
          cursor: not-allowed;
        }}

        #undo-toggle.disabled {{
          opacity: 0.5;
          cursor: not-allowed;
        }}

        #top-left-controls {{
          position: fixed;
          top: 16px;
          left: 16px;
          z-index: 12;
          display: flex;
          gap: 10px;
          align-items: flex-start;
          pointer-events: auto;
        }}

        #legend-panel {{
          position: relative;
          background: #ffffff;
          border-radius: 10px;
          border: 1px solid #1f242a;
          box-shadow: 0 10px 20px rgba(17, 17, 17, 0.15);
          padding: 8px 10px;
          font-size: 11px;
          display: grid;
          gap: 6px;
          pointer-events: auto;
        }}

        #legend-panel .legend-title {{
          font-weight: 600;
          font-size: 11px;
          letter-spacing: 0.02em;
        }}

        #legend-panel .legend-row {{
          display: flex;
          align-items: center;
          gap: 8px;
          white-space: nowrap;
        }}

        #legend-panel .legend-dot {{
          width: 10px;
          height: 10px;
          border-radius: 999px;
          border: 1px solid var(--node-stroke);
          flex: 0 0 auto;
        }}

        #legend-panel .legend-dot.target {{
          background: var(--target);
        }}

        #legend-panel .legend-dot.intermediate {{
          background: var(--intermediate);
        }}

        #legend-panel .legend-dot.starting material {{
          background: var(--starting material);
        }}

        #legend-panel .legend-line {{
          width: 16px;
          height: 0;
          border-top: 2px solid var(--edge);
          opacity: 0.8;
          flex: 0 0 auto;
        }}

        #legend-panel .legend-line.solved {{
          border-top-color: #ff7a59;
          opacity: 0.9;
        }}

        #reassemble-toggle {{
          padding: 10px 16px;
          border-radius: 999px;
          border: none;
          background: #1f242a;
          color: #f4f1eb;
          font-size: 14px;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          cursor: pointer;
          box-shadow: 0 12px 24px rgba(17, 17, 17, 0.2);
          box-sizing: border-box;
        }}

        #reset-toggle {{
          padding: 10px 16px;
          border-radius: 999px;
          border: none;
          background: #ffffff;
          color: #1f242a;
          font-size: 14px;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          cursor: pointer;
          box-shadow: 0 12px 24px rgba(17, 17, 17, 0.2);
          box-sizing: border-box;
        }}

        #route-panel {{
          position: fixed;
          top: 68px;
          right: 16px;
          width: 180px;
          height: 260px;
          max-height: 260px;
          background: #ffffff;
          border-radius: 12px;
          border: 2px solid #1f242a;
          box-shadow: 0 14px 28px rgba(17, 17, 17, 0.18);
          z-index: 11;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          pointer-events: auto;
        }}

        #route-panel .panel-header {{
          padding: 8px 10px;
          background: #f4f1eb;
          font-size: 12px;
          font-weight: 600;
        }}

        #route-panel .panel-body {{
          padding: 8px 10px;
          background: #ffffff;
          flex: 1;
          min-height: 0;
          display: flex;
          flex-direction: column;
        }}

        #route-list {{
          --route-btn-height: 22px;
          --route-btn-gap: 6px;
          display: grid;
          gap: var(--route-btn-gap);
          flex: 1;
          min-height: 0;
          overflow-y: auto;
        }}

        .route-btn {{
          height: var(--route-btn-height);
          border-radius: 8px;
          border: 1px solid #1f242a;
          background: #f6f5f2;
          font-size: 12px;
          cursor: pointer;
          text-align: center;
        }}

        .route-btn.active {{
          background: #4aa3ff;
          color: #ffffff;
          border-color: #4aa3ff;
        }}

        #route-detail {{
          position: fixed;
          right: 16px;
          bottom: 16px;
          width: 240px;
          height: 220px;
          min-width: 180px;
          min-height: 140px;
          background: #ffffff;
          border-radius: 12px;
          border: 2px solid #1f242a;
          box-shadow: 0 14px 28px rgba(17, 17, 17, 0.18);
          z-index: 11;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          pointer-events: auto;
          box-sizing: border-box;
        }}

        #route-detail.hidden {{
          display: none;
        }}

        #route-detail.collapsed {{
          height: auto;
          min-height: 0;
        }}

        #route-detail-resize {{
          position: absolute;
          top: 0;
          left: 0;
          width: 18px;
          height: 18px;
          cursor: nwse-resize;
          touch-action: none;
          z-index: 2;
          background: linear-gradient(
            135deg,
            rgba(31, 36, 42, 0.0) 40%,
            rgba(31, 36, 42, 0.35) 40% 45%,
            rgba(31, 36, 42, 0.0) 45% 60%,
            rgba(31, 36, 42, 0.35) 60% 65%,
            rgba(31, 36, 42, 0.0) 65%
          );
        }}

        #route-detail.collapsed .panel-body {{
          display: none;
        }}

        #route-detail .panel-header {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 8px 10px;
          background: #f4f1eb;
          font-size: 12px;
          font-weight: 600;
        }}

        #route-detail .panel-body {{
          padding: 8px 10px;
          background: #ffffff;
          overflow: auto;
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
        }}

        #route-detail svg {{
          max-width: 100%;
          max-height: 100%;
          width: auto;
          height: auto;
          display: block;
        }}

        #route-detail-toggle {{
          border: none;
          background: #1f242a;
          color: #f4f1eb;
          border-radius: 999px;
          padding: 4px 8px;
          font-size: 11px;
          cursor: pointer;
        }}

        .edges line {{
          stroke: var(--edge);
          stroke-width: 1.2;
          opacity: 0.6;
          vector-effect: non-scaling-stroke;
        }}

        .edges line.edge-winning {{
          stroke: #ff7a59;
          stroke-width: 2.4;
          opacity: 0.9;
        }}

        .edges line.edge-route {{
          stroke: #4aa3ff;
          stroke-width: 2.6;
          opacity: 0.95;
        }}

        .edges line.edge-invalid {{
          display: none;
        }}

        .node {{
          cursor: pointer;
          stroke: var(--node-stroke);
          stroke-width: 0.8;
          opacity: 0.95;
          transform-box: fill-box;
          transform-origin: center;
          animation: popIn 0.35s ease-out both;
        }}

        .node-target {{ fill: var(--target); }}
        .node-intermediate {{ fill: var(--intermediate); }}
        .node-starting material {{ fill: var(--starting material); }}

        .node:hover {{
          stroke-width: 2;
        }}

        .node.active {{
          stroke-width: 2.4;
        }}

        #modal-backdrop {{
          position: fixed;
          inset: 0;
          display: none;
          align-items: center;
          justify-content: center;
          background: rgba(15, 20, 25, 0.35);
          backdrop-filter: blur(2px);
        }}

        #modal-backdrop.show {{
          display: flex;
        }}

        #modal {{
          width: 180px;
          min-height: 120px;
          background: var(--modal-bg);
          border-radius: 14px;
          box-shadow: 0 20px 40px var(--modal-shadow);
          padding: 12px;
          position: fixed;
          left: 16px;
          top: 16px;
        }}

        #modal-body {{
          min-height: 140px;
        }}

        #modal-body .mol-frame {{
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 6px;
          border-radius: 12px;
          background: #f6f5f2;
        }}

        #modal-body .meta-table {{
          width: 100%;
          border-collapse: collapse;
          margin-top: 12px;
          font-size: 12px;
        }}

        #modal-body .meta-table th,
        #modal-body .meta-table td {{
          text-align: left;
          padding: 6px 8px;
          border-bottom: 1px solid #d7d2cb;
        }}

        #modal-body .meta-table th {{
          color: #5a5a5a;
          font-weight: 600;
        }}

        #modal-body svg {{
          width: 85%;
          height: auto;
          max-height: 140px;
        }}

        #target-panel {{
          position: fixed;
          left: 16px;
          bottom: 16px;
          width: 360px;
          max-height: 320px;
          background: #ffffff;
          border-radius: 14px;
          border: 2px solid #1f242a;
          box-shadow: 0 18px 36px rgba(17, 17, 17, 0.22);
          z-index: 11;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          pointer-events: auto;
        }}

        #target-panel.collapsed .panel-body {{
          display: none;
        }}

        #target-panel .panel-header {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 14px;
          background: #f4f1eb;
          font-size: 14px;
          font-weight: 600;
        }}

        #target-panel .panel-body {{
          padding: 10px;
          background: #ffffff;
          overflow: auto;
        }}

        #target-selection {{
          margin-top: 8px;
          font-size: 12px;
          color: #3a3a3a;
          word-break: break-word;
        }}

        #cluster-row {{
          margin-top: 6px;
          font-size: 12px;
          color: #3a3a3a;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
        }}

        #cluster-status {{
          flex: 1;
        }}

        #cluster-exact-label {{
          display: inline-flex;
          align-items: center;
          gap: 4px;
          font-size: 12px;
        }}

        #cluster-exact {{
          accent-color: #1f242a;
        }}

        #target-selection code {{
          display: inline-block;
          background: #f4f1eb;
          padding: 4px 6px;
          border-radius: 6px;
        }}

        #target-panel .panel-body svg {{
          width: 100%;
          height: auto;
          display: block;
        }}

        #target-toggle {{
          border: none;
          background: #1f242a;
          color: #f4f1eb;
          border-radius: 999px;
          padding: 6px 10px;
          font-size: 12px;
          cursor: pointer;
        }}

        .target-bond {{
          stroke: #1f242a55;
          stroke-width: 0.08;
          stroke-linecap: round;
          opacity: 0.9;
          cursor: pointer;
          pointer-events: stroke;
        }}

        .target-bond.bond-double {{
          stroke-width: 0.07;
        }}

        .target-bond.bond-triple {{
          stroke-width: 0.06;
        }}

        .target-bond.bond-aromatic {{
          stroke-dasharray: 0.22 0.14;
        }}

        .target-bond.possible {{
          stroke: #f2d04b;
        }}

        .target-bond.excluded {{
          stroke: #1f242a;
          opacity: 0.7;
        }}

        .target-bond.selected {{
          stroke: #32d16d;
          stroke-width: 0.14;
        }}

        .target-atom {{
          opacity: 0.9;
        }}

        .target-atom-mask {{
          fill: #ffffff;
        }}

        .target-atom-label {{
          font-family: "Avenir Next", "Futura", "Gill Sans", sans-serif;
        }}

        .filtered-out {{
          display: none;
        }}

        .route-hidden {{
          display: none;
        }}

        .zoom-rect {{
          fill: rgba(74, 163, 255, 0.18);
          stroke: #4aa3ff;
          stroke-width: 1.2;
          stroke-dasharray: 6 6;
          vector-effect: non-scaling-stroke;
          pointer-events: none;
        }}

        @keyframes popIn {{
          from {{
            opacity: 0;
            transform: scale(0.7);
          }}
          to {{
            opacity: 1;
            transform: scale(1);
          }}
        }}
      </style>
    </head>
    <body>
        <div id="canvas">
        <div id="view-full" class="view-pane">
          {svg_full}
        </div>
        <div id="view-solved" class="view-pane active">
          {svg_solved}
        </div>
      </div>
        <div id="hud">
        <div id="top-left-controls">
          <button id="reassemble-toggle" type="button">Reassemble</button>
          <button id="reset-toggle" type="button">Reset</button>
          <div id="legend-panel">
            <div class="legend-title">Legend</div>
            <div class="legend-row">
              <span class="legend-dot target"></span>
              <span>Target molecule</span>
            </div>
            <div class="legend-row">
              <span class="legend-dot intermediate"></span>
              <span>Intermediate product</span>
            </div>
            <div class="legend-row">
              <span class="legend-dot starting material"></span>
              <span>Building block</span>
            </div>
            <div class="legend-row">
              <span class="legend-line solved"></span>
              <span>Solved pathway</span>
            </div>
            <div class="legend-row">
              <span class="legend-line"></span>
              <span>Unsolved pathway</span>
            </div>
          </div>
        </div>
        <div id="mode-controls">
          <button id="undo-toggle" type="button">Undo</button>
          <button id="view-toggle" type="button">Full</button>
        </div>
        <div id="route-panel">
          <div class="panel-header" id="route-header">
            Route ID: Total {len(tree.winning_nodes)} Current 0
          </div>
          <div class="panel-body">
            <div id="route-list"></div>
          </div>
        </div>
        <div id="route-detail" class="hidden collapsed">
          <div id="route-detail-resize" aria-hidden="true"></div>
          <div class="panel-header">
            <span id="route-detail-title">Route</span>
            <button id="route-detail-toggle" type="button">Show</button>
          </div>
          <div class="panel-body" id="route-detail-body"></div>
        </div>
        <div id="target-panel">
          <div class="panel-header">
            <span>Target molecule</span>
            <button id="target-toggle" type="button">Hide</button>
          </div>
          <div class="panel-body">
            {target_svg if target_svg else "<div>Target depiction unavailable.</div>"}
            <div id="target-selection">
              Selected bonds (CGRtools atom ids): <code>[]</code>
            </div>
            <div id="cluster-row">
              <span id="cluster-status">Clusters: not loaded</span>
              <label id="cluster-exact-label">
                <input id="cluster-exact" type="checkbox" />
                Exact
              </label>
            </div>
          </div>
        </div>
      </div>
      <div id="modal-backdrop">
        <div id="modal">
          <div id="modal-body"></div>
        </div>
      </div>
      <script>
        const nodeMeta = {json.dumps(node_meta, ensure_ascii=True)};
        const modalBackdrop = document.getElementById("modal-backdrop");
        const modal = document.getElementById("modal");
        const modalBody = document.getElementById("modal-body");
        let nodes = [];
        const canvas = document.getElementById("canvas");
        const reassembleToggle = document.getElementById("reassemble-toggle");
        const resetToggle = document.getElementById("reset-toggle");
        const viewToggle = document.getElementById("view-toggle");
        const undoToggle = document.getElementById("undo-toggle");
        const viewFull = document.getElementById("view-full");
        const viewSolved = document.getElementById("view-solved");
        const routePanel = document.getElementById("route-panel");
        const routeHeader = document.getElementById("route-header");
        const routeList = document.getElementById("route-list");
        const routeDetail = document.getElementById("route-detail");
        const routeDetailBody = document.getElementById("route-detail-body");
        const routeDetailTitle = document.getElementById("route-detail-title");
        const routeDetailToggle = document.getElementById("route-detail-toggle");
        const routeDetailResize = document.getElementById("route-detail-resize");
        const targetPanel = document.getElementById("target-panel");
        const targetToggle = document.getElementById("target-toggle");
        const targetSvg = document.getElementById("target-svg");
        const targetSelection = document.getElementById("target-selection");
        const clusterStatus = document.getElementById("cluster-status");
        const clusterExact = document.getElementById("cluster-exact");
        const hasSolvedRoutes = {json.dumps(bool(winning_edges))};
        let viewMode = "solved";
        const totalRoutes = {len(tree.winning_nodes)};
        let activeRouteId = null;
        let focusedRouteId = null;
        const selectedBondIds = new Set();
        const targetSmiles = {json.dumps(target_smiles, ensure_ascii=True)};
        const clusters = {json.dumps(clusters_payload, ensure_ascii=True)};
        const routeNodes = {json.dumps(route_nodes, ensure_ascii=True)};
        const routeExtras = {json.dumps(route_extras, ensure_ascii=True)};
        const routeSvgs = {json.dumps(route_svgs, ensure_ascii=True)};
        const clustersLoaded = clusters.length > 0;
        const extraMeta = {{}};
        const zoomState = {{
          active: false,
          svg: null,
          rect: null,
          start: null,
        }};
        const undoStack = [];
        const maxUndo = 20;
        let isRestoring = false;
        let currentRouteIds = [];
        const originalFullHtml = viewFull ? viewFull.innerHTML : "";
        const originalSolvedHtml = viewSolved ? viewSolved.innerHTML : "";

        clusters.forEach((cluster) => {{
          const bonds = Array.isArray(cluster.bonds) ? cluster.bonds : [];
          cluster.bondKeys = bonds.map((pair) => {{
            const left = Math.min(pair[0], pair[1]);
            const right = Math.max(pair[0], pair[1]);
            return `${{left}}-${{right}}`;
          }});
        }});

        const allClusterBondKeys = new Set();
        clusters.forEach((cluster) => {{
          (cluster.bondKeys || []).forEach((key) => allClusterBondKeys.add(key));
        }});

        function centerCanvas() {{
          canvas.scrollLeft = Math.max(0, (canvas.scrollWidth - canvas.clientWidth) / 2);
          canvas.scrollTop = Math.max(0, (canvas.scrollHeight - canvas.clientHeight) / 2);
        }}

        function clientToSvg(svg, clientX, clientY) {{
          const rect = svg.getBoundingClientRect();
          const viewBox = svg.viewBox.baseVal;
          const scaleX = rect.width / viewBox.width;
          const scaleY = rect.height / viewBox.height;
          const scale = Math.min(scaleX, scaleY) || 1;
          const renderedWidth = viewBox.width * scale;
          const renderedHeight = viewBox.height * scale;
          const offsetX = rect.left + (rect.width - renderedWidth) / 2;
          const offsetY = rect.top + (rect.height - renderedHeight) / 2;
          let x = viewBox.x + (clientX - offsetX) / scale;
          let y = viewBox.y + (clientY - offsetY) / scale;
          x = Math.max(viewBox.x, Math.min(viewBox.x + viewBox.width, x));
          y = Math.max(viewBox.y, Math.min(viewBox.y + viewBox.height, y));
          return {{ x, y }};
        }}

        function ensureZoomRect(svg) {{
          let rect = svg.querySelector(".zoom-rect");
          if (!rect) {{
            rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            rect.classList.add("zoom-rect");
            svg.appendChild(rect);
          }}
          return rect;
        }}

        function bindFullZoom() {{
          if (!viewFull) {{
            return;
          }}
          const svg = viewFull.querySelector("svg.tree-svg");
          if (!svg || svg.dataset.zoomBound === "true") {{
            return;
          }}
          svg.dataset.zoomBound = "true";

          svg.addEventListener("pointerdown", (event) => {{
            if (viewMode !== "full") {{
              return;
            }}
            if (event.button !== 0) {{
              return;
            }}
            if (event.target.closest && event.target.closest("circle.node")) {{
              return;
            }}
            event.preventDefault();
            zoomState.active = true;
            zoomState.svg = svg;
            zoomState.start = clientToSvg(svg, event.clientX, event.clientY);
            zoomState.rect = ensureZoomRect(svg);
            zoomState.rect.setAttribute("x", zoomState.start.x.toFixed(2));
            zoomState.rect.setAttribute("y", zoomState.start.y.toFixed(2));
            zoomState.rect.setAttribute("width", "0");
            zoomState.rect.setAttribute("height", "0");
          }});
        }}

        function updateZoomRect(event) {{
          if (!zoomState.active || !zoomState.svg || !zoomState.start) {{
            return;
          }}
          const pos = clientToSvg(zoomState.svg, event.clientX, event.clientY);
          const x = Math.min(zoomState.start.x, pos.x);
          const y = Math.min(zoomState.start.y, pos.y);
          const width = Math.abs(pos.x - zoomState.start.x);
          const height = Math.abs(pos.y - zoomState.start.y);
          if (zoomState.rect) {{
            zoomState.rect.setAttribute("x", x.toFixed(2));
            zoomState.rect.setAttribute("y", y.toFixed(2));
            zoomState.rect.setAttribute("width", width.toFixed(2));
            zoomState.rect.setAttribute("height", height.toFixed(2));
          }}
        }}

        function finishZoom(event) {{
          if (!zoomState.active || !zoomState.svg || !zoomState.start) {{
            return;
          }}
          const pos = clientToSvg(zoomState.svg, event.clientX, event.clientY);
          const x = Math.min(zoomState.start.x, pos.x);
          const y = Math.min(zoomState.start.y, pos.y);
          const width = Math.abs(pos.x - zoomState.start.x);
          const height = Math.abs(pos.y - zoomState.start.y);
          const minSize = Math.min(
            zoomState.svg.viewBox.baseVal.width,
            zoomState.svg.viewBox.baseVal.height
          ) * 0.02;
          if (width > minSize && height > minSize) {{
            captureUndoState();
            zoomState.svg.setAttribute(
              "viewBox",
              x.toFixed(2) +
                " " +
                y.toFixed(2) +
                " " +
                width.toFixed(2) +
                " " +
                height.toFixed(2)
            );
          }}
          if (zoomState.rect) {{
            zoomState.rect.remove();
          }}
          zoomState.active = false;
          zoomState.svg = null;
          zoomState.rect = null;
          zoomState.start = null;
        }}

        function setView(mode) {{
          viewMode = mode;
          if (mode === "solved") {{
            viewFull.classList.remove("active");
            viewSolved.classList.add("active");
            viewToggle.textContent = "Full";
          }} else {{
            viewSolved.classList.remove("active");
            viewFull.classList.add("active");
            viewToggle.textContent = "Solved";
          }}
          setTimeout(centerCanvas, 0);
        }}

        window.addEventListener("load", () => {{
          if (hasSolvedRoutes) {{
            setView("solved");
          }} else {{
            setView("full");
            viewToggle.classList.add("disabled");
            viewToggle.disabled = true;
          }}
          initRouteResize();
          bindFullZoom();
          updateUndoButton();
          syncSelectedBonds();
        }});

        if (undoToggle) {{
          undoToggle.addEventListener("click", () => {{
            restoreUndoState();
          }});
        }}

        viewToggle.addEventListener("click", () => {{
          if (!hasSolvedRoutes) {{
            return;
          }}
          captureUndoState();
          setView(viewMode === "full" ? "solved" : "full");
        }});

        if (targetToggle) {{
          targetToggle.addEventListener("click", () => {{
            targetPanel.classList.toggle("collapsed");
            targetToggle.textContent = targetPanel.classList.contains("collapsed")
              ? "Show"
              : "Hide";
            setTimeout(centerCanvas, 0);
          }});
        }}

        if (routeDetailToggle && routeDetail) {{
          routeDetailToggle.addEventListener("click", () => {{
            const collapsed = routeDetail.classList.contains("collapsed");
            setRouteDetailCollapsed(!collapsed);
          }});
        }}

        if (reassembleToggle) {{
          reassembleToggle.addEventListener("click", () => {{
            captureUndoState();
            reassembleView(viewSolved);
            reassembleView(viewFull);
            setTimeout(centerCanvas, 0);
          }});
        }}

        if (resetToggle) {{
          resetToggle.addEventListener("click", () => {{
            captureUndoState();
            if (targetSvg) {{
              targetSvg
                .querySelectorAll(".target-bond.selected")
                .forEach((bond) => bond.classList.remove("selected"));
            }}
            selectedBondIds.clear();
            if (clusterExact) {{
              clusterExact.checked = false;
            }}
            updateBondColors();
            activeRouteId = null;
            focusedRouteId = null;
            clearRouteEdgeHighlight();
            setRouteDetailVisible(false);
            if (routeDetail) {{
              routeDetail.dataset.expandedWidth = "";
              routeDetail.dataset.expandedHeight = "";
              routeDetail.style.width = "";
              routeDetail.style.height = "";
              setRouteDetailCollapsed(true);
            }}
            if (viewFull && originalFullHtml) {{
              viewFull.innerHTML = originalFullHtml;
            }}
            if (viewSolved && originalSolvedHtml) {{
              viewSolved.innerHTML = originalSolvedHtml;
            }}
            pruneInvalidEdges(viewFull);
            pruneInvalidEdges(viewSolved);
            bindNodeClicks();
            bindFullZoom();
            closeModal();
            if (hasSolvedRoutes) {{
              setView("solved");
            }} else {{
              setView("full");
            }}
            syncSelectedBonds();
            updateUndoButton();
          }});
        }}

        function refreshSelectedBondsDisplay() {{
          const selected = Array.from(selectedBondIds).map((id) =>
            id.split("-").map((value) => parseInt(value, 10))
          );
          window.selectedBonds = selected;
          if (targetPanel) {{
            targetPanel.dataset.selectedBonds = JSON.stringify(selected);
          }}
          if (targetSelection) {{
            const code = targetSelection.querySelector("code");
            if (code) {{
              code.textContent = JSON.stringify(selected);
            }}
            targetSelection.dataset.targetSmiles = targetSmiles;
          }}
          updateBondColors();
        }}

        function syncSelectedBonds() {{
          refreshSelectedBondsDisplay();
          applyClusterFilter();
        }}

        function setClusterStatus(message) {{
          if (clusterStatus) {{
            clusterStatus.textContent = message;
          }}
        }}

        function updateUndoButton() {{
          if (!undoToggle) {{
            return;
          }}
          const hasUndo = undoStack.length > 0;
          undoToggle.disabled = !hasUndo;
          undoToggle.classList.toggle("disabled", !hasUndo);
        }}

        function captureUndoState() {{
          if (isRestoring) {{
            return;
          }}
          const state = {{
            viewFullHtml: viewFull ? viewFull.innerHTML : "",
            viewSolvedHtml: viewSolved ? viewSolved.innerHTML : "",
            viewMode,
            activeRouteId,
            focusedRouteId,
            selectedBondIds: Array.from(selectedBondIds),
            clusterExact: clusterExact ? clusterExact.checked : false,
            clusterStatus: clusterStatus ? clusterStatus.textContent : "",
            routeIds: currentRouteIds.slice(),
            routeDetailVisible: routeDetail ? !routeDetail.classList.contains("hidden") : false,
            routeDetailCollapsed: routeDetail ? routeDetail.classList.contains("collapsed") : true,
            routeDetailWidth: routeDetail ? routeDetail.style.width : "",
            routeDetailHeight: routeDetail ? routeDetail.style.height : "",
            routeDetailTitle: routeDetailTitle ? routeDetailTitle.textContent : "",
            routeDetailBody: routeDetailBody ? routeDetailBody.innerHTML : "",
            targetCollapsed: targetPanel ? targetPanel.classList.contains("collapsed") : false,
            targetSvg: targetSvg ? targetSvg.innerHTML : "",
            extraMeta: JSON.parse(JSON.stringify(extraMeta)),
            scrollLeft: canvas ? canvas.scrollLeft : 0,
            scrollTop: canvas ? canvas.scrollTop : 0,
          }};
          undoStack.push(state);
          if (undoStack.length > maxUndo) {{
            undoStack.shift();
          }}
          updateUndoButton();
        }}

        function restoreUndoState() {{
          if (!undoStack.length) {{
            return;
          }}
          const state = undoStack.pop();
          isRestoring = true;
          if (viewFull) {{
            viewFull.innerHTML = state.viewFullHtml || "";
          }}
          if (viewSolved) {{
            viewSolved.innerHTML = state.viewSolvedHtml || "";
          }}
          if (targetSvg && state.targetSvg !== undefined) {{
            targetSvg.innerHTML = state.targetSvg;
          }}
          Object.keys(extraMeta).forEach((key) => {{
            delete extraMeta[key];
          }});
          if (state.extraMeta) {{
            Object.entries(state.extraMeta).forEach(([key, value]) => {{
              extraMeta[key] = value;
            }});
          }}
          selectedBondIds.clear();
          (state.selectedBondIds || []).forEach((bondId) => selectedBondIds.add(bondId));
          if (clusterExact) {{
            clusterExact.checked = !!state.clusterExact;
          }}
          if (clusterStatus) {{
            clusterStatus.textContent = state.clusterStatus || "";
          }}
          activeRouteId = state.activeRouteId;
          focusedRouteId = state.focusedRouteId;
          setView(state.viewMode || viewMode);
          if (targetPanel) {{
            targetPanel.classList.toggle("collapsed", !!state.targetCollapsed);
            if (targetToggle) {{
              targetToggle.textContent = targetPanel.classList.contains("collapsed")
                ? "Show"
                : "Hide";
            }}
          }}
          bindNodeClicks();
          pruneInvalidEdges(viewFull);
          pruneInvalidEdges(viewSolved);
          bindFullZoom();
          bindTargetBonds();
          updateRoutePanel(state.routeIds || []);
          if (routeDetail) {{
            routeDetail.classList.toggle("hidden", !state.routeDetailVisible);
            routeDetail.classList.toggle("collapsed", !!state.routeDetailCollapsed);
            if (routeDetailTitle) {{
              routeDetailTitle.textContent = state.routeDetailTitle || "";
            }}
            if (routeDetailBody) {{
              routeDetailBody.innerHTML = state.routeDetailBody || "";
            }}
            routeDetail.style.width = state.routeDetailWidth || "";
            routeDetail.style.height = state.routeDetailHeight || "";
            if (routeDetailToggle) {{
              routeDetailToggle.textContent = routeDetail.classList.contains("collapsed")
                ? "Show"
                : "Hide";
            }}
          }}
          refreshSelectedBondsDisplay();
          if (canvas) {{
            setTimeout(() => {{
              canvas.scrollLeft = state.scrollLeft || 0;
              canvas.scrollTop = state.scrollTop || 0;
            }}, 0);
          }}
          isRestoring = false;
          updateUndoButton();
        }}

        function getMatchingClusters() {{
          if (!clustersLoaded) {{
            return [];
          }}
          if (selectedBondIds.size === 0) {{
            return [];
          }}
          const exactMode = clusterExact && clusterExact.checked;
          const selected = Array.from(selectedBondIds);
          return clusters.filter((cluster) => {{
            if (exactMode && selected.length !== cluster.bondKeys.length) {{
              return false;
            }}
            return selected.every((bondId) => cluster.bondKeys.includes(bondId));
          }});
        }}

        function getColorClusters() {{
          if (!clustersLoaded) {{
            return [];
          }}
          if (selectedBondIds.size === 0) {{
            return [];
          }}
          const selected = Array.from(selectedBondIds);
          return clusters.filter((cluster) =>
            selected.every((bondId) => cluster.bondKeys.includes(bondId))
          );
        }}

        function setBondClass(bondId, className, enabled) {{
          if (!targetSvg) {{
            return;
          }}
          targetSvg
            .querySelectorAll('[data-bond="' + bondId + '"]')
            .forEach((el) => {{
              if (enabled) {{
                el.classList.add(className);
              }} else {{
                el.classList.remove(className);
              }}
            }});
        }}

        function updateBondColors() {{
          if (!targetSvg) {{
            return;
          }}
          const selected = new Set(selectedBondIds);
          const matches = getColorClusters();
          const possibleBonds = new Set();
          if (selected.size === 0) {{
            allClusterBondKeys.forEach((bondId) => possibleBonds.add(bondId));
          }} else {{
            matches.forEach((cluster) => {{
              (cluster.bondKeys || []).forEach((bondId) => possibleBonds.add(bondId));
            }});
          }}
          const bondIds = new Set();
          targetSvg.querySelectorAll(".target-bond").forEach((bond) => {{
            if (bond.dataset.bond) {{
              bondIds.add(bond.dataset.bond);
            }}
          }});
          bondIds.forEach((bondId) => {{
            const isSelected = selected.has(bondId);
            const isClusterBond = allClusterBondKeys.has(bondId);
            if (isSelected) {{
              setBondClass(bondId, "selected", true);
              setBondClass(bondId, "possible", false);
              setBondClass(bondId, "excluded", false);
              return;
            }}
            if (isClusterBond) {{
              const isPossible = selected.size === 0 || possibleBonds.has(bondId);
              setBondClass(bondId, "possible", isPossible);
              setBondClass(bondId, "excluded", !isPossible);
              setBondClass(bondId, "selected", false);
              return;
            }}
            setBondClass(bondId, "selected", false);
            setBondClass(bondId, "possible", false);
            setBondClass(bondId, "excluded", false);
          }});
        }}

        function bindNodeClicks() {{
          nodes = document.querySelectorAll("circle.node");
          nodes.forEach((node) => {{
            if (node.dataset.clickBound === "true") {{
              return;
            }}
            node.dataset.clickBound = "true";
            if (node.classList.contains("route-extra")) {{
              node.addEventListener("click", (event) => {{
                event.stopPropagation();
                nodes.forEach((n) => n.classList.remove("active"));
                node.classList.add("active");
                openModal(node.dataset.nodeId);
              }});
              return;
            }}
            node.addEventListener("click", (event) => {{
              event.stopPropagation();
              nodes.forEach((n) => n.classList.remove("active"));
              node.classList.add("active");
              openModal(node.dataset.nodeId);
            }});
          }});
        }}

        function bindTargetBonds() {{
          if (!targetSvg) {{
            return;
          }}
          targetSvg.querySelectorAll(".target-bond").forEach((bond) => {{
            bond.addEventListener("click", (event) => {{
              event.stopPropagation();
              const bondId = bond.dataset.bond;
              if (!bondId) {{
                return;
              }}
              captureUndoState();
              const isSelected = selectedBondIds.has(bondId);
              if (isSelected) {{
                selectedBondIds.delete(bondId);
              }} else {{
                selectedBondIds.add(bondId);
              }}
              syncSelectedBonds();
            }});
          }});
        }}

        function pruneInvalidEdges(view) {{
          if (!view) {{
            return;
          }}
          view.querySelectorAll(".edges line").forEach((edge) => {{
            const childId = edge.dataset.childId;
            const parentId = edge.dataset.parentId;
            if (!childId || !parentId) {{
              return;
            }}
            const childNode = view.querySelector(
              "circle.node[data-node-id='" + childId + "']"
            );
            if (!childNode) {{
              return;
            }}
            const actualParent = childNode.dataset.parentId;
            if (actualParent && actualParent !== parentId) {{
              edge.classList.add("edge-invalid");
            }} else {{
              edge.classList.remove("edge-invalid");
            }}
          }});
        }}

        function setRouteHeader(currentCount) {{
          if (routeHeader) {{
            routeHeader.textContent =
              "Route ID: Total " + totalRoutes + " Current " + currentCount;
          }}
        }}

        function setRouteDetailVisible(isVisible) {{
          if (!routeDetail) {{
            return;
          }}
          if (isVisible) {{
            routeDetail.classList.remove("hidden");
          }} else {{
            routeDetail.classList.add("hidden");
          }}
        }}

        function setRouteDetailCollapsed(isCollapsed) {{
          if (!routeDetail || !routeDetailToggle) {{
            return;
          }}
          if (isCollapsed) {{
            if (!routeDetail.dataset.expandedWidth || !routeDetail.dataset.expandedHeight) {{
              const rect = routeDetail.getBoundingClientRect();
              routeDetail.dataset.expandedWidth = String(Math.round(rect.width));
              routeDetail.dataset.expandedHeight = String(Math.round(rect.height));
            }}
            routeDetail.classList.add("collapsed");
            routeDetailToggle.textContent = "Show";
            const header = routeDetail.querySelector(".panel-header");
            if (header) {{
              routeDetail.style.height = String(Math.round(header.offsetHeight)) + "px";
            }}
          }} else {{
            routeDetail.classList.remove("collapsed");
            routeDetailToggle.textContent = "Hide";
            const width = parseInt(routeDetail.dataset.expandedWidth || "240", 10);
            const height = parseInt(routeDetail.dataset.expandedHeight || "220", 10);
            routeDetail.style.width = String(width) + "px";
            routeDetail.style.height = String(height) + "px";
          }}
        }}

        function showRouteDetail(routeId) {{
          if (!routeDetailBody || !routeDetailTitle) {{
            return;
          }}
          const svg = routeSvgs[String(routeId)];
          if (svg) {{
            routeDetailBody.innerHTML = svg;
          }} else {{
            routeDetailBody.innerHTML = "<div>Route SVG unavailable.</div>";
          }}
          routeDetailTitle.textContent = "Route " + routeId;
          setRouteDetailVisible(true);
          setRouteDetailCollapsed(false);
        }}

        function positionModal() {{
          if (!modal || !targetPanel) {{
            return;
          }}
          const targetRect = targetPanel.getBoundingClientRect();
          const modalRect = modal.getBoundingClientRect();
          const margin = 12;
          const left = Math.max(16, targetRect.left);
          let top = targetRect.top - modalRect.height - margin;
          if (top < 16) {{
            top = 16;
          }}
          modal.style.left = String(Math.round(left)) + "px";
          modal.style.top = String(Math.round(top)) + "px";
          modal.style.right = "auto";
          modal.style.bottom = "auto";
        }}

        function initRouteResize() {{
          if (!routeDetail || !routeDetailResize) {{
            return;
          }}
          let startX = 0;
          let startY = 0;
          let startWidth = 0;
          let startHeight = 0;
          let resizing = false;

          const onPointerMove = (event) => {{
            if (!resizing) {{
              return;
            }}
            const deltaX = startX - event.clientX;
            const deltaY = startY - event.clientY;
            const minWidth = 180;
            const minHeight = 140;
            const maxWidth = Math.max(minWidth, window.innerWidth - 32);
            const maxHeight = Math.max(minHeight, window.innerHeight - 32);
            const nextWidth = Math.min(
              maxWidth,
              Math.max(minWidth, startWidth + deltaX)
            );
            const nextHeight = Math.min(
              maxHeight,
              Math.max(minHeight, startHeight + deltaY)
            );
            routeDetail.style.width = String(Math.round(nextWidth)) + "px";
            routeDetail.style.height = String(Math.round(nextHeight)) + "px";
            routeDetail.dataset.expandedWidth = String(Math.round(nextWidth));
            routeDetail.dataset.expandedHeight = String(Math.round(nextHeight));
          }};

          const stopResize = () => {{
            if (!resizing) {{
              return;
            }}
            resizing = false;
            document.removeEventListener("pointermove", onPointerMove);
            document.removeEventListener("pointerup", stopResize);
            document.removeEventListener("pointercancel", stopResize);
          }};

          routeDetailResize.addEventListener("pointerdown", (event) => {{
            event.preventDefault();
            if (routeDetail.classList.contains("collapsed")) {{
              return;
            }}
            resizing = true;
            if (routeDetailResize.setPointerCapture) {{
              routeDetailResize.setPointerCapture(event.pointerId);
            }}
            startX = event.clientX;
            startY = event.clientY;
            const rect = routeDetail.getBoundingClientRect();
            startWidth = rect.width;
            startHeight = rect.height;
            routeDetail.dataset.expandedWidth = String(Math.round(startWidth));
            routeDetail.dataset.expandedHeight = String(Math.round(startHeight));
            document.addEventListener("pointermove", onPointerMove);
            document.addEventListener("pointerup", stopResize);
            document.addEventListener("pointercancel", stopResize);
          }});
        }}

        function clearRouteEdgeHighlight() {{
          [viewFull, viewSolved].forEach((view) => {{
            if (!view) {{
              return;
            }}
            view.querySelectorAll(".edges line.edge-route").forEach((edge) => {{
              edge.classList.remove("edge-route");
            }});
          }});
        }}

        function highlightRoute(routeId) {{
          clearRouteEdgeHighlight();
          if (!routeId) {{
            return;
          }}
          const nodes = routeNodes[String(routeId)];
          if (!nodes || nodes.length < 2) {{
            return;
          }}
          const edgeKeys = new Set();
          for (let i = 0; i < nodes.length - 1; i += 1) {{
            const parent = nodes[i + 1];
            const child = nodes[i];
            edgeKeys.add(String(parent) + "-" + String(child));
          }}
          [viewFull, viewSolved].forEach((view) => {{
            if (!view) {{
              return;
            }}
            view.querySelectorAll(".edges line").forEach((edge) => {{
              const parentId = edge.dataset.parentId;
              const childId = edge.dataset.childId;
              if (edgeKeys.has(String(parentId) + "-" + String(childId))) {{
                edge.classList.add("edge-route");
              }}
            }});
          }});
        }}

        function clearRouteFocus() {{
          focusedRouteId = null;
          [viewFull, viewSolved].forEach((view) => {{
            if (!view) {{
              return;
            }}
            view.querySelectorAll(".route-hidden").forEach((el) => {{
              el.classList.remove("route-hidden");
            }});
          }});
          clearRouteExtras();
        }}

        function applyRouteFocus(routeId) {{
          if (!routeId) {{
            clearRouteFocus();
            return;
          }}
          const nodes = routeNodes[String(routeId)];
          if (!nodes || nodes.length === 0) {{
            clearRouteFocus();
            return;
          }}
          focusedRouteId = routeId;
          const allowedNodes = new Set(nodes.map((nodeId) => parseInt(nodeId, 10)));
          const edgeKeys = new Set();
          for (let i = 0; i < nodes.length - 1; i += 1) {{
            const parent = nodes[i + 1];
            const child = nodes[i];
            edgeKeys.add(String(parent) + "-" + String(child));
          }}
          [viewFull, viewSolved].forEach((view) => {{
            if (!view) {{
              return;
            }}
            view.querySelectorAll("circle.node").forEach((node) => {{
              const nodeId = parseInt(node.dataset.nodeId || "0", 10);
              if (allowedNodes.has(nodeId)) {{
                node.classList.remove("route-hidden");
              }} else {{
                node.classList.add("route-hidden");
              }}
            }});
            view.querySelectorAll(".edges line").forEach((edge) => {{
              const parentId = edge.dataset.parentId;
              const childId = edge.dataset.childId;
              const key = String(parentId) + "-" + String(childId);
              if (edgeKeys.has(key)) {{
                edge.classList.remove("route-hidden");
              }} else {{
                edge.classList.add("route-hidden");
              }}
            }});
          }});
          renderRouteExtras(routeId);
        }}

        function clearRouteExtras() {{
          [viewFull, viewSolved].forEach((view) => {{
            if (!view) {{
              return;
            }}
            view.querySelectorAll(".route-extra").forEach((el) => el.remove());
            view
              .querySelectorAll(".route-extra-edge")
              .forEach((el) => el.remove());
          }});
          Object.keys(extraMeta).forEach((key) => {{
            delete extraMeta[key];
          }});
        }}

        function renderRouteExtras(routeId) {{
          clearRouteExtras();
          const payload = routeExtras[String(routeId)];
          if (!payload || !payload.by_parent) {{
            return;
          }}
          [viewFull, viewSolved].forEach((view) => {{
            if (!view) {{
              return;
            }}
            const svg = view.querySelector("svg.tree-svg");
            const edgesGroup = view.querySelector(".edges");
            const nodesGroup = view.querySelector(".nodes");
            if (!svg || !edgesGroup || !nodesGroup) {{
              return;
            }}
            const rootNode = view.querySelector("circle.node[data-node-id='1']");
            if (!rootNode) {{
              return;
            }}
            const rootX = parseFloat(rootNode.getAttribute("cx") || "0");
            const rootY = parseFloat(rootNode.getAttribute("cy") || "0");
            let nodeRadius = 18;
            const sampleNode = view.querySelector(
              "circle.node:not(.filtered-out):not(.route-hidden)"
            );
            if (sampleNode) {{
              nodeRadius = parseFloat(sampleNode.getAttribute("r")) || nodeRadius;
            }}
            const offset = nodeRadius * 3.2;
            const nodePadding = nodeRadius * 0.6;
            const edgePadding = nodeRadius * 0.9;
            const obstacleNodes = [];
            const obstacleEdges = [];
            const placedExtras = [];
            const avoidEdges = view === viewSolved;

            view.querySelectorAll("circle.node").forEach((node) => {{
              if (
                node.classList.contains("filtered-out") ||
                node.classList.contains("route-hidden") ||
                node.classList.contains("route-extra")
              ) {{
                return;
              }}
              const cx = parseFloat(node.getAttribute("cx") || "0");
              const cy = parseFloat(node.getAttribute("cy") || "0");
              const r = parseFloat(node.getAttribute("r")) || nodeRadius;
              obstacleNodes.push({{ x: cx, y: cy, r }});
            }});

            if (avoidEdges) {{
              view.querySelectorAll(".edges line").forEach((edge) => {{
                if (
                  edge.classList.contains("filtered-out") ||
                  edge.classList.contains("route-hidden") ||
                  edge.classList.contains("route-extra-edge")
                ) {{
                  return;
                }}
                const x1 = parseFloat(edge.getAttribute("x1") || "0");
                const y1 = parseFloat(edge.getAttribute("y1") || "0");
                const x2 = parseFloat(edge.getAttribute("x2") || "0");
                const y2 = parseFloat(edge.getAttribute("y2") || "0");
                obstacleEdges.push({{ x1, y1, x2, y2 }});
              }});
            }}

            function pointSegmentDistance(px, py, x1, y1, x2, y2) {{
              const dx = x2 - x1;
              const dy = y2 - y1;
              if (dx === 0 && dy === 0) {{
                return Math.hypot(px - x1, py - y1);
              }}
              const t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);
              const clamped = Math.max(0, Math.min(1, t));
              const projX = x1 + clamped * dx;
              const projY = y1 + clamped * dy;
              return Math.hypot(px - projX, py - projY);
            }}

            function collides(x, y) {{
              for (const obstacle of obstacleNodes) {{
                const minDist = obstacle.r + nodeRadius + nodePadding;
                if (Math.hypot(x - obstacle.x, y - obstacle.y) < minDist) {{
                  return true;
                }}
              }}
              for (const placed of placedExtras) {{
                const minDist = placed.r + nodeRadius + nodePadding;
                if (Math.hypot(x - placed.x, y - placed.y) < minDist) {{
                  return true;
                }}
              }}
              if (avoidEdges) {{
                for (const edge of obstacleEdges) {{
                  if (
                    pointSegmentDistance(x, y, edge.x1, edge.y1, edge.x2, edge.y2) <
                    nodeRadius + edgePadding
                  ) {{
                    return true;
                  }}
                }}
              }}
              return false;
            }}

            Object.entries(payload.by_parent).forEach(([parentId, extras]) => {{
              const parentNode = view.querySelector(
                "circle.node[data-node-id='" + parentId + "']"
              );
              if (
                !parentNode ||
                parentNode.classList.contains("filtered-out") ||
                parentNode.classList.contains("route-hidden")
              ) {{
                return;
              }}
              const px = parseFloat(parentNode.getAttribute("cx") || "0");
              const py = parseFloat(parentNode.getAttribute("cy") || "0");
              const angleToRoot = Math.atan2(rootY - py, rootX - px);
              const baseAngle = angleToRoot + Math.PI;
              const count = extras.length;
              const spread = Math.min(Math.PI / 3, Math.PI / (count + 1));
              extras.forEach((extra, index) => {{
                const baseOffset = (index - (count - 1) / 2) * spread;
                const angleOffsets = [];
                const steps = Math.max(4, count + 2);
                for (let step = 0; step < steps; step += 1) {{
                  const shift = Math.ceil(step / 2);
                  const sign = step % 2 === 0 ? 1 : -1;
                  angleOffsets.push(sign * shift * spread);
                }}
                const radiusSteps = [1, 1.4, 1.8, 2.3, 2.8, 3.4];
                let ex = px + Math.cos(baseAngle + baseOffset) * offset;
                let ey = py + Math.sin(baseAngle + baseOffset) * offset;
                let placed = false;
                for (const radiusMult of radiusSteps) {{
                  for (const angleShift of angleOffsets) {{
                    const angle = baseAngle + baseOffset + angleShift;
                    const candidateX = px + Math.cos(angle) * offset * radiusMult;
                    const candidateY = py + Math.sin(angle) * offset * radiusMult;
                    if (!collides(candidateX, candidateY)) {{
                      ex = candidateX;
                      ey = candidateY;
                      placed = true;
                      break;
                    }}
                  }}
                  if (placed) {{
                    break;
                  }}
                }}
                const extraId = `extra-${{parentId}}-${{index}}`;
                extraMeta[extraId] = {{
                  node_id: extraId,
                  route_id: routeId,
                  visits: "",
                  num_children: "",
                  rule_id: "",
                  solved: extra.status === "starting material",
                  smiles: extra.smiles || "",
                  svg: extra.svg || "",
                  is_extra: true,
                }};
                const line = document.createElementNS(
                  "http://www.w3.org/2000/svg",
                  "line"
                );
                line.setAttribute("x1", px.toFixed(2));
                line.setAttribute("y1", py.toFixed(2));
                line.setAttribute("x2", ex.toFixed(2));
                line.setAttribute("y2", ey.toFixed(2));
                line.setAttribute("data-parent-id", parentId);
                line.setAttribute("data-child-id", extraId);
                line.setAttribute("class", "edge-route route-extra-edge");
                edgesGroup.appendChild(line);

                const circle = document.createElementNS(
                  "http://www.w3.org/2000/svg",
                  "circle"
                );
                const status = extra.status === "starting material" ? "starting material" : "intermediate";
                circle.setAttribute("cx", ex.toFixed(2));
                circle.setAttribute("cy", ey.toFixed(2));
                circle.setAttribute("r", nodeRadius.toFixed(2));
                circle.setAttribute("data-node-id", extraId);
                circle.setAttribute("data-node-status", status);
                circle.setAttribute("class", `node node-${{status}} route-extra`);
                nodesGroup.appendChild(circle);
                placedExtras.push({{ x: ex, y: ey, r: nodeRadius }});
              }});
            }});
          }});
          bindNodeClicks();
        }}

        function renderRouteButtons(routeIds) {{
          if (!routeList) {{
            return;
          }}
          routeList.textContent = "";
          routeIds.forEach((routeId) => {{
            const button = document.createElement("button");
            button.type = "button";
            button.className = "route-btn";
            button.textContent = routeId;
            button.dataset.routeId = routeId;
            if (String(routeId) === String(activeRouteId)) {{
              button.classList.add("active");
            }}
            button.addEventListener("click", () => {{
              captureUndoState();
              if (String(routeId) === String(activeRouteId)) {{
                activeRouteId = null;
                focusedRouteId = null;
                routeList.querySelectorAll(".route-btn").forEach((btn) => {{
                  btn.classList.remove("active");
                }});
                clearRouteEdgeHighlight();
                clearRouteFocus();
                setRouteDetailVisible(false);
                return;
              }}
              activeRouteId = routeId;
              focusedRouteId = routeId;
              routeList.querySelectorAll(".route-btn").forEach((btn) => {{
                btn.classList.remove("active");
              }});
              button.classList.add("active");
              highlightRoute(routeId);
              showRouteDetail(routeId);
              applyRouteFocus(routeId);
            }});
            routeList.appendChild(button);
          }});
        }}

        function updateRoutePanel(routeIds) {{
          const unique = Array.from(new Set(routeIds)).sort((a, b) => a - b);
          currentRouteIds = unique.slice();
          setRouteHeader(unique.length);
          renderRouteButtons(unique);
          const hasActive = unique.some(
            (routeId) => String(routeId) === String(activeRouteId)
          );
          if (!hasActive) {{
            activeRouteId = null;
            focusedRouteId = null;
            clearRouteFocus();
            clearRouteEdgeHighlight();
            setRouteDetailVisible(false);
          }}
        }}

        function clearClusterFilter() {{
          [viewFull, viewSolved].forEach((view) => {{
            if (!view) {{
              return;
            }}
            view.querySelectorAll(".filtered-out").forEach((el) => {{
              el.classList.remove("filtered-out");
            }});
          }});
        }}

        function applyFilterToView(view, allowedNodes) {{
          if (!view) {{
            return;
          }}
          view.querySelectorAll("circle.node").forEach((node) => {{
            if (node.classList.contains("route-extra")) {{
              return;
            }}
            const nodeId = parseInt(node.dataset.nodeId || "0", 10);
            if (allowedNodes.has(nodeId)) {{
              node.classList.remove("filtered-out");
            }} else {{
              node.classList.add("filtered-out");
            }}
          }});
          view.querySelectorAll(".edges line").forEach((edge) => {{
            if (edge.classList.contains("route-extra-edge")) {{
              return;
            }}
            const parentId = parseInt(edge.dataset.parentId || "0", 10);
            const childId = parseInt(edge.dataset.childId || "0", 10);
            if (allowedNodes.has(parentId) && allowedNodes.has(childId)) {{
              edge.classList.remove("filtered-out");
            }} else {{
              edge.classList.add("filtered-out");
            }}
          }});
        }}

        function reassembleView(view) {{
          if (!view) {{
            return;
          }}
          const svg = view.querySelector("svg.tree-svg");
          if (!svg) {{
            return;
          }}
          const visibleNodes = new Set();
          view.querySelectorAll("circle.node").forEach((node) => {{
            if (
              !node.classList.contains("filtered-out") &&
              !node.classList.contains("route-hidden")
            ) {{
              const nodeId = parseInt(node.dataset.nodeId || "0", 10);
              if (nodeId) {{
                visibleNodes.add(nodeId);
              }}
            }}
          }});
          if (!visibleNodes.has(1)) {{
            return;
          }}
          const childrenMap = {{}};
          visibleNodes.forEach((nodeId) => {{
            childrenMap[nodeId] = [];
          }});
          view.querySelectorAll(".edges line").forEach((edge) => {{
            if (
              edge.classList.contains("filtered-out") ||
              edge.classList.contains("route-hidden")
            ) {{
              return;
            }}
            const parentId = parseInt(edge.dataset.parentId || "0", 10);
            const childId = parseInt(edge.dataset.childId || "0", 10);
            if (visibleNodes.has(parentId) && visibleNodes.has(childId)) {{
              childrenMap[parentId].push(childId);
            }}
          }});
          Object.values(childrenMap).forEach((children) => {{
            children.sort((a, b) => a - b);
          }});

          const depths = {{}};
          const queue = [];
          depths[1] = 0;
          queue.push(1);
          while (queue.length > 0) {{
            const nodeId = queue.shift();
            const children = childrenMap[nodeId] || [];
            children.forEach((childId) => {{
              if (depths[childId] !== undefined) {{
                return;
              }}
              depths[childId] = depths[nodeId] + 1;
              queue.push(childId);
            }});
          }}

          const order = [];
          const stack = [1];
          while (stack.length > 0) {{
            const nodeId = stack.pop();
            order.push(nodeId);
            const children = childrenMap[nodeId] || [];
            for (let i = children.length - 1; i >= 0; i -= 1) {{
              stack.push(children[i]);
            }}
          }}
          const leafCounts = {{}};
          for (let i = order.length - 1; i >= 0; i -= 1) {{
            const nodeId = order[i];
            const children = childrenMap[nodeId] || [];
            if (children.length === 0) {{
              leafCounts[nodeId] = 1;
            }} else {{
              let total = 0;
              children.forEach((childId) => {{
                total += leafCounts[childId] || 1;
              }});
              leafCounts[nodeId] = total;
            }}
          }}

          const angles = {{}};
          const angleStack = [[1, 0.0, Math.PI * 2]];
          while (angleStack.length > 0) {{
            const item = angleStack.pop();
            const nodeId = item[0];
            const startAngle = item[1];
            const endAngle = item[2];
            angles[nodeId] = (startAngle + endAngle) / 2.0;
            const children = childrenMap[nodeId] || [];
            if (children.length === 0) {{
              continue;
            }}
            const span = Math.max(0.0, endAngle - startAngle);
            let total = 0;
            children.forEach((childId) => {{
              total += leafCounts[childId] || 1;
            }});
            if (total <= 0 || span <= 0) {{
              continue;
            }}
            let gap = 0.0;
            if (children.length > 1) {{
              const maxGap = (span * 0.15) / (children.length - 1);
              gap = Math.min(0.04, maxGap);
            }}
            let spanForChildren = span - gap * (children.length - 1);
            if (spanForChildren <= 0) {{
              gap = 0.0;
              spanForChildren = span;
            }}
            let cursor = startAngle;
            children.forEach((childId) => {{
              const frac = (leafCounts[childId] || 1) / total;
              const childSpan = spanForChildren * frac;
              const childStart = cursor;
              const childEnd = cursor + childSpan;
              angleStack.push([childId, childStart, childEnd]);
              cursor = childEnd + gap;
            }});
          }}

          const byDepth = {{}};
          Object.keys(depths).forEach((nodeId) => {{
            const depth = depths[nodeId];
            if (!byDepth[depth]) {{
              byDepth[depth] = [];
            }}
            byDepth[depth].push(parseInt(nodeId, 10));
          }});
          Object.values(byDepth).forEach((nodesList) => nodesList.sort((a, b) => a - b));

          let nodeRadius = 20;
          const sampleNode = Array.from(view.querySelectorAll("circle.node")).find(
            (node) =>
              !node.classList.contains("filtered-out") &&
              !node.classList.contains("route-hidden") &&
              node.dataset.nodeStatus !== "target"
          );
          if (sampleNode) {{
            nodeRadius = parseFloat(sampleNode.getAttribute("r")) || nodeRadius;
          }} else {{
            const rootNode = view.querySelector("circle.node[data-node-id='1']");
            if (rootNode) {{
              nodeRadius = (parseFloat(rootNode.getAttribute("r")) || nodeRadius) / 2;
            }}
          }}

          let baseStep = 200;
          let foundStep = false;
          const depthNodes = Object.keys(depths).filter((nodeId) => depths[nodeId] === 1);
          if (depthNodes.length > 0) {{
            let total = 0;
            depthNodes.forEach((nodeId) => {{
              const nodeEl = view.querySelector(`circle.node[data-node-id='${{nodeId}}']`);
              if (!nodeEl) {{
                return;
              }}
              const x = parseFloat(nodeEl.getAttribute("cx"));
              const y = parseFloat(nodeEl.getAttribute("cy"));
              if (!Number.isNaN(x) && !Number.isNaN(y)) {{
                total += Math.hypot(x, y);
              }}
            }});
            if (total > 0) {{
              baseStep = total / depthNodes.length;
              foundStep = true;
            }}
          }}
          if (!foundStep) {{
            const ring = view.querySelector(".depth-ring");
            if (ring) {{
              const ringValue = parseFloat(ring.getAttribute("r"));
              if (!Number.isNaN(ringValue)) {{
                baseStep = ringValue;
              }}
            }}
          }}
          baseStep *= 0.7;

          const minDistance = nodeRadius * 2.6;
          let scale = Math.max(1, minDistance / Math.max(baseStep, 1e-6));
          const rootGap = nodeRadius * 2.8;
          scale = Math.max(scale, rootGap / Math.max(baseStep, 1e-6));
          Object.keys(byDepth).forEach((depthStr) => {{
            const depth = parseInt(depthStr, 10);
            const nodesList = byDepth[depth];
            if (depth === 0 || nodesList.length < 2) {{
              return;
            }}
            const radius = depth * baseStep;
            const depthAngles = nodesList
              .map((nodeId) => angles[nodeId] || 0.0)
              .sort((a, b) => a - b);
            const deltas = [];
            for (let i = 0; i < depthAngles.length - 1; i += 1) {{
              deltas.push(depthAngles[i + 1] - depthAngles[i]);
            }}
            deltas.push(
              Math.PI * 2 - depthAngles[depthAngles.length - 1] + depthAngles[0]
            );
            const minDelta = Math.max(Math.min(...deltas), 1e-6);
            const required = minDistance / (radius * minDelta);
            scale = Math.max(scale, required);
          }});
          const radiusStep = baseStep * scale;

          Object.keys(childrenMap).forEach((parentId) => {{
            const children = childrenMap[parentId] || [];
            if (children.length === 1) {{
              const childId = children[0];
              const depth = depths[childId] || 1;
              const nudge = 0.08 / (depth + 1);
              const direction = parseInt(childId, 10) % 2 === 0 ? 1 : -1;
              angles[childId] = (angles[childId] || 0) + direction * nudge;
            }}
          }});

          Object.keys(byDepth).forEach((depthStr) => {{
            const depth = parseInt(depthStr, 10);
            if (!depth || depth < 1) {{
              return;
            }}
            const nodesList = byDepth[depth];
            if (!nodesList || nodesList.length < 2) {{
              return;
            }}
            const radius = depth * radiusStep;
            const minAngle = Math.min(
              (Math.PI * 2) / nodesList.length,
              (nodeRadius * 2.4) / Math.max(radius, 1e-6)
            );
            const sorted = nodesList
              .slice()
              .sort((a, b) => (angles[a] || 0) - (angles[b] || 0));
            let prevAngle = angles[sorted[0]] || 0;
            for (let i = 1; i < sorted.length; i += 1) {{
              const nodeId = sorted[i];
              let angle = angles[nodeId] || 0;
              if (angle - prevAngle < minAngle) {{
                angle = prevAngle + minAngle;
                angles[nodeId] = angle;
              }}
              prevAngle = angle;
            }}
          }});

          const maxDepthByNode = {{}};
          const farthestAngles = {{}};
          for (let i = order.length - 1; i >= 0; i -= 1) {{
            const nodeId = order[i];
            const children = childrenMap[nodeId] || [];
            if (children.length === 0) {{
              maxDepthByNode[nodeId] = depths[nodeId];
              farthestAngles[nodeId] = [angles[nodeId] || 0];
              continue;
            }}
            let maxDepth = -1;
            let angleList = [];
            children.forEach((childId) => {{
              const childDepth = maxDepthByNode[childId];
              if (childDepth === undefined) {{
                return;
              }}
              const childAngles = farthestAngles[childId] || [];
              if (childDepth > maxDepth) {{
                maxDepth = childDepth;
                angleList = childAngles.slice();
              }} else if (childDepth === maxDepth) {{
                angleList = angleList.concat(childAngles);
              }}
            }});
            maxDepthByNode[nodeId] = maxDepth;
            farthestAngles[nodeId] = angleList;
            if (angleList.length > 0) {{
              const sum = angleList.reduce((acc, value) => acc + value, 0);
              angles[nodeId] = sum / angleList.length;
            }}
          }}

          const positions = {{}};
          Object.keys(depths).forEach((nodeId) => {{
            const depth = depths[nodeId];
            if (depth === 0) {{
              positions[nodeId] = [0.0, 0.0];
              return;
            }}
            const angle = angles[nodeId] || 0.0;
            const radius = depth * radiusStep;
            positions[nodeId] = [
              radius * Math.cos(angle),
              radius * Math.sin(angle),
            ];
          }});

          view.querySelectorAll("circle.node").forEach((node) => {{
            if (
              node.classList.contains("filtered-out") ||
              node.classList.contains("route-hidden")
            ) {{
              return;
            }}
            const nodeId = node.dataset.nodeId;
            const pos = positions[nodeId];
            if (!pos) {{
              return;
            }}
            node.setAttribute("cx", pos[0].toFixed(2));
            node.setAttribute("cy", pos[1].toFixed(2));
          }});

          view.querySelectorAll(".edges line").forEach((edge) => {{
            if (
              edge.classList.contains("filtered-out") ||
              edge.classList.contains("route-hidden")
            ) {{
              return;
            }}
            const parentId = edge.dataset.parentId;
            const childId = edge.dataset.childId;
            const parentPos = positions[parentId];
            const childPos = positions[childId];
            if (!parentPos || !childPos) {{
              return;
            }}
            edge.setAttribute("x1", parentPos[0].toFixed(2));
            edge.setAttribute("y1", parentPos[1].toFixed(2));
            edge.setAttribute("x2", childPos[0].toFixed(2));
            edge.setAttribute("y2", childPos[1].toFixed(2));
          }});

          const depthGroup = view.querySelector(".depth-rings");
          if (depthGroup) {{
            const maxDepth = Math.max(...Object.values(depths));
            let ringMarkup = "";
            for (let depth = 1; depth <= maxDepth; depth += 1) {{
              ringMarkup +=
                '<circle class="depth-ring" cx="0" cy="0" r="' +
                (depth * radiusStep).toFixed(2) +
                '" />';
            }}
            depthGroup.innerHTML = ringMarkup;
          }}

          const xs = [];
          const ys = [];
          Object.keys(positions).forEach((nodeId) => {{
            const pos = positions[nodeId];
            xs.push(pos[0]);
            ys.push(pos[1]);
          }});
          if (xs.length > 0 && ys.length > 0) {{
            const pad = nodeRadius * 4;
            const minX = Math.min(...xs) - pad;
            const maxX = Math.max(...xs) + pad;
            const minY = Math.min(...ys) - pad;
            const maxY = Math.max(...ys) + pad;
            const width = maxX - minX;
            const height = maxY - minY;
            svg.setAttribute(
              "viewBox",
              minX.toFixed(2) +
                " " +
                minY.toFixed(2) +
                " " +
                width.toFixed(2) +
                " " +
                height.toFixed(2)
            );
          }}
        }}

        function applyClusterFilter() {{
          if (!clustersLoaded) {{
            clearClusterFilter();
            setClusterStatus("Clusters: not loaded");
            updateRoutePanel([]);
            return;
          }}
          if (selectedBondIds.size === 0) {{
            clearClusterFilter();
            setClusterStatus("Clusters: no bonds selected");
            const allRoutes = Object.keys(routeNodes)
              .map((routeId) => parseInt(routeId, 10))
              .filter((routeId) => !Number.isNaN(routeId));
            updateRoutePanel(allRoutes);
            return;
          }}
          const exactMode = clusterExact && clusterExact.checked;
          const selected = Array.from(selectedBondIds);
          const matches = getMatchingClusters();
          if (matches.length === 0) {{
            clearClusterFilter();
            if (exactMode) {{
              setClusterStatus("no clusters chosen, please check other");
            }} else {{
              setClusterStatus("Clusters matched: 0");
            }}
            updateRoutePanel([]);
            return;
          }}
          const matchedRouteIds = new Set();
          const allowedNodes = new Set();
          matches.forEach((cluster) => {{
            (cluster.route_ids || []).forEach((routeId) => {{
              const routeNum = parseInt(routeId, 10);
              if (!Number.isNaN(routeNum)) {{
                matchedRouteIds.add(routeNum);
              }}
              const nodes = routeNodes[String(routeId)];
              if (!nodes) {{
                return;
              }}
              nodes.forEach((nodeId) => allowedNodes.add(nodeId));
            }});
          }});
          updateRoutePanel(Array.from(matchedRouteIds));
          if (allowedNodes.size === 0) {{
            clearClusterFilter();
            setClusterStatus(`Clusters matched: ${{matches.length}}`);
            return;
          }}
          applyFilterToView(viewFull, allowedNodes);
          applyFilterToView(viewSolved, allowedNodes);
          setClusterStatus(`Clusters matched: ${{matches.length}}`);
        }}

        bindTargetBonds();

        if (clusterExact) {{
          clusterExact.addEventListener("change", () => {{
            captureUndoState();
            updateBondColors();
            applyClusterFilter();
          }});
        }}

        function openModal(nodeId) {{
          modalBackdrop.classList.add("show");
          modalBackdrop.dataset.nodeId = nodeId;
          const meta = extraMeta[nodeId] || nodeMeta[nodeId] || {{}};
          const imageBlock = meta.svg
            ? `<div class="mol-frame">${{meta.svg}}</div>`
            : "";
          const rows = [
            ["node_id", meta.node_id],
            ["route_id", meta.route_id ?? ""],
            ["visits", meta.visits],
            ["num_children", meta.num_children],
            ["rule_id", meta.rule_id ?? ""],
            ["solved", meta.solved],
          ];
          const tableRows = rows
            .map(([label, value]) => `<tr><th>${{label}}</th><td>${{value}}</td></tr>`)
            .join("");
          const table = `<table class="meta-table">${{tableRows}}</table>`;
          modalBody.innerHTML = `${{imageBlock}}${{table}}`;
          setTimeout(positionModal, 0);
        }}

        function closeModal() {{
          modalBackdrop.classList.remove("show");
          modalBackdrop.dataset.nodeId = "";
          modalBody.textContent = "";
        }}

        bindNodeClicks();
        pruneInvalidEdges(viewFull);
        pruneInvalidEdges(viewSolved);

        modalBackdrop.addEventListener("click", (event) => {{
          if (event.target === modalBackdrop) {{
            closeModal();
          }}
        }});

        window.addEventListener("pointermove", (event) => {{
          updateZoomRect(event);
        }});

        window.addEventListener("pointerup", (event) => {{
          finishZoom(event);
        }});

        window.addEventListener("pointercancel", (event) => {{
          finishZoom(event);
        }});

        document.addEventListener("keydown", (event) => {{
          if (event.key === "Escape") {{
            closeModal();
          }}
        }});
      </script>
    </body>
    </html>
    """

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a simple HTML visualization for a SynPlanner MCTS tree."
    )
    parser.add_argument(
        "--tree",
        type=Path,
        required=True,
        help="Path to a pickled Tree or TreeWrapper object.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tree_visualization.html"),
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--radius-step",
        type=float,
        default=280.0,
        help="Radial distance between depth rings.",
    )
    parser.add_argument(
        "--node-radius",
        type=float,
        default=80.0,
        help="Node radius in SVG units (pixels when not fit-to-screen).",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=0.25,
        help="Scale factor applied to the final render (1.0 = full size).",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=None,
        help="Optional path to a clusters pickle (cluster_routes output).",
    )
    args = parser.parse_args()

    tree = _load_tree(args.tree_pkl)
    if not isinstance(tree, Tree):
        raise TypeError("Loaded object is not a Tree.")

    generate_tree_html(
        tree,
        output_path=args.out,
        radius_step=args.radius_step,
        node_radius=args.node_radius,
        render_scale=args.render_scale,
        clusters_pkl=args.clusters_pkl,
    )
    print(f"Tree visualization written to {args.out}")


if __name__ == "__main__":
    main()
