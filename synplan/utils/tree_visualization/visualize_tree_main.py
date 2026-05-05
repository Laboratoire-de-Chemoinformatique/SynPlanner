#!/usr/bin/env python3
"""
Expansion timeline visualization for a SynPlanner MCTS tree.

This module approximates expansion order by node creation order (node_id).
In the current Tree implementation, node IDs are assigned sequentially in
Tree._add_node.

python -m synplan.utils.tree_visualization.visualize_tree_main --tree tree.pkl --output expansion_evol.html
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import math
from collections.abc import Iterable
from pathlib import Path

from synplan.mcts.tree import Tree
from synplan.utils.tree_visualization.clusters import (
    build_cluster_payload as _build_cluster_payload,
)
from synplan.utils.tree_visualization.layout import (
    bounds_with_pad as _bounds_with_pad,
)
from synplan.utils.tree_visualization.layout import (
    build_children_map as _build_children_map,
)
from synplan.utils.tree_visualization.layout import (
    compute_depths as _compute_depths,
)
from synplan.utils.tree_visualization.layout import (
    edge_key as _edge_key,
)
from synplan.utils.tree_visualization.layout import (
    radial_layout as _radial_layout,
)
from synplan.utils.tree_visualization.layout import (
    scale_positions as _scale_positions,
)
from synplan.utils.tree_visualization.molecules import (
    build_target_svg as _build_target_svg,
)
from synplan.utils.tree_visualization.molecules import (
    curr_precursor_key as _curr_precursor_key,
)
from synplan.utils.tree_visualization.molecules import (
    molecule_key as _molecule_key,
)
from synplan.utils.tree_visualization.molecules import (
    molecule_smiles_and_svg as _molecule_smiles_and_svg,
)
from synplan.utils.tree_visualization.molecules import (
    node_product_molecules as _node_product_molecules,
)
from synplan.utils.tree_visualization.routes import (
    route_nodes_by_route as _route_nodes_by_route,
)
from synplan.utils.tree_visualization.tree_io import (
    load_clusters as _load_clusters,
)
from synplan.utils.tree_visualization.tree_io import (
    load_tree as _load_tree,
)


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
        return "leaf" if node.is_solved() else "intermediate"
    except Exception:
        return "intermediate"


def _spread_product_positions(
    base_x: float,
    base_y: float,
    count: int,
    bubble_radius: float,
    *,
    parent_pos: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
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

    # Use a small ring whose radius grows gently with the product count.
    ring_scale = 2.2 + 0.45 * math.sqrt(max(0, count - 1))
    ring_radius = bubble_radius * ring_scale

    # Keep the primary (anchor) product at the base position to preserve layout.
    positions: list[tuple[float, float]] = [(base_x, base_y)]
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


def _select_node_ids(tree: Tree, max_nodes: int | None) -> list[int]:
    node_ids = sorted(int(nid) for nid in tree.nodes)
    if max_nodes and max_nodes > 0:
        node_ids = node_ids[:max_nodes]
    if 1 in tree.nodes and 1 not in node_ids:
        node_ids = [1, *node_ids]
    return node_ids


def _creation_steps(node_ids: Iterable[int], sample_step: int) -> dict[int, int]:
    sample_step = max(1, int(sample_step))
    ordered = sorted(set(int(nid) for nid in node_ids))
    steps: dict[int, int] = {}
    for idx, node_id in enumerate(ordered):
        steps[node_id] = idx // sample_step
    return steps


def _has_expandable_nodes(tree: Tree, node_ids: set[int]) -> bool:
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


def _winning_route_paths(
    tree: Tree,
    *,
    allowed_tree_nodes: set[int],
    tree_steps: dict[int, int],
    anchor_by_tree: dict[int, int],
    render_edge_keys_by_tree_edge: dict[str, list[str]],
) -> tuple[set[str], list[dict[str, object]], int | None, int | None]:
    winning_nodes = list(getattr(tree, "winning_nodes", []) or [])
    winning_edge_keys: set[str] = set()
    paths: list[dict[str, object]] = []
    win_steps: list[int] = []

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

        edge_keys: list[str] = []
        current = node_id
        seen: set[int] = set()
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


def _build_payload(
    tree: Tree,
    *,
    max_nodes: int | None,
    sample_step: int,
    with_svg: bool,
    radius_step: float,
    render_scale: float,
    node_radius: float | None,
    bubble_scale: float,
    clusters_pkl: Path | None = None,
    clusters_data: dict[str, dict] | None = None,
) -> dict[str, object]:
    selected_ids = _select_node_ids(tree, max_nodes=max_nodes)
    allowed_tree_nodes: set[int] = set(selected_ids)
    allowed_tree_nodes.add(1)

    children_map = _build_children_map(tree, allowed_nodes=allowed_tree_nodes)
    nodes_depth_tree = _compute_depths(children_map)
    if not nodes_depth_tree:
        raise ValueError("Tree has no nodes to render.")

    num_nodes = len(nodes_depth_tree)
    base_radius = (
        node_radius if node_radius is not None else _auto_node_radius(num_nodes)
    )
    bubble_scale = max(0.1, float(bubble_scale))
    has_expandable_nodes = _has_expandable_nodes(tree, allowed_tree_nodes)
    layout_positions = _radial_layout(
        nodes_depth_tree,
        children_map,
        radius_step=radius_step,
        node_radius=base_radius,
    )
    positions_tree = _scale_positions(layout_positions, render_scale)
    base_scaled_radius = base_radius * max(0.01, float(render_scale))
    bubble_radius = base_scaled_radius * bubble_scale

    winning_nodes = list(getattr(tree, "winning_nodes", []) or [])
    route_index_by_tree = {
        int(node_id): idx for idx, node_id in enumerate(winning_nodes)
    }

    tree_steps = _creation_steps(nodes_depth_tree.keys(), sample_step=sample_step)
    total_steps = max(tree_steps.values()) if tree_steps else 0

    nodes_visit = getattr(tree, "nodes_visit", {}) or {}
    nodes_rules = getattr(tree, "nodes_rules", {}) or {}
    nodes_rule_label = getattr(tree, "nodes_rule_label", {}) or {}
    children = getattr(tree, "children", {}) or {}

    # Molecule-level render nodes: one bubble per product molecule.
    tree_to_render_ids: dict[int, list[int]] = {}
    anchor_by_tree: dict[int, int] = {}
    render_positions: dict[int, tuple[float, float]] = {}
    render_steps: dict[int, int] = {}

    nodes_payload: list[dict[str, object]] = []
    molecule_svgs: dict[str, str] = {}
    next_render_id = (max(allowed_tree_nodes) + 1) if allowed_tree_nodes else 2

    for node_id in sorted(nodes_depth_tree.keys()):
        node = tree.nodes.get(node_id)
        if node is None:
            continue

        base_x, base_y = positions_tree.get(node_id, (0.0, 0.0))
        parent_tree_id = (
            int(tree.parents.get(node_id, 0)) if hasattr(tree, "parents") else 0
        )
        parent_pos = (
            positions_tree.get(parent_tree_id)
            if parent_tree_id in positions_tree
            else None
        )

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

        render_ids: list[int] = []
        anchor_render_id = int(node_id)
        extra_counter = 0
        for product_index, molecule in enumerate(molecules):
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
            if svg:
                molecule_svgs[str(render_id)] = svg
            num_children = len(children.get(node_id, [])) if is_anchor else 0
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
                    "parent": int(parent_tree_id) if parent_tree_id else 0,
                    "visits": int(nodes_visit.get(node_id, 0)),
                    "num_children": num_children,
                    "rule_id": nodes_rules.get(node_id),
                    "rule_label": nodes_rule_label.get(node_id),
                    "status": _node_status(tree, node_id),
                    "step": int(render_steps.get(render_id, 0)),
                    "route_index": route_index_by_tree.get(node_id),
                    "smiles": smiles,
                }
            )

        tree_to_render_ids[node_id] = render_ids
        anchor_by_tree[node_id] = anchor_render_id

    # Render edges: connect each parent anchor to every product bubble.
    edges_payload: list[dict[str, object]] = []
    render_edge_keys_by_tree_edge: dict[str, list[str]] = {}
    for child_id, parent_id in getattr(tree, "parents", {}).items():
        if not parent_id or child_id == 1:
            continue
        if child_id not in nodes_depth_tree or parent_id not in nodes_depth_tree:
            continue

        parent_anchor = int(anchor_by_tree.get(parent_id, parent_id))
        child_render_ids = tree_to_render_ids.get(int(child_id)) or [int(child_id)]
        tree_edge_key = _edge_key(parent_id, child_id)

        render_keys: list[str] = []
        for render_child_id in child_render_ids:
            render_key = _edge_key(parent_anchor, render_child_id)
            render_keys.append(render_key)
            edges_payload.append(
                {
                    "parent": int(parent_anchor),
                    "child": int(render_child_id),
                    "step": int(tree_steps.get(child_id, 0)),
                    "key": render_key,
                    "tree_key": tree_edge_key,
                    "winning": False,
                }
            )
        render_edge_keys_by_tree_edge[tree_edge_key] = render_keys

    winning_edge_keys, winning_paths, win_first_step, win_last_step = (
        _winning_route_paths(
            tree,
            allowed_tree_nodes=allowed_tree_nodes,
            tree_steps=tree_steps,
            anchor_by_tree=anchor_by_tree,
            render_edge_keys_by_tree_edge=render_edge_keys_by_tree_edge,
        )
    )

    if winning_edge_keys:
        for edge in edges_payload:
            if edge.get("key") in winning_edge_keys:
                edge["winning"] = True

    # Recolor extra product bubbles that touch winning edges to green.
    winning_edge_connected_ids: set[int] = set()
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
    cumulative_counts: list[int] = []
    running = 0
    for count in step_counts:
        running += count
        cumulative_counts.append(running)

    winning_node_ids: set[int] = set()
    for path in winning_paths:
        with contextlib.suppress(Exception):
            winning_node_ids.add(int(path.get("nodeId")))
    for edge in edges_payload:
        if edge.get("winning"):
            winning_node_ids.add(int(edge.get("parent", 0)))
            winning_node_ids.add(int(edge.get("child", 0)))

    target_svg = _build_target_svg(tree)
    target_smiles = str(tree.nodes[1].curr_precursor) if tree.nodes.get(1) else ""
    clusters_payload: list[dict[str, object]] = []
    route_nodes: dict[str, list[int]] = {}
    if clusters_data is not None or clusters_pkl:
        clusters = (
            clusters_data if clusters_data is not None else _load_clusters(clusters_pkl)
        )
        clusters_payload = _build_cluster_payload(clusters)
        cluster_route_ids = {
            route_id
            for cluster in clusters_payload
            for route_id in cluster.get("route_ids", [])
        }
        route_nodes = _route_nodes_by_route(tree, cluster_route_ids)

    return {
        "nodes": nodes_payload,
        "edges": edges_payload,
        "stats": {
            "totalNodesInTree": len(getattr(tree, "nodes", {}) or {}),
            "renderedNodes": len(nodes_payload),
            "renderedEdges": len(edges_payload),
            "maxDepth": int(max_depth),
            "totalSteps": int(total_steps),
            "sampleStep": int(max(1, sample_step)),
            "withSvg": bool(with_svg),
            "bubbleScale": float(bubble_scale),
            "spacingScale": 1.0,
            "hasExpandableNodes": bool(has_expandable_nodes),
            "winningNodesTotal": len(winning_node_ids),
            "winningEdgesTotal": len(winning_edge_keys),
            "winningFirstStep": int(win_first_step)
            if win_first_step is not None
            else None,
            "winningLastStep": int(win_last_step)
            if win_last_step is not None
            else None,
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
        "targetSvg": target_svg,
        "targetSmiles": target_smiles,
        "clusters": clusters_payload,
        "routeNodes": route_nodes,
        "moleculeSvgs": molecule_svgs,
    }


def generate_expansion_html(
    tree: Tree,
    output_path: Path,
    *,
    max_nodes: int | None,
    sample_step: int,
    with_svg: bool,
    radius_step: float,
    render_scale: float,
    node_radius: float | None,
    bubble_scale: float,
    clusters_pkl: Path | None = None,
    clusters_data: dict[str, dict] | None = None,
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
        clusters_pkl=clusters_pkl,
        clusters_data=clusters_data,
    )

    nodes_json = json.dumps(payload["nodes"], ensure_ascii=True)
    edges_json = json.dumps(payload["edges"], ensure_ascii=True)
    stats_json = json.dumps(payload["stats"], ensure_ascii=True)
    winning_paths_json = json.dumps(payload.get("winningPaths", []), ensure_ascii=True)
    steps_json = json.dumps(payload["steps"], ensure_ascii=True)
    view_box_json = json.dumps(payload["viewBox"], ensure_ascii=True)
    clusters_json = json.dumps(payload.get("clusters", []), ensure_ascii=True)
    route_nodes_json = json.dumps(payload.get("routeNodes", {}), ensure_ascii=True)
    molecule_svg_records = "\n".join(
        f"{node_id}\t{base64.b64encode(svg.encode('utf-8')).decode('ascii')}"
        for node_id, svg in sorted(
            (payload.get("moleculeSvgs", {}) or {}).items(),
            key=lambda item: int(item[0]),
        )
    )
    target_smiles = payload.get("targetSmiles", "")
    target_svg = payload.get("targetSvg", "")
    node_radius = float(payload["nodeRadius"])

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SynPlanner Tree Expansion Timeline</title>
    <style>
      :root {{
        color-scheme: dark;
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
        --active-stroke: #ffffff;
        --active-glow: rgba(255, 255, 255, 0.35);
        --map-center: #0f1623;
        --map-edge: #0b0f14;
        --divider: #1a2437;
        --control-bg: #1a2332;
        --control-border: #263247;
        --control-border-hover: #3a4b68;
        --stat-bg: #0f1522;
        --panel-shadow: rgba(0, 0, 0, 0.35);
        --molecule-panel-bg: #ffffff;
      }}

      :root[data-theme="light"] {{
        color-scheme: light;
        --bg: #eef3fb;
        --panel: #f8fbff;
        --panel-border: #cdd9ea;
        --text: #132034;
        --muted: #59708d;
        --edge: #aebdd3;
        --edge-visible: #6d809d;
        --edge-winning: #c88a00;
        --edge-winning-glow: rgba(200, 138, 0, 0.25);
        --node-stroke: #f8fbff;
        --active-stroke: #132034;
        --active-glow: rgba(19, 32, 52, 0.18);
        --map-center: #ffffff;
        --map-edge: #dbe7f5;
        --divider: #d7e1ee;
        --control-bg: #ffffff;
        --control-border: #bccbdd;
        --control-border-hover: #8ca0bb;
        --stat-bg: #edf4fb;
        --panel-shadow: rgba(45, 71, 110, 0.12);
        --molecule-panel-bg: #ffffff;
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

      #tree {{
        width: 100%;
        height: 100%;
        display: block;
        background: radial-gradient(1200px 900px at 50% 45%, var(--map-center) 0%, var(--map-edge) 70%);
        cursor: grab;
        touch-action: none;
      }}

      #tree.dragging {{
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
        stroke: var(--active-stroke);
        stroke-width: {min(max(node_radius * 0.6, 1.2), 10.0):.3f};
        filter: drop-shadow(0 0 6px var(--active-glow));
      }}

      .panel {{
        position: absolute;
        background: color-mix(in srgb, var(--panel) 86%, transparent);
        border: 1px solid var(--panel-border);
        border-radius: 10px;
        box-shadow: 0 10px 30px var(--panel-shadow);
        backdrop-filter: blur(6px);
      }}

      #controls {{
        top: 14px;
        left: 14px;
        padding: 0;
        width: min(620px, calc(100vw - 28px));
        overflow: hidden;
      }}

      #controls.collapsed #controls-body {{
        display: none;
      }}

      #hide-panel {{
        left: 14px;
        bottom: 14px;
        padding: 8px 10px;
        z-index: 2;
      }}

      #hide-panel button.active {{
        border-color: var(--edge-winning);
        color: var(--edge-winning);
      }}

      #hide-panel button:disabled {{
        opacity: 0.6;
        cursor: not-allowed;
      }}

      #recenter-panel {{
        left: 50%;
        bottom: 14px;
        transform: translateX(-50%);
        padding: 8px 10px;
        z-index: 2;
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

      #strategic-panel {{
        left: 14px;
        bottom: 64px;
        padding: 0;
        width: min(360px, calc(100vw - 28px));
        max-height: min(48vh, 460px);
        overflow: hidden;
        z-index: 2;
        display: none;
      }}

      #strategic-panel.open {{
        display: block;
      }}

      #strategic-panel.collapsed #strategic-body {{
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
      #strategic-panel .panel-header,
      #winning-panel .panel-header,
      #info .panel-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 10px 12px;
        border-bottom: 1px solid var(--divider);
      }}

      #controls.collapsed .panel-header,
      #strategic-panel.collapsed .panel-header,
      #winning-panel.collapsed .panel-header,
      #info.collapsed .panel-header {{
        border-bottom: none;
      }}

      .panel-actions {{
        display: flex;
        align-items: center;
        gap: 8px;
      }}

      #controls .panel-header h3,
      #strategic-panel .panel-header h3,
      #winning-panel .panel-header h3,
      #info .panel-header h3 {{
        margin: 0;
        font-size: 14px;
      }}

      #controls .panel-header button,
      #strategic-panel .panel-header button,
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

      #strategic-body {{
        padding: 10px 12px 12px 12px;
        max-height: calc(min(58vh, 520px) - 46px);
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

      .timeline-row {{
        flex-wrap: nowrap;
      }}

      .timeline-row #step-label {{
        min-width: 120px;
        text-align: right;
        color: var(--muted);
        font-size: 12px;
      }}

      .playback-row {{
        gap: 8px;
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
        background: var(--control-bg);
        color: var(--text);
        border: 1px solid var(--control-border);
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 12px;
        cursor: pointer;
      }}

      button:hover, select:hover {{
        border-color: var(--control-border-hover);
      }}

      .stat-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 6px;
        margin-top: 8px;
      }}

      .stat {{
        background: var(--stat-bg);
        border: 1px solid var(--divider);
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
      #strategic-panel .muted,
      #winning-panel .muted {{
        color: var(--muted);
        font-size: 12px;
      }}

      #info .kv,
      #strategic-panel .kv,
      #winning-panel .kv {{
        display: grid;
        grid-template-columns: 110px minmax(0, 1fr);
        gap: 4px 8px;
        font-size: 12px;
        margin-top: 6px;
      }}

      #info .kv div:nth-child(odd),
      #strategic-panel .kv div:nth-child(odd),
      #winning-panel .kv div:nth-child(odd) {{
        color: var(--muted);
      }}

      #target-svg-wrap {{
        margin-top: 8px;
        border: 1px solid var(--divider);
        border-radius: 8px;
        padding: 8px;
        background: var(--molecule-panel-bg);
        overflow: auto;
      }}

      #target-svg-wrap svg {{
        width: 100%;
        height: auto;
        display: block;
      }}

      .target-bond {{
        stroke: #9aa7b3;
        stroke-width: 0.08;
        stroke-linecap: round;
        cursor: pointer;
        pointer-events: stroke;
        transition: stroke 140ms ease, stroke-width 140ms ease, opacity 140ms ease;
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
        stroke: #4e8cff;
        stroke-width: 0.13;
      }}

      .target-bond.excluded {{
        opacity: 0.2;
      }}

      .target-bond.selected {{
        stroke: #ff5c5c;
        stroke-width: 0.18;
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

      #mol-svg {{
        margin-top: 8px;
        border: 1px solid var(--divider);
        border-radius: 8px;
        padding: 6px;
        background: var(--molecule-panel-bg);
        min-height: 84px;
        overflow: auto;
      }}

      #mol-svg svg {{
        width: min(100%, 320px);
        height: auto;
        display: block;
        margin: 0 auto;
      }}

      .edges line.filtered-out {{
        opacity: 0 !important;
        pointer-events: none;
      }}

      .nodes circle.filtered-out {{
        opacity: 0.05 !important;
        pointer-events: none;
      }}

      @media (max-width: 980px) {{
        #strategic-panel {{
          top: auto;
          bottom: 64px;
          left: 14px;
        }}
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
      <canvas id="tree" aria-label="Tree expansion timeline"></canvas>

      <div class="panel" id="controls">
        <div class="panel-header">
          <h3>Expansion Timeline</h3>
          <div class="panel-actions">
            <button id="theme-toggle" type="button" aria-label="Switch theme" title="Switch theme">☀</button>
            <button id="controls-toggle" type="button" aria-expanded="true">Collapse</button>
          </div>
        </div>
        <div id="controls-body">
          <div class="row timeline-row">
            <label for="step">Step</label>
            <input id="step" type="range" min="0" value="0" />
            <span id="step-label">0</span>
          </div>
          <div class="row playback-row">
            <label>Playback</label>
            <button id="play">Play</button>
            <button id="reset">Reset</button>
            <select id="speed" aria-label="Playback speed">
              <option value="40">Very Fast</option>
              <option value="80" selected>Fast</option>
              <option value="140">Medium</option>
              <option value="220">Slow</option>
            </select>
            <button id="to-start" type="button">To start</button>
            <button id="to-end" type="button">To end</button>
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

      <div class="panel" id="strategic-panel" role="region" aria-live="polite">
        <div class="panel-header">
          <h3>Strategic Bonds</h3>
          <button id="strategic-toggle" type="button" aria-expanded="true">Collapse</button>
        </div>
        <div id="strategic-body">
          <div class="row">
            <label id="cluster-exact-label">
              <input id="cluster-exact" type="checkbox" />
              Exact
            </label>
          </div>
          <div class="muted" id="cluster-status">Clusters: not loaded</div>
          <div class="muted" id="selected-bonds">Selected bonds: []</div>
          <div id="target-svg-wrap">
            {target_svg if target_svg else "<div class='muted'>Target depiction unavailable.</div>"}
          </div>
        </div>
      </div>

      <div class="panel" id="hide-panel">
        <button id="hide-nonwinning-btn" type="button" aria-pressed="false">Hide</button>
      </div>

      <div class="panel" id="recenter-panel">
        <button id="recenter-btn" type="button">Recenter</button>
      </div>

      <div class="panel" id="info" role="dialog" aria-live="polite">
        <div class="panel-header">
          <h3>Node Details</h3>
          <button id="info-toggle" type="button" aria-expanded="true">Collapse</button>
        </div>
        <div id="info-body">
          <div class="muted">Click a node to inspect it.</div>
          <div class="kv" id="kv"></div>
          <div id="mol-svg"></div>
        </div>
      </div>
    </div>

    <script id="molecule-svg-store" type="text/plain">{molecule_svg_records}</script>
    <script>
      const nodes = {nodes_json};
      const edges = {edges_json};
      const stats = {stats_json};
      const winningPaths = {winning_paths_json};
      const steps = {steps_json};
      const viewBox = {view_box_json};
      const clusters = {clusters_json};
      const routeNodes = {route_nodes_json};
      const targetSmiles = {json.dumps(target_smiles, ensure_ascii=True)};

      const canvas = document.getElementById("tree");
      const ctx = canvas.getContext("2d", {{ alpha: true }});
      const stepInput = document.getElementById("step");
      const stepLabel = document.getElementById("step-label");
      const playBtn = document.getElementById("play");
      const resetBtn = document.getElementById("reset");
      const speedSel = document.getElementById("speed");
      const toStartBtn = document.getElementById("to-start");
      const toEndBtn = document.getElementById("to-end");
      const controlsEl = document.getElementById("controls");
      const controlsToggleBtn = document.getElementById("controls-toggle");
      const themeToggleBtn = document.getElementById("theme-toggle");
      const statsEl = document.getElementById("stats");
      const hidePanelEl = document.getElementById("hide-panel");
      const hideNonWinningBtn = document.getElementById("hide-nonwinning-btn");
      const winningPanelEl = document.getElementById("winning-panel");
      const winningToggleBtn = document.getElementById("winning-toggle");
      const winningSummaryEl = document.getElementById("winning-summary");
      const winningKvEl = document.getElementById("winning-kv");
      const strategicPanelEl = document.getElementById("strategic-panel");
      const strategicToggleBtn = document.getElementById("strategic-toggle");
      const clusterStatusEl = document.getElementById("cluster-status");
      const clusterExactEl = document.getElementById("cluster-exact");
      const selectedBondsEl = document.getElementById("selected-bonds");
      const targetSvgWrapEl = document.getElementById("target-svg-wrap");
      const targetSvgEl = targetSvgWrapEl
        ? targetSvgWrapEl.querySelector("svg")
        : null;
      const infoEl = document.getElementById("info");
      const infoToggleBtn = document.getElementById("info-toggle");
      const kvEl = document.getElementById("kv");
      const molSvgEl = document.getElementById("mol-svg");
      const recenterBtn = document.getElementById("recenter-btn");

      const nodeRadiusWorld = {node_radius:.12f};
      const totalSteps = Math.max(0, stats.totalSteps || 0);
      stepInput.max = String(totalSteps);

      const rootEl = document.documentElement;
      const themeStorageKey = "synplanner-expansion-theme";

      function updateThemeToggle(theme) {{
        if (!themeToggleBtn) return;
        const nextIsLight = theme !== "light";
        themeToggleBtn.textContent = nextIsLight ? "☀" : "🌙";
        themeToggleBtn.setAttribute(
          "aria-label",
          nextIsLight ? "Switch to light mode" : "Switch to dark mode"
        );
        themeToggleBtn.title = nextIsLight
          ? "Switch to light mode"
          : "Switch to dark mode";
      }}

      function applyTheme(theme) {{
        const nextTheme = theme === "light" ? "light" : "dark";
        rootEl.setAttribute("data-theme", nextTheme);
        updateThemeToggle(nextTheme);
        try {{
          window.localStorage.setItem(themeStorageKey, nextTheme);
        }} catch (err) {{
          // Ignore localStorage failures in restricted contexts.
        }}
      }}

      function getInitialTheme() {{
        try {{
          const storedTheme = window.localStorage.getItem(themeStorageKey);
          if (storedTheme === "light" || storedTheme === "dark") {{
            return storedTheme;
          }}
        }} catch (err) {{
          // Ignore localStorage failures in restricted contexts.
        }}
        return "dark";
      }}

      applyTheme(getInitialTheme());

      const defaultViewState = {{
        x: Number(viewBox.minX),
        y: Number(viewBox.minY),
        width: Number(viewBox.width),
        height: Number(viewBox.height),
      }};
      let currentViewState = {{ ...defaultViewState }};

      function applyViewState() {{
        requestDraw();
      }}

      function clientToWorld(clientX, clientY) {{
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) {{
          return {{
            x: currentViewState.x + currentViewState.width / 2,
            y: currentViewState.y + currentViewState.height / 2,
          }};
        }}

        const relX = (clientX - rect.left) / rect.width;
        const relY = (clientY - rect.top) / rect.height;
        return {{
          x: currentViewState.x + relX * currentViewState.width,
          y: currentViewState.y + relY * currentViewState.height,
        }};
      }}

      function setViewState(nextState) {{
        currentViewState = {{
          x: Number(nextState.x),
          y: Number(nextState.y),
          width: Number(nextState.width),
          height: Number(nextState.height),
        }};
        applyViewState();
      }}

      function recenterView() {{
        setViewState(defaultViewState);
      }}

      function zoomAtClient(clientX, clientY, zoomFactor) {{
        const oldWidth = currentViewState.width;
        const oldHeight = currentViewState.height;
        const focus = clientToWorld(clientX, clientY);
        const relX =
          oldWidth > 0 ? (focus.x - currentViewState.x) / oldWidth : 0.5;
        const relY =
          oldHeight > 0 ? (focus.y - currentViewState.y) / oldHeight : 0.5;

        const minScale = 0.04;
        const maxScale = 8.0;
        const minWidth = defaultViewState.width * minScale;
        const maxWidth = defaultViewState.width * maxScale;
        const minHeight = defaultViewState.height * minScale;
        const maxHeight = defaultViewState.height * maxScale;

        const nextWidth = Math.max(
          minWidth,
          Math.min(maxWidth, oldWidth * zoomFactor)
        );
        const nextHeight = Math.max(
          minHeight,
          Math.min(maxHeight, oldHeight * zoomFactor)
        );

        setViewState({{
          x: focus.x - relX * nextWidth,
          y: focus.y - relY * nextHeight,
          width: nextWidth,
          height: nextHeight,
        }});
      }}

      const nodeMeta = new Map(nodes.map((n) => [n.id, n]));

      const nodesByStepMeta = Array.from({{ length: totalSteps + 1 }}, () => []);
      const edgesByStepMeta = Array.from({{ length: totalSteps + 1 }}, () => []);
      const edgeMetaByKey = new Map();

      function stepIndex(value) {{
        const num = Number.isFinite(value) ? value : parseInt(value, 10);
        if (!Number.isFinite(num)) return 0;
        return Math.max(0, Math.min(totalSteps, Math.trunc(num)));
      }}

      for (const node of nodes) {{
        nodesByStepMeta[stepIndex(node.step)].push(node);
      }}
      for (const edge of edges) {{
        const edgeKey = String(edge.key || `${{edge.parent}}->${{edge.child}}`);
        edgeMetaByKey.set(edgeKey, edge);
        edgesByStepMeta[stepIndex(edge.step)].push(edge);
      }}

      function statCard(label, value) {{
        const div = document.createElement("div");
        div.className = "stat";
        div.innerHTML = `<div class="label">${{label}}</div><div class="value">${{value}}</div>`;
        return div;
      }}

      statsEl.appendChild(statCard("Tree Nodes", stats.totalNodesInTree));
      statsEl.appendChild(statCard("Rendered", stats.renderedNodes));
      statsEl.appendChild(statCard("Max Depth", stats.maxDepth));
      statsEl.appendChild(statCard("Steps", stats.totalSteps));

      let currentStep = -1;
      let playing = false;
      let timer = null;
      let activeNodeId = null;
      let infoCollapsed = false;
      let infoOpen = false;
      let winningCollapsed = false;
      let strategicCollapsed = false;
      let controlsCollapsed = false;
      let dragState = null;
      let suppressClickAfterDrag = false;
      let activeWinningEdgeKeys = new Set();
      let activeWinningNodeIds = new Set();

      const winningPathsData = Array.isArray(winningPaths) ? winningPaths : [];
      const parentById = new Map(nodes.map((n) => [n.id, Number(n.parent || 0)]));
      const clustersLoaded = Array.isArray(clusters) && clusters.length > 0;
      const selectedBondIds = new Set();
      let clusterFilteredTreeNodes = null;
      let clusterFilteredTreeEdges = null;
      const allClusterBondKeys = new Set();
      const routeTreeEdgeKeys = new Map();

      function normalizeBondKey(pair) {{
        if (!Array.isArray(pair) || pair.length < 2) return "";
        const a = Number(pair[0]);
        const b = Number(pair[1]);
        if (!Number.isFinite(a) || !Number.isFinite(b)) return "";
        return a < b ? `${{a}}-${{b}}` : `${{b}}-${{a}}`;
      }}

      clusters.forEach((cluster) => {{
        const bonds = Array.isArray(cluster.bonds) ? cluster.bonds : [];
        cluster.bondKeys = bonds
          .map((pair) => normalizeBondKey(pair))
          .filter((key) => key);
        cluster.bondKeys.forEach((key) => allClusterBondKeys.add(key));
      }});

      Object.entries(routeNodes || {{}}).forEach(([routeId, path]) => {{
        if (!Array.isArray(path)) return;
        const edgeKeys = new Set();
        for (let i = 0; i < path.length - 1; i += 1) {{
          const parent = Number(path[i + 1]);
          const child = Number(path[i]);
          if (Number.isFinite(parent) && Number.isFinite(child)) {{
            edgeKeys.add(`${{parent}}->${{child}}`);
          }}
        }}
        routeTreeEdgeKeys.set(String(routeId), edgeKeys);
      }});

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

      let hideNonWinning = false;

      function updateHideButton() {{
        if (!hideNonWinningBtn) return;
        hideNonWinningBtn.textContent = hideNonWinning ? "Show" : "Hide";
        hideNonWinningBtn.classList.toggle("active", hideNonWinning);
        hideNonWinningBtn.setAttribute("aria-pressed", String(hideNonWinning));
      }}

      function shouldHideNonWinningNode(nodeId) {{
        if (!hideNonWinning) return false;
        const meta = nodeMeta.get(nodeId);
        if (!meta) return false;
        return meta.status === "intermediate" && !winningEdgeConnectedNodeIds.has(nodeId);
      }}

      function shouldHideNonWinningEdge(edgeKey) {{
        if (!hideNonWinning) return false;
        const meta = edgeMetaByKey.get(String(edgeKey));
        if (!meta) return false;
        return shouldHideNonWinningNode(meta.parent) || shouldHideNonWinningNode(meta.child);
      }}

      function cssVar(name) {{
        return getComputedStyle(rootEl).getPropertyValue(name).trim();
      }}

      function palette() {{
        return {{
          edge: cssVar("--edge-visible") || "#5a6c8a",
          edgeWinning: cssVar("--edge-winning") || "#ffd166",
          target: cssVar("--target") || "#4e8cff",
          leaf: cssVar("--leaf") || "#4cd97b",
          intermediate: cssVar("--intermediate") || "#ff5c5c",
          nodeStroke: cssVar("--node-stroke") || "#0b0f14",
          activeStroke: cssVar("--active-stroke") || "#ffffff",
        }};
      }}

      let canvasCssWidth = 1;
      let canvasCssHeight = 1;
      let drawQueued = false;

      function resizeCanvas() {{
        const rect = canvas.getBoundingClientRect();
        const dpr = Math.max(1, window.devicePixelRatio || 1);
        canvasCssWidth = Math.max(1, rect.width || window.innerWidth || 1);
        canvasCssHeight = Math.max(1, rect.height || window.innerHeight || 1);
        const nextWidth = Math.max(1, Math.round(canvasCssWidth * dpr));
        const nextHeight = Math.max(1, Math.round(canvasCssHeight * dpr));
        if (canvas.width !== nextWidth || canvas.height !== nextHeight) {{
          canvas.width = nextWidth;
          canvas.height = nextHeight;
        }}
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        requestDraw();
      }}

      function worldToCanvas(x, y) {{
        return {{
          x: ((x - currentViewState.x) / currentViewState.width) * canvasCssWidth,
          y: ((y - currentViewState.y) / currentViewState.height) * canvasCssHeight,
        }};
      }}

      function worldScaleToCanvas() {{
        if (!currentViewState.width || !currentViewState.height) return 1;
        const xScale = canvasCssWidth / currentViewState.width;
        const yScale = canvasCssHeight / currentViewState.height;
        return Math.max(0.0001, (xScale + yScale) / 2);
      }}

      function isNodeClusterHidden(meta) {{
        if (!(clusterFilteredTreeNodes instanceof Set)) return false;
        const treeId = meta ? Number(meta.tree_id || meta.id) : NaN;
        return !clusterFilteredTreeNodes.has(treeId);
      }}

      function isEdgeClusterHidden(meta) {{
        if (!(clusterFilteredTreeEdges instanceof Set)) return false;
        const treeKey = meta ? String(meta.tree_key || meta.key || "") : "";
        return !clusterFilteredTreeEdges.has(treeKey);
      }}

      function isNodeVisible(meta) {{
        if (!meta) return false;
        if (stepIndex(meta.step) > currentStep) return false;
        if (shouldHideNonWinningNode(meta.id)) return false;
        if (isNodeClusterHidden(meta)) return false;
        return true;
      }}

      function isEdgeVisible(edge) {{
        if (!edge) return false;
        if (stepIndex(edge.step) > currentStep) return false;
        const edgeKey = String(edge.key || `${{edge.parent}}->${{edge.child}}`);
        if (shouldHideNonWinningEdge(edgeKey)) return false;
        if (isEdgeClusterHidden(edge)) return false;
        return true;
      }}

      function nodeFill(meta, colors) {{
        if (meta.status === "target") return colors.target;
        if (meta.status === "leaf") return colors.leaf;
        return colors.intermediate;
      }}

      function drawCanvas() {{
        if (!ctx) return;
        ctx.clearRect(0, 0, canvasCssWidth, canvasCssHeight);
        const colors = palette();
        const scale = worldScaleToCanvas();
        const nodeRadiusPx = Math.max(2.2, Math.min(18, nodeRadiusWorld * scale));
        const edgeWidthPx = Math.max(0.65, Math.min(5.5, nodeRadiusPx * 0.28));
        const winningWidthPx = Math.max(1.1, Math.min(8, nodeRadiusPx * 0.45));

        ctx.lineCap = "round";
        for (const edge of edges) {{
          if (!isEdgeVisible(edge)) continue;
          const parent = nodeMeta.get(edge.parent);
          const child = nodeMeta.get(edge.child);
          if (!parent || !child) continue;
          const parentPoint = worldToCanvas(parent.x, parent.y);
          const childPoint = worldToCanvas(child.x, child.y);
          const edgePad = Math.max(12, winningWidthPx * 2);
          if (
            Math.max(parentPoint.x, childPoint.x) < -edgePad ||
            Math.min(parentPoint.x, childPoint.x) > canvasCssWidth + edgePad ||
            Math.max(parentPoint.y, childPoint.y) < -edgePad ||
            Math.min(parentPoint.y, childPoint.y) > canvasCssHeight + edgePad
          ) {{
            continue;
          }}
          const edgeKey = String(edge.key || `${{edge.parent}}->${{edge.child}}`);
          const isWinning = activeWinningEdgeKeys.has(edgeKey);
          ctx.beginPath();
          ctx.moveTo(parentPoint.x, parentPoint.y);
          ctx.lineTo(childPoint.x, childPoint.y);
          ctx.strokeStyle = isWinning ? colors.edgeWinning : colors.edge;
          ctx.globalAlpha = isWinning ? 1 : 0.82;
          ctx.lineWidth = isWinning ? winningWidthPx : edgeWidthPx;
          ctx.stroke();
        }}

        ctx.globalAlpha = 1;
        for (const meta of nodes) {{
          if (!isNodeVisible(meta)) continue;
          const point = worldToCanvas(meta.x, meta.y);
          const nodePad = nodeRadiusPx + 2;
          if (
            point.x < -nodePad ||
            point.x > canvasCssWidth + nodePad ||
            point.y < -nodePad ||
            point.y > canvasCssHeight + nodePad
          ) {{
            continue;
          }}
          ctx.beginPath();
          ctx.arc(point.x, point.y, nodeRadiusPx, 0, Math.PI * 2);
          ctx.fillStyle = nodeFill(meta, colors);
          ctx.fill();
          ctx.strokeStyle =
            activeNodeId === meta.id || activeWinningNodeIds.has(meta.id)
              ? colors.activeStroke
              : colors.nodeStroke;
          ctx.lineWidth =
            activeNodeId === meta.id || activeWinningNodeIds.has(meta.id)
              ? Math.max(1.3, nodeRadiusPx * 0.28)
              : Math.max(0.8, nodeRadiusPx * 0.18);
          ctx.stroke();
        }}
      }}

      function requestDraw() {{
        if (drawQueued) return;
        drawQueued = true;
        window.requestAnimationFrame(() => {{
          drawQueued = false;
          drawCanvas();
        }});
      }}

      function updateHiddenNonWinningEdges() {{
        requestDraw();
      }}

      function updateHiddenNonWinningNodes() {{
        requestDraw();
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

      function setClusterStatus(message) {{
        if (clusterStatusEl) {{
          clusterStatusEl.textContent = message;
        }}
      }}

      function updateSelectedBondsDisplay() {{
        if (!selectedBondsEl) return;
        const selected = Array.from(selectedBondIds).sort();
        selectedBondsEl.textContent = `Selected bonds: [${{selected.join(", ")}}]`;
      }}

      function getMatchingClusters() {{
        if (!clustersLoaded || selectedBondIds.size === 0) return [];
        const exactMode = clusterExactEl && clusterExactEl.checked;
        const selected = Array.from(selectedBondIds);
        return clusters.filter((cluster) => {{
          const bondKeys = Array.isArray(cluster.bondKeys) ? cluster.bondKeys : [];
          if (exactMode && selected.length !== bondKeys.length) {{
            return false;
          }}
          return selected.every((bondId) => bondKeys.includes(bondId));
        }});
      }}

      function getColorClusters() {{
        if (!clustersLoaded || selectedBondIds.size === 0) return [];
        const selected = Array.from(selectedBondIds);
        return clusters.filter((cluster) => {{
          const bondKeys = Array.isArray(cluster.bondKeys) ? cluster.bondKeys : [];
          return selected.every((bondId) => bondKeys.includes(bondId));
        }});
      }}

      function setBondClass(bondId, className, enabled) {{
        if (!targetSvgEl) return;
        targetSvgEl
          .querySelectorAll('[data-bond="' + bondId + '"]')
          .forEach((el) => {{
            el.classList.toggle(className, Boolean(enabled));
          }});
      }}

      function updateBondColors() {{
        if (!targetSvgEl) return;
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
        targetSvgEl.querySelectorAll(".target-bond").forEach((bond) => {{
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

      function routeTreeNodes(routeId) {{
        const path = routeNodes[String(routeId)];
        if (!Array.isArray(path)) return [];
        return path
          .map((nodeId) => Number(nodeId))
          .filter((nodeId) => Number.isFinite(nodeId));
      }}

      function updateClusterVisibility() {{
        requestDraw();
      }}

      function clearClusterFilter() {{
        clusterFilteredTreeNodes = null;
        clusterFilteredTreeEdges = null;
        updateClusterVisibility();
      }}

      function applyClusterFilter() {{
        if (!clustersLoaded) {{
          clearClusterFilter();
          setClusterStatus("Clusters: not loaded");
          return;
        }}
        if (selectedBondIds.size === 0) {{
          clearClusterFilter();
          setClusterStatus("Clusters: no bonds selected");
          return;
        }}

        const matches = getMatchingClusters();
        if (matches.length === 0) {{
          clearClusterFilter();
          setClusterStatus(
            clusterExactEl && clusterExactEl.checked
              ? "No exact cluster match"
              : "Clusters matched: 0"
          );
          return;
        }}

        const matchedRouteIds = new Set();
        clusterFilteredTreeNodes = new Set([1]);
        clusterFilteredTreeEdges = new Set();

        matches.forEach((cluster) => {{
          (cluster.route_ids || []).forEach((routeId) => {{
            const routeKey = String(routeId);
            matchedRouteIds.add(routeKey);
            routeTreeNodes(routeKey).forEach((nodeId) => {{
              clusterFilteredTreeNodes.add(nodeId);
            }});
            const edgeKeys = routeTreeEdgeKeys.get(routeKey);
            if (edgeKeys) {{
              edgeKeys.forEach((key) => clusterFilteredTreeEdges.add(key));
            }}
          }});
        }});

        updateClusterVisibility();
        setClusterStatus(
          `Clusters matched: ${{matches.length}} · routes ${{matchedRouteIds.size}}`
        );
      }}

      function syncStrategicBondSelection() {{
        updateSelectedBondsDisplay();
        updateBondColors();
        applyClusterFilter();
      }}

      function bindStrategicBonds() {{
        if (!targetSvgEl) return;
        targetSvgEl.querySelectorAll(".target-bond").forEach((bond) => {{
          bond.addEventListener("click", (event) => {{
            event.preventDefault();
            event.stopPropagation();
            const bondId = bond.dataset.bond;
            if (!bondId) return;
            if (selectedBondIds.has(bondId)) {{
              selectedBondIds.delete(bondId);
            }} else {{
              selectedBondIds.add(bondId);
            }}
            syncStrategicBondSelection();
          }});
        }});
      }}

      if (strategicPanelEl && clustersLoaded) {{
        strategicPanelEl.classList.add("open");
      }}
      bindStrategicBonds();
      updateSelectedBondsDisplay();
      updateBondColors();
      setClusterStatus(
        clustersLoaded
          ? "Clusters: no bonds selected"
          : "Clusters: not loaded"
      );
      if (clusterExactEl) {{
        clusterExactEl.addEventListener("change", () => {{
          updateBondColors();
          applyClusterFilter();
        }});
      }}

      function hitTestNode(clientX, clientY) {{
        const world = clientToWorld(clientX, clientY);
        const rect = canvas.getBoundingClientRect();
        const pxToWorld = rect.width ? currentViewState.width / rect.width : 1;
        const hitRadius = Math.max(nodeRadiusWorld * 1.8, pxToWorld * 8);
        const hitRadiusSq = hitRadius * hitRadius;
        let bestNode = null;
        let bestDistSq = hitRadiusSq;
        for (let idx = nodes.length - 1; idx >= 0; idx -= 1) {{
          const meta = nodes[idx];
          if (!isNodeVisible(meta)) continue;
          const dx = Number(meta.x) - world.x;
          const dy = Number(meta.y) - world.y;
          const distSq = dx * dx + dy * dy;
          if (distSq <= bestDistSq) {{
            bestDistSq = distSq;
            bestNode = meta;
          }}
        }}
        return bestNode;
      }}

      canvas.addEventListener(
        "wheel",
        (event) => {{
          event.preventDefault();
          const zoomFactor = event.deltaY < 0 ? 0.9 : 1.1;
          zoomAtClient(event.clientX, event.clientY, zoomFactor);
        }},
        {{ passive: false }}
      );

      canvas.addEventListener("pointerdown", (event) => {{
        if (event.button !== 0) return;
        dragState = {{
          pointerId: event.pointerId,
          startClientX: event.clientX,
          startClientY: event.clientY,
          originX: currentViewState.x,
          originY: currentViewState.y,
          moved: false,
        }};
        canvas.classList.add("dragging");
        canvas.setPointerCapture(event.pointerId);
      }});

      canvas.addEventListener("pointermove", (event) => {{
        if (!dragState || dragState.pointerId !== event.pointerId) return;

        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;

        const dxClient = event.clientX - dragState.startClientX;
        const dyClient = event.clientY - dragState.startClientY;
        if (Math.abs(dxClient) > 3 || Math.abs(dyClient) > 3) {{
          dragState.moved = true;
        }}

        const dxWorld = (dxClient / rect.width) * currentViewState.width;
        const dyWorld = (dyClient / rect.height) * currentViewState.height;
        setViewState({{
          x: dragState.originX - dxWorld,
          y: dragState.originY - dyWorld,
          width: currentViewState.width,
          height: currentViewState.height,
        }});
      }});

      function stopDragging(event) {{
        if (!dragState) return;
        const wasMoved = Boolean(dragState.moved);
        if (event && dragState.pointerId === event.pointerId) {{
          try {{
            canvas.releasePointerCapture(event.pointerId);
          }} catch (err) {{
            // Ignore stale pointer-capture release failures.
          }}
        }}
        dragState = null;
        canvas.classList.remove("dragging");

        if (event && !wasMoved) {{
          const nodeHit = hitTestNode(event.clientX, event.clientY);
          if (nodeHit) {{
            setActiveNode(Number(nodeHit.id));
          }}
        }}

        suppressClickAfterDrag = wasMoved;
      }}

      canvas.addEventListener("pointerup", stopDragging);
      canvas.addEventListener("pointercancel", stopDragging);
      canvas.addEventListener(
        "click",
        (event) => {{
          if (!suppressClickAfterDrag) return;
          event.preventDefault();
          event.stopPropagation();
          suppressClickAfterDrag = false;
        }},
        true
      );

      if (recenterBtn) {{
        recenterBtn.addEventListener("click", () => {{
          recenterView();
        }});
      }}

      if (themeToggleBtn) {{
        themeToggleBtn.addEventListener("click", () => {{
          const currentTheme =
            rootEl.getAttribute("data-theme") === "light" ? "light" : "dark";
          applyTheme(currentTheme === "light" ? "dark" : "light");
          requestDraw();
        }});
      }}

      function setStrategicCollapsed(nextCollapsed) {{
        if (!strategicPanelEl || !strategicToggleBtn) return;
        strategicCollapsed = Boolean(nextCollapsed);
        strategicPanelEl.classList.toggle("collapsed", strategicCollapsed);
        strategicToggleBtn.textContent = strategicCollapsed ? "Expand" : "Collapse";
        strategicToggleBtn.setAttribute("aria-expanded", String(!strategicCollapsed));
      }}

      if (strategicToggleBtn) {{
        strategicToggleBtn.addEventListener("click", () => {{
          setStrategicCollapsed(!strategicCollapsed);
        }});
        setStrategicCollapsed(false);
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

      function showStep(step) {{
        currentStep = clampStep(step);
        requestDraw();
      }}

      function hideStep(step) {{
        currentStep = clampStep(step - 1);
        requestDraw();
      }}

      function setStep(nextStep) {{
        const target = clampStep(nextStep);
        if (target === currentStep) return;

        currentStep = target;
        stepInput.value = String(target);
        const rendered = steps.cumulative?.[target] ?? "";
        stepLabel.textContent = `${{target}} / ${{totalSteps}} · nodes ${{rendered}}`;
        updateWinningHighlights(currentStep);
        updateHiddenNonWinningNodes();
        requestDraw();
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

      toStartBtn.addEventListener("click", () => {{
        stopPlaying();
        setStep(0);
      }});

      toEndBtn.addEventListener("click", () => {{
        stopPlaying();
        setStep(totalSteps);
      }});

      stepInput.addEventListener("input", (e) => {{
        stopPlaying();
        const value = parseInt(e.target.value, 10);
        setStep(value);
      }});

      speedSel.addEventListener("change", () => {{
        if (playing) {{
          stopPlaying();
          startPlaying();
        }}
      }});

      function setActiveNode(nodeId) {{
        activeNodeId = nodeId;
        ensureInfoOpen();
        renderNodeDetails(nodeMeta.get(nodeId));
        requestDraw();
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
        const active = computeActiveWinning(step);
        activeWinningEdgeKeys = active.activeEdgeKeys;
        activeWinningNodeIds = active.activeWinningNodeIds;
        updateWinningPanel(step, activeWinningNodeIds.size, activeWinningEdgeKeys.size);
        requestDraw();
      }}

      const moleculeSvgStoreEl = document.getElementById("molecule-svg-store");
      const moleculeSvgCache = new Map();
      let moleculeSvgStoreText = null;

      function decodeBase64Utf8(encoded) {{
        try {{
          const binary = window.atob(encoded);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i += 1) {{
            bytes[i] = binary.charCodeAt(i);
          }}
          return new TextDecoder("utf-8").decode(bytes);
        }} catch (err) {{
          return "";
        }}
      }}

      function moleculeSvgForNode(nodeId) {{
        const key = String(nodeId);
        if (!/^\\d+$/.test(key)) {{
          return "";
        }}
        if (moleculeSvgCache.has(key)) {{
          return moleculeSvgCache.get(key);
        }}
        if (moleculeSvgStoreText === null) {{
          moleculeSvgStoreText = moleculeSvgStoreEl
            ? moleculeSvgStoreEl.textContent || ""
            : "";
        }}
        if (!moleculeSvgStoreText) {{
          moleculeSvgCache.set(key, "");
          return "";
        }}
        const pattern = new RegExp("(?:^|\\n)" + key + "\\t([^\\n]*)");
        const match = moleculeSvgStoreText.match(pattern);
        const svgMarkup = match ? decodeBase64Utf8(match[1]) : "";
        moleculeSvgCache.set(key, svgMarkup);
        return svgMarkup;
      }}

      function buildDisplaySvg(svgMarkup) {{
        if (!svgMarkup) return null;
        try {{
          const doc = new DOMParser().parseFromString(svgMarkup, "image/svg+xml");
          const svgEl = doc.documentElement;
          if (!svgEl || svgEl.tagName.toLowerCase() !== "svg") return null;

          const parserError = svgEl.querySelector("parsererror");
          if (parserError) return null;

          const rect = doc.createElementNS("http://www.w3.org/2000/svg", "rect");
          rect.setAttribute("width", "100%");
          rect.setAttribute("height", "100%");
          rect.setAttribute("fill", "#ffffff");
          rect.setAttribute("pointer-events", "none");
          svgEl.insertBefore(rect, svgEl.firstChild);

          const style = svgEl.getAttribute("style") || "";
          svgEl.setAttribute(
            "style",
            `${{style}};background:#ffffff;max-width:100%;height:auto;display:block;margin:0 auto;`
          );
          svgEl.setAttribute("preserveAspectRatio", "xMidYMid meet");
          svgEl.removeAttribute("width");
          svgEl.removeAttribute("height");

          return document.importNode(svgEl, true);
        }} catch (err) {{
          return null;
        }}
      }}

      function renderMoleculeSvg(svgMarkup) {{
        molSvgEl.replaceChildren();
        if (!svgMarkup) return false;

        const svgNode = buildDisplaySvg(svgMarkup);
        if (svgNode) {{
          molSvgEl.appendChild(svgNode);
          return true;
        }}

        try {{
          molSvgEl.innerHTML = svgMarkup;
          return true;
        }} catch (err) {{
          molSvgEl.textContent = "Molecule depiction unavailable.";
          return false;
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
          ["Rule Label", safe(meta.rule_label)],
          ["SMILES", safe(meta.smiles)],
          ["Status", meta.status],
        ];
        kvEl.innerHTML = rows
          .map(([k, v]) => `<div>${{k}}</div><div>${{safe(v)}}</div>`)
          .join("");

        const svgMarkup = moleculeSvgForNode(meta.id);
        if (svgMarkup) {{
          renderMoleculeSvg(svgMarkup);
        }} else {{
          molSvgEl.textContent = "Molecule depiction unavailable.";
        }}
      }}

      // Initialize at step 0 and focus the root if present.
      window.addEventListener("resize", resizeCanvas);
      resizeCanvas();
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
        default=1.0,
        help="Base radial step between depths (default: 1.0).",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=220.0,
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
    parser.add_argument(
        "--clusters",
        type=Path,
        default=None,
        help="Optional path to a clusters pickle for Strategic Bond Search Mode.",
    )

    args = parser.parse_args()

    tree = _load_tree(args.tree)
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
        clusters_pkl=args.clusters,
    )
    print(f"Wrote expansion timeline HTML to: {args.output}")


if __name__ == "__main__":
    main()
