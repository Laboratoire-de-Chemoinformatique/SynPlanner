"""Module containing functions for analysis and visualization of the built tree."""

import base64
import contextlib
from collections import deque
from datetime import datetime
from itertools import count, islice, pairwise
from typing import Any

from chython import depict_settings
from chython import smiles as read_smiles
from chython.algorithms.depict import _graph_svg, _render_config
from chython.containers.molecule import MoleculeContainer
from IPython.display import HTML, display

from synplan.chem.reaction_routes.io import make_dict
from synplan.chem.reaction_routes.visualisation import (
    cgr_display,
    depict_custom_reaction,
)
from synplan.mcts.tree import Tree


def get_child_nodes(
    tree: Tree,
    molecule: MoleculeContainer,
    graph: dict[MoleculeContainer, dict[str, Any]],
) -> dict[str, Any]:
    """Extracts the child nodes of the given molecule.

    :param tree: The built tree.
    :param molecule: The molecule in the tree from which to extract child nodes.
    :param graph: The relationship between the given molecule and the reaction
        metadata for its child nodes.
    :return: The dict with extracted child nodes.
    """

    reaction = graph.get(molecule)
    if reaction is None:
        return []

    nodes = []
    for precursor in reaction["children"]:
        temp_obj = {
            "smiles": str(precursor),
            "type": "mol",
            "in_stock": str(precursor) in tree.building_blocks,
        }
        node = get_child_nodes(tree, precursor, graph)
        if node:
            temp_obj["children"] = [node]
        nodes.append(temp_obj)

    reaction_node = {"type": "reaction", "children": nodes}
    if reaction.get("rule_key"):
        reaction_node["rule_key"] = reaction["rule_key"]
    if reaction.get("policy_rank") is not None:
        reaction_node["policy_rank"] = reaction["policy_rank"]
    return reaction_node


def extract_routes(
    tree: Tree, extended: bool = False, min_mol_size: int = 0
) -> list[dict[str, Any]]:
    """Takes the target and the dictionary of successors and predecessors and returns a
    list of dictionaries that contain the target and the list of successors.

    :param tree: The built tree.
    :param extended: If True, generates the extended route representation.
    :param min_mol_size: If the size of the Precursor is equal or smaller than
            min_mol_size it is automatically classified as building block.
    :return: A list of dictionaries. Each dictionary contains a target, a list of
        children, and a boolean indicating whether the target is in building_blocks.
    """
    target = tree.nodes[1].precursors_to_expand[0].molecule
    target_in_stock = tree.nodes[1].curr_precursor.is_building_block(
        tree.building_blocks, min_mol_size
    )

    # append encoded routes to list
    routes_block = []
    winning_nodes = []
    if extended:
        # collect routes
        for i, node in tree.nodes.items():
            if node.is_solved():
                winning_nodes.append(i)
    else:
        winning_nodes = tree.winning_nodes
    if winning_nodes:
        for winning_node in winning_nodes:
            # Create graph for route
            graph = {}
            path_ids = []
            nid = winning_node
            while nid:
                path_ids.append(nid)
                nid = tree.parents[nid]
            path_ids.reverse()

            for before_id, after_id in pairwise(path_ids):
                before = tree.nodes[before_id].curr_precursor.molecule
                graph[before] = {
                    "children": [
                        precursor.molecule
                        for precursor in tree.nodes[after_id].new_precursors
                    ],
                    "rule_key": tree.nodes_rule_key.get(after_id),
                    "policy_rank": tree.nodes_policy_rank.get(after_id),
                }

            routes_block.append(
                {
                    "type": "mol",
                    "smiles": str(target),
                    "in_stock": target_in_stock,
                    "children": [get_child_nodes(tree, target, graph)],
                }
            )
    else:
        routes_block = [
            {
                "type": "mol",
                "smiles": str(target),
                "in_stock": target_in_stock,
                "children": [],
            }
        ]
    return routes_block


def render_svg(pred, columns, box_colors, labeled: bool = False):
    """
    Renders an SVG representation of a retrosynthetic route.

    This function takes the predicted reaction steps, the molecules organized
    into columns representing reaction stages, and a mapping of molecule status
    to box colors, and generates an SVG string visualizing the route. It
    calculates positions for molecules and arrows, and constructs the SVG
    elements.

    Args:
        pred (tuple): A tuple of tuples representing the predicted reaction
                      steps. Each inner tuple is (source_molecule_index,
                      target_molecule_index). The indices correspond to the
                      flattened list of molecules across all columns.
        columns (list): A list of lists, where each inner list contains
                        Molecule objects for a specific stage (column) in the
                        retrosynthetic route.
        box_colors (dict): A dictionary mapping molecule status strings (e.g.,
                          'target', 'mulecule', 'instock') to SVG color strings
                          for the boxes around the molecules.
        labeled (bool): If True, upstream preparation may include the full
                        ``rule_key`` in the rendered arrow labels.

    Returns:
        str: A string containing the complete SVG code for the retrosynthetic
             route visualization.
    """
    x_shift = 0.0
    c_max_x = 0.0
    c_max_y = 0.0
    render = []
    flat_molecules = []
    cx = count()
    cy = count()
    arrow_points = {}

    def _get_plane(mol):
        return {n: (a.x, a.y) for n, a in mol._atoms.items()}

    def _set_plane(mol, plane):
        for n, (x, y) in plane.items():
            mol._atoms[n].xy = (x, y)

    for ms in columns:
        heights = []
        for m in ms:
            flat_molecules.append(m)
            m.clean2d()
            plane = _get_plane(m)
            # X-shift for target
            min_x = min(x for x, y in plane.values()) - x_shift
            min_y = min(y for x, y in plane.values())
            plane = {n: (x - min_x, y - min_y) for n, (x, y) in plane.items()}
            _set_plane(m, plane)
            max_x = max(x for x, y in plane.values())

            c_max_x = max(c_max_x, max_x)

            arrow_points[next(cx)] = [x_shift, max_x]
            heights.append(max(y for x, y in plane.values()))

        x_shift = c_max_x + 5.0  # between columns gap
        # calculate Y-shift
        y_shift = sum(heights) + 3.0 * (len(heights) - 1)

        c_max_y = max(c_max_y, y_shift)

        y_shift /= 2.0
        for m, h in zip(ms, heights):
            plane = {n: (x, y - y_shift) for n, (x, y) in _get_plane(m).items()}
            _set_plane(m, plane)

            # calculate coordinates for boxes
            max_x = max(x for x, y in plane.values()) + 0.9  # max x
            min_x = min(x for x, y in plane.values()) - 0.6  # min x
            max_y = -(max(y for x, y in plane.values()) + 0.45)  # max y
            min_y = -(min(y for x, y in plane.values()) - 0.45)  # min y
            x_delta = abs(max_x - min_x)
            y_delta = abs(max_y - min_y)
            box = (
                f'<rect x="{min_x}" y="{max_y}" rx="{y_delta * 0.1}" ry="{y_delta * 0.1}" width="{x_delta}" height="{y_delta}"'
                f' stroke="black" stroke-width=".0025" fill="{box_colors[m.meta["status"]]}" fill-opacity="0.30"/>'
            )
            arrow_points[next(cy)].append(y_shift - h / 2.0)
            y_shift -= h + 3.0
            atoms, bonds, define, masks, uid, *_ = m.depict(_embedding=True)
            depicted_molecule = [atoms, bonds, define, masks, uid]
            depicted_molecule.append(box)
            render.append(depicted_molecule)

    # calculate mid-X coordinate to draw square arrows
    graph = {}
    for s, p in pred:
        try:
            graph[s].append(p)
        except KeyError:
            graph[s] = [p]
    for s, ps in graph.items():
        mid_x = float("-inf")
        for p in ps:
            s_min_x, _s_max, s_y = arrow_points[s][:3]  # s
            _p_min_x, p_max, p_y = arrow_points[p][:3]  # p
            p_max += 1
            mid = p_max + (s_min_x - p_max) / 3
            mid_x = max(mid_x, mid)
        for p in ps:
            arrow_points[p].append(mid_x)

    config = _render_config
    font_size = config["font_size"]
    font125 = 1.25 * font_size
    width = c_max_x + 4.0 * font_size  # 3.0 by default
    height = c_max_y + 3.5 * font_size  # 2.5 by default
    box_y = height / 2.0
    svg = [
        f'<svg width="{0.6 * width:.2f}cm" height="{0.6 * height:.2f}cm" '
        f'viewBox="{-font125:.2f} {-box_y:.2f} {width:.2f} '
        f'{height:.2f}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">',
        '  <defs>\n    <marker id="arrow" markerWidth="10" markerHeight="10" '
        'refX="0" refY="3" orient="auto">\n      <path d="M0,0 L0,6 L9,3"/>\n    </marker>\n  </defs>',
    ]

    rendered_arrow_labels = set()
    arrow_label_elements = []
    for s, p in pred:
        s_min_x, _s_max, s_y = arrow_points[s][:3]
        _p_min_x, p_max, p_y = arrow_points[p][:3]
        p_max += 1
        mid_x = arrow_points[p][-1]  # p_max + (s_min_x - p_max) / 3
        arrow = f"""  <polyline points="{p_max:.2f} {p_y:.2f}, {mid_x:.2f} {p_y:.2f}, {mid_x:.2f} {s_y:.2f}, {s_min_x - 1.:.2f} {s_y:.2f}"
                fill="none" stroke="black" stroke-width=".04" marker-end="url(#arrow)"/>"""
        if p_y != s_y:
            arrow += f'  <circle cx="{mid_x}" cy="{p_y}" r="0.1"/>'
        svg.append(arrow)

        if s not in rendered_arrow_labels:
            label_text = flat_molecules[s].meta.get("label")
            if label_text:
                # Keep the rule label clear of the arrow head and bend.
                label_x = ((mid_x + (s_min_x - 1.0)) / 2.0) - 0.45
                label_y = s_y
                label_width = max(1.2, 0.18 * len(label_text) + 0.35)
                label_height = 0.5
                label_rect = (
                    f'  <rect x="{label_x - label_width / 2.0:.2f}" '
                    f'y="{label_y - label_height / 2.0:.2f}" '
                    f'width="{label_width:.2f}" height="{label_height:.2f}" '
                    'rx="0.08" ry="0.08" fill="white" stroke="black" '
                    'stroke-width=".02"/>'
                )
                label_svg = (
                    f'  <text x="{label_x:.2f}" y="{label_y:.2f}" '
                    'text-anchor="middle" dominant-baseline="central" '
                    'font-family="monospace" font-size="0.24" fill="black">'
                    f"{label_text}</text>"
                )
                arrow_label_elements.append(label_rect)
                arrow_label_elements.append(label_svg)
            rendered_arrow_labels.add(s)
    svg.extend(arrow_label_elements)
    for depicted_molecule in render:
        atoms, bonds, define, masks, uid, box, *extras = depicted_molecule
        molecule_svg = _graph_svg(
            atoms, bonds, define, masks, uid, -font125, -box_y, width, height
        )
        molecule_svg.insert(0, box)
        if extras:
            molecule_svg[1:1] = extras
        svg.extend(molecule_svg)
    svg.append("</svg>")
    return "\n".join(svg)


def _route_box_colors() -> dict[str, str]:
    return {
        "target": "#98EEFF",
        "mulecule": "#F0AB90",
        "instock": "#9BFAB3",
    }


def _mirror_route_layout(columns, pred):
    """Mirror root-to-leaf route columns so the target stays on the right."""
    if len(columns) <= 1:
        return columns, tuple(pred)

    old_to_new = {}
    new_col_starts = []
    acc = 0
    for column in reversed(columns):
        new_col_starts.append(acc)
        acc += len(column)

    old_idx = 0
    for col_idx, column in enumerate(columns):
        new_start = new_col_starts[len(columns) - 1 - col_idx]
        for pos_in_col in range(len(column)):
            old_to_new[old_idx] = new_start + pos_in_col
            old_idx += 1

    mirrored_columns = list(reversed(columns))
    mirrored_pred = tuple(
        (old_to_new[parent_idx], old_to_new[child_idx])
        for parent_idx, child_idx in pred
    )
    return mirrored_columns, mirrored_pred


def _render_route_svg(columns, pred, labeled: bool = False) -> str:
    """Shared backend for prepared route columns and parent->child edges."""
    columns, pred = _mirror_route_layout(columns, pred)
    return render_svg(pred, columns, _route_box_colors(), labeled=labeled)


def _format_arrow_label(
    rule_key: str | None,
    policy_rank: int | None,
    *,
    include_rule_key: bool,
) -> str | None:
    """Build the per-arrow label text for route SVGs."""

    parts = []
    if include_rule_key and rule_key:
        parts.append(rule_key)
    if policy_rank is not None:
        parts.append(f"Top-{policy_rank}")
    if parts:
        return " | ".join(parts)
    return None


def _prepare_tree_route_svg_inputs(
    tree: Tree,
    node_id: int,
    labeled: bool = False,
    allow_unsolved: bool = False,
):
    if node_id not in tree.nodes:
        return None
    if not allow_unsolved and node_id not in tree.winning_nodes:
        return None

    path_ids = []
    nid = node_id
    while nid:
        path_ids.append(nid)
        nid = tree.parents[nid]
    path_ids.reverse()

    nodes = [tree.nodes[i] for i in path_ids]

    # Clear previous route annotations so repeated calls do not leak labels.
    for node in nodes:
        curr_precursor = getattr(node, "curr_precursor", None)
        if curr_precursor is not None and hasattr(curr_precursor, "molecule"):
            curr_precursor.molecule.meta.pop("label", None)
            curr_precursor.molecule.meta.pop("status", None)

        for precursor in getattr(node, "new_precursors", ()):
            if hasattr(precursor, "molecule"):
                precursor.molecule.meta.pop("label", None)
                precursor.molecule.meta.pop("status", None)

    for parent_idx in range(len(path_ids) - 1):
        child_id = path_ids[parent_idx + 1]
        label_text = _format_arrow_label(
            tree.nodes_rule_key.get(child_id),
            tree.nodes_policy_rank.get(child_id),
            include_rule_key=labeled,
        )
        if not label_text:
            continue

        parent_node = nodes[parent_idx]
        curr_precursor = getattr(parent_node, "curr_precursor", None)
        if curr_precursor is not None and hasattr(curr_precursor, "molecule"):
            curr_precursor.molecule.meta["label"] = label_text

    for node in nodes:
        for precursor in getattr(node, "new_precursors", ()):
            precursor.molecule.meta["status"] = (
                "instock"
                if precursor.is_building_block(
                    tree.building_blocks, tree.config.min_mol_size
                )
                else "mulecule"
            )
    nodes[0].curr_precursor.molecule.meta["status"] = "target"

    columns = [[nodes[0].curr_precursor.molecule]]
    pred = []

    if len(nodes) == 1:
        return columns, tuple(pred)

    first_layer_precursors = list(nodes[1].new_precursors)
    columns.append([x.molecule for x in first_layer_precursors])
    pred.extend((0, idx) for idx in range(1, len(first_layer_precursors) + 1))

    frontier = [
        idx
        for idx, precursor in enumerate(first_layer_precursors, 1)
        if not precursor.is_building_block(
            tree.building_blocks, tree.config.min_mol_size
        )
    ]

    route_iter = iter(nodes[2:])
    next_idx = count(len(first_layer_precursors) + 1)

    while frontier:
        parent_indices = list(frontier)
        frontier = []
        layer_precursors = []

        for child_node, parent_idx in zip(islice(route_iter, len(parent_indices)), parent_indices):
            for precursor in child_node.new_precursors:
                layer_precursors.append(precursor)
                child_idx = next(next_idx)
                pred.append((parent_idx, child_idx))

                if not precursor.is_building_block(
                    tree.building_blocks, tree.config.min_mol_size
                ):
                    frontier.append(child_idx)

        if layer_precursors:
            columns.append([x.molecule for x in layer_precursors])

    return columns, tuple(pred)


def get_route_svg(
    tree: Tree,
    node_id: int,
    labeled: bool = False,
    allow_unsolved: bool = False,
) -> str:
    """Visualizes the retrosynthetic route.

    :param tree: The built tree.
    :param node_id: The id of the node from which to visualize the route.
    :param labeled: If True, include each disconnection's ``nodes_rule_key`` in
        the arrow label. Stored policy ranks are shown as ``Top-N`` whenever
        available, even when ``labeled`` is False.
    :param allow_unsolved: If True, also render partial routes ending at non-winning
        nodes. Default keeps the historical solved-only behavior.
    :return: The SVG string.
    """
    prepared = _prepare_tree_route_svg_inputs(
        tree,
        node_id,
        labeled=labeled,
        allow_unsolved=allow_unsolved,
    )
    if prepared is None:
        return None

    columns, pred = prepared
    return _render_route_svg(columns, pred, labeled=labeled)


def _get_root(routes_json: dict, route_id: int) -> dict:
    """
    Retrieve the root tree for the given route_id, supporting int or str keys.
    Raises ValueError if not found.
    """
    if route_id in routes_json:
        return routes_json[route_id]
    if str(route_id) in routes_json:
        return routes_json[str(route_id)]
    raise ValueError(f"Route ID {route_id} not found in routes_json.")


def _extract_levels_and_parents(root: dict):
    """
    BFS traversal of the tree to collect molecules by depth
    and record parent links for each mol-node.

    Returns (levels, parent_of, outgoing_reaction_of) where:
      - levels[d] is a list of mol dicts at depth d
      - parent_of[node_id] = parent_mol_dict or None for root
      - outgoing_reaction_of[node_id] = reaction dict branching from the mol node
    """
    levels = []
    parent_of = {}
    outgoing_reaction_of = {}
    queue = deque([(root, 0, None)])

    while queue:
        node, depth, parent = queue.popleft()
        if not isinstance(node, dict) or node.get("type") != "mol":
            continue
        # ensure depth list exists
        if depth >= len(levels):
            levels.extend([] for _ in range(depth - len(levels) + 1))
        levels[depth].append(node)
        parent_of[id(node)] = parent
        outgoing_reaction_of[id(node)] = None

        # enqueue next-layer molecule children
        for reaction in node.get("children") or []:
            if not isinstance(reaction, dict) or reaction.get("type") != "reaction":
                continue
            outgoing_reaction_of[id(node)] = reaction
            for mol_child in reaction.get("children") or []:
                if isinstance(mol_child, dict) and mol_child.get("type") == "mol":
                    queue.append((mol_child, depth + 1, node))

    return levels, parent_of, outgoing_reaction_of


def _prepare_json_route_svg_inputs(routes_json: dict, route_id: int, labeled: bool = False):
    root = _get_root(routes_json, route_id)
    levels, parent_of, outgoing_reaction_of = _extract_levels_and_parents(root)

    mol_container = {}
    for depth, mols in enumerate(levels):
        for mol in mols:
            container = read_smiles(mol["smiles"])
            if depth == 0:
                container.meta["status"] = "target"
            else:
                container.meta["status"] = (
                    "instock" if mol.get("in_stock") else "mulecule"
                )

            reaction = outgoing_reaction_of.get(id(mol))
            label_text = _format_arrow_label(
                reaction.get("rule_key") if reaction else None,
                reaction.get("policy_rank") if reaction else None,
                include_rule_key=labeled,
            )
            if label_text:
                container.meta["label"] = label_text

            mol_container[id(mol)] = container

    pred = []
    flat_ids = [id(mol) for level in levels for mol in level]
    index_map = {node_id: idx for idx, node_id in enumerate(flat_ids)}
    for node_id, parent in parent_of.items():
        if parent is not None:
            pred.append((index_map[id(parent)], index_map[node_id]))

    columns = [[mol_container[id(mol)] for mol in level] for level in levels]
    return columns, tuple(pred)


def get_route_svg_from_json(
    routes_json: dict, route_id: int, labeled: bool = False
) -> str:
    """
    Visualize the retrosynthetic route for routes_json[route_id] as an SVG.
    """
    columns, pred = _prepare_json_route_svg_inputs(
        routes_json, route_id, labeled=labeled
    )
    return _render_route_svg(columns, pred, labeled=labeled)


def generate_results_html(
    tree: Tree, html_path: str, aam: bool = False, extended: bool = False
) -> None:
    """Writes an HTML page with the synthesis routes in SVG format and corresponding
    reactions in SMILES format.

    :param tree: The built tree.
    :param extended: If True, generates the extended route representation.
    :param html_path: The path to the file where to store resulting HTML.
    :param aam: If True, depict atom-to-atom mapping.
    :return: None.
    """
    if aam:
        depict_settings(aam=True)
    else:
        depict_settings(aam=False)

    routes = []
    if extended:
        # Gather paths
        for idx, node in tree.nodes.items():
            if node.is_solved():
                routes.append(idx)
    else:
        routes = tree.winning_nodes
    # HTML Tags
    td = '<td style="text-align: left; border: 1px solid black; border-spacing: 0">'
    font_head = "<font style='font-weight: bold; font-size: 18px'>"
    font_normal = "<font style='font-weight: normal; font-size: 18px'>"
    font_close = "</font>"

    template_begin = """
    <!doctype html>
    <html lang="en">
    <head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
    rel="stylesheet"
    integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3"
    crossorigin="anonymous">
    <script
    src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"
    integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p"
    crossorigin="anonymous">
    </script>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Predicted Paths Report</title>
    <meta name="description" content="A simple HTML5 Template for new projects.">
    <meta name="author" content="SitePoint">
    </head>
    <body>
    """
    template_end = """
    </body>
    </html>
    """
    # SVG Template
    box_mark = """
    <svg width="30" height="30" viewBox="0 0 1 1" xmlns="http://www.w3.org/2000/svg">
    <circle cx="0.5" cy="0.5" r="0.5" fill="rgb()" fill-opacity="0.35" />
    </svg>
    """
    # table = f"<table><thead><{th}>Retrosynthetic Routes</th></thead><tbody>"
    table = """
    <table class="table table-striped table-hover caption-top">
    <caption><h3>Retrosynthetic Routes Report</h3></caption>
    <tbody>"""

    # Gather path data
    table += f"<tr>{td}{font_normal}Target Molecule: {tree.nodes[1].curr_precursor!s}{font_close}</td></tr>"
    table += f"<tr>{td}{font_normal}Tree Size: {len(tree)}{font_close} nodes</td></tr>"
    table += f"<tr>{td}{font_normal}Number of visited nodes: {len(tree.visited_nodes)}{font_close}</td></tr>"
    table += f"<tr>{td}{font_normal}Found paths: {len(routes)}{font_close}</td></tr>"
    table += f"<tr>{td}{font_normal}Time: {round(tree.curr_time, 4)}{font_close} seconds</td></tr>"
    table += f"""
    <tr>{td}
                 <div>
    {box_mark.replace("rgb()", "rgb(152, 238, 255)")}
    Target Molecule
    {box_mark.replace("rgb()", "rgb(240, 171, 144)")}
    Molecule Not In Stock
    {box_mark.replace("rgb()", "rgb(155, 250, 179)")}
    Molecule In Stock
    </div>
    </td></tr>
    """

    for route in routes:
        svg = get_route_svg(tree, route)  # get SVG
        full_route = tree.synthesis_route(route)  # get route
        # write SMILES of all reactions in synthesis path
        step = 1
        reactions = ""
        for synth_step in full_route:
            reactions += f"<b>Step {step}:</b> {synth_step!s}<br>"
            step += 1
        # Concatenate all content of path
        route_score = round(tree.route_score(route), 3)
        table += (
            f'<tr style="line-height: 250%">{td}{font_head}Route {route}; '
            f"Steps: {len(full_route)}; "
            f"Cumulated nodes' value: {route_score}{font_close}</td></tr>"
        )
        # f"Cumulated nodes' value: {node._probabilities[path]}{font_close}</td></tr>"
        table += f"<tr>{td}{svg}</td></tr>"
        table += f"<tr>{td}{reactions}</td></tr>"
    table += "</tbody>"
    if html_path is None:
        return table
    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(template_begin)
        html_file.write(table)
        html_file.write(template_end)


def html_top_routes_cluster(
    clusters: dict, tree: Tree, target_smiles: str, html_path: str | None = None
) -> str:
    """Clustering Results Download: Providing functionality to download the clustering results with styled HTML report."""

    # Compute summary
    total_routes = sum(len(data.get("route_ids", [])) for data in clusters.values())
    total_clusters = len(clusters)

    # Build styled HTML report using Bootstrap
    html = []

    html.append("<!doctype html><html lang='en'><head>")
    html.append(
        "<meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
    )
    html.append(
        "<link href='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css' rel='stylesheet'>"
    )
    now = datetime.now()
    created_time = now.strftime("%Y-%m-%d %H:%M:%S")
    html.append("<title>Clustering Results Report</title>")
    html.append(
        "<style> svg{max-width:100%;height:auto;} .report-table th,.report-table td{vertical-align:top;border:1px solid #dee2e6;} </style>"
    )
    html.append("</head><body><div class='container my-4'>")
    # Report header
    html.append(
        f"""
    <div class="d-flex justify-content-between align-items-center mb-3">
        <h1 class="mb-0">Best route from each cluster</h1>
        <div class="text-end" style="min-width:180px;">
            <p class="mb-1" style="font-size: 1rem;">Report created time:</p>
            <p class="mb-0" style="font-size: 1rem;">{created_time}</p>
        </div>
    </div>
    """
    )
    html.append(f"<p><strong>Target molecule (SMILES):</strong> {target_smiles}</p>")
    html.append(f"<p><strong>Total number of routes:</strong> {total_routes}</p>")
    html.append(f"<p><strong>Total number of clusters:</strong> {total_clusters}</p>")
    # Table header
    html.append(
        "<table class='table report-table'><colgroup><col style='width:5%'><colgroup><col style='width:5%'><col style='width:15%'><col style='width:75%'></colgroup><thead><tr>"
    )
    html.append("<th>Cluster index</th><th>Size</th><th>SB-CGR</th><th>Best Route</th>")
    html.append("</tr></thead><tbody>")

    # Rows per cluster
    for cluster_num, group_data in clusters.items():
        route_ids = group_data.get("route_ids", [])
        if not route_ids:
            continue
        route_id = route_ids[0]
        # Get SVGs
        svg = get_route_svg(tree, route_id)
        r_cgr = group_data.get("sb_cgr")
        r_cgr_svg = None
        if r_cgr:
            r_cgr.clean2d()
            r_cgr_svg = cgr_display(r_cgr)
        # Start row
        html.append(f"<tr><td>{cluster_num}</td>")
        html.append(f"<td>{len(route_ids)}</td>")
        html.append("<td>")
        if r_cgr_svg:
            b64_r = base64.b64encode(r_cgr_svg.encode("utf-8")).decode()
            html.append(
                f"<img src='data:image/svg+xml;base64,{b64_r}' alt='SB-CGR' class='img-fluid'/>"
            )
        html.append("</td>")
        # Best Route cell
        html.append("<td>")
        if svg:
            b64_svg = base64.b64encode(svg.encode("utf-8")).decode()
            html.append(
                f"<img src='data:image/svg+xml;base64,{b64_svg}' alt='Route {route_id}' class='img-fluid'/>"
            )
        html.append("</td></tr>")

    # Close table and HTML
    html.append("</tbody></table>")
    html.append("</div></body></html>")

    report_html = "".join(html)
    if html_path:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        return f"Written to {html_path}"
    return report_html


def routes_clustering_report(
    source: Tree | dict,
    clusters: dict,
    group_index: str,
    sb_cgrs_dict: dict,
    aam: bool = False,
    html_path: str | None = None,
) -> str:
    """
    Generates an HTML report visualizing a cluster of retrosynthetic routes.

    This function takes a source of retrosynthetic routes (either a Tree object
    or a dictionary representing routes in JSON format), cluster information,
    and a dictionary of SB-CGRs, and produces a comprehensive HTML report.
    The report includes details about the cluster, a representative SB-CGR,
    and SVG visualizations of each route within the specified cluster.

    Args:
        source (Union[Tree, dict]): The source of retrosynthetic routes.
                                     Can be a Tree object containing the full
                                     search tree, or a dictionary loaded from
                                     a routes JSON file.
        clusters (dict): A dictionary containing clustering results. It should
                       contain information about different clusters, typically
                       including a list of 'route_ids' for each cluster.
        group_index (str): The key identifying the specific cluster within the
                           `clusters` dictionary for which the report should be
                           generated.
        sb_cgrs_dict (dict): A dictionary mapping route IDs (integers) to
                             SB-CGR objects. Used to display a representative
                             SB-CGR for the cluster.
        aam (bool, optional): Whether to enable atom-atom mapping visualization
                              in molecule depictions. Defaults to False.
        html_path (str, optional): The file path where the generated HTML
                                   report should be saved. If provided, the
                                   function saves the report to this file and
                                   returns a confirmation message. If None,
                                   the function returns the HTML string
                                   directly. Defaults to None.

    Returns:
        str: The generated HTML report as a string, or a string confirming
             the file path where the report was saved if `html_path` is
             provided. Returns an error message string if the input `source`
             or `clusters` are invalid, or if the specified `group_index` is
             not found.
    """
    # --- Depict Settings ---
    with contextlib.suppress(Exception):
        depict_settings(aam=bool(aam))

    # --- Figure out what `source` is ---
    using_tree = False
    if hasattr(source, "nodes") and hasattr(source, "route_to_node"):
        tree = source
        using_tree = True
    elif isinstance(source, dict):
        routes_json = source
        tree = None
    else:
        return "<html><body>Error: first argument must be a Tree or a routes_json dict.</body></html>"

    # --- Validate clusters ---
    if not isinstance(clusters, dict):
        return "<html><body>Error: clusters must be a dict.</body></html>"

    group = clusters.get(group_index)
    if group is None:
        return f"<html><body>Error: no group with index {group_index!r}.</body></html>"

    cluster_route_ids = group.get("route_ids", [])
    # Filter valid routes
    valid_routes = []

    if using_tree:
        for nid in cluster_route_ids:
            if nid in tree.nodes and tree.nodes[nid].is_solved():
                valid_routes.append(nid)
    else:
        # JSON mode: check if the route ID exists in the routes_dict
        routes_dict = make_dict(routes_json)
        for nid in cluster_route_ids:
            if nid in routes_dict:
                valid_routes.append(nid)
    if not valid_routes:
        return f"""
        <!doctype html><html><body>
          <h3>Cluster {group_index} Report</h3>
          <p>No valid routes found in this cluster.</p>
        </body></html>
        """

    # --- Boilerplate HTML head/tail omitted for brevity ---
    template_begin = (
        """<!doctype html><html><head>…</head><body><div class="container">"""
    )
    template_end = """</div></body></html>"""

    table = f"""
      <table class="table">
        <caption><h3>Cluster {group_index} Routes</h3></caption>
        <tbody>
    """

    # show target
    if using_tree:
        try:
            target_smiles = str(tree.nodes[1].curr_precursor)
        except Exception:
            target_smiles = "N/A"
    else:
        # JSON mode: take the root smiles of the first route
        try:
            key = valid_routes[0]
            target_smiles = routes_json.get(key, routes_json.get(str(key), {})).get(
                "smiles", "N/A"
            )
        except Exception:
            target_smiles = "N/A"

    # --- HTML Templates & Tags ---
    td = '<td style="text-align: left; border: 1px solid black; border-spacing: 0">'
    font_normal = "<font style='font-weight: normal; font-size: 18px'>"
    font_close = "</font>"

    template_begin = f"""
    <!doctype html>
    <html lang="en">
    <head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
    rel="stylesheet"
    integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3"
    crossorigin="anonymous">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Cluster {group_index} Routes Report</title>
    <style>
        /* Optional: Add some basic styling */
        .table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        tr:nth-child(even) {{ background-color: #ffffff; }}
        caption {{ caption-side: top; font-size: 1.5em; margin: 1em 0; }}
        svg {{ max-width: 100%; height: auto; }}
    </style>
    </head>
    <body>
    <div class="container"> """

    template_end = """
    </div> <script
    src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"
    integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p"
    crossorigin="anonymous">
    </script>
    </body>
    </html>
    """

    box_mark = """
    <svg width="30" height="30" viewBox="0 0 1 1" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 5px;">
    <circle cx="0.5" cy="0.5" r="0.5" fill="rgb()" fill-opacity="0.35" />
    </svg>
    """

    # --- Build HTML Table ---
    table = f"""
    <table class="table table-hover caption-top">
    <caption><h3>Retrosynthetic Routes Report - Cluster {group_index}</h3></caption>
    <tbody>"""

    table += (
        f"<tr>{td}{font_normal}Target Molecule: {target_smiles}{font_close}</td></tr>"
    )
    table += f"<tr>{td}{font_normal}Group index: {group_index}{font_close}</td></tr>"
    table += f"<tr>{td}{font_normal}Size of Cluster: {len(valid_routes)} routes{font_close} </td></tr>"

    # --- Add SB-CGR Image ---
    first_route_id = valid_routes[0] if valid_routes else None

    if first_route_id and sb_cgrs_dict:
        try:
            sb_cgr = sb_cgrs_dict[first_route_id]
            sb_cgr.clean2d()
            sb_cgr_svg = cgr_display(sb_cgr)

            if sb_cgr_svg.strip().startswith("<svg"):
                table += f"<tr>{td}{font_normal}Identified Strategic Bonds{font_close}<br>{sb_cgr_svg}</td></tr>"
            else:
                table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR (from Route {first_route_id}):{font_close}<br><i>Invalid SVG format retrieved.</i></td></tr>"
                print(
                    f"Warning: Expected SVG for SB-CGR of route {first_route_id}, but got: {sb_cgr_svg[:100]}..."
                )
        except Exception as e:
            table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR (from Route {first_route_id}):{font_close}<br><i>Error retrieving/displaying SB-CGR: {e}</i></td></tr>"
    else:
        if first_route_id:
            table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR (from Route {first_route_id}):{font_close}<br><i>Not found in provided SB-CGR dictionary.</i></td></tr>"
        else:
            table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR:{font_close}<br><i>No valid routes in cluster to select from.</i></td></tr>"

    table += f"""
    <tr>{td}
        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 15px;">
            <span>{box_mark.replace("rgb()", "rgb(152, 238, 255)")} Target Molecule</span>
            <span>{box_mark.replace("rgb()", "rgb(240, 171, 144)")} Molecule Not In Stock</span>
            <span>{box_mark.replace("rgb()", "rgb(155, 250, 179)")} Molecule In Stock</span>
        </div>
    </td></tr>
    """
    for route_id in valid_routes:
        if using_tree:
            # 1) SVG from Tree
            svg = get_route_svg(tree, route_id)
            # 2) Reaction steps & score
            steps = tree.synthesis_route(route_id)
            score = round(tree.route_score(route_id), 3)
            # build reaction list
            reac_html = "".join(
                f"<b>Step {i+1}:</b> {r!s}<br>" for i, r in enumerate(steps)
            )
            header = f"Route {route_id} — {len(steps)} steps, score={score}"
            table += f"<tr><td><b>{header}</b></td></tr>"
            table += f"<tr><td>{svg}</td></tr>"
            table += f"<tr><td>{reac_html}</td></tr>"
        else:
            # 1) SVG from JSON
            svg = get_route_svg_from_json(routes_json, route_id)
            steps = routes_dict[route_id]
            reac_html = "".join(
                f"<b>Step {i+1}:</b> {r!s}<br>" for i, r in steps.items()
            )

            header = f"Route {route_id} — {len(steps)} steps"
            table += f"<tr><td><b>{header}</b></td></tr>"
            table += f"<tr><td>{svg}</td></tr>"
            table += f"<tr><td>{reac_html}</td></tr>"

    table += "</tbody></table>"

    html = template_begin + table + template_end

    if html_path:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return f"Written to {html_path}"
    return html


def lg_table_2_html(subcluster, routes_to_display=None, if_display=True):
    """
    Generates an HTML table visualizing leaving groups (X) 'marks' for routes within a subcluster.

    This function creates an HTML table where each row represents a routes
    from the specified subcluster (or a subset of routes), and columns
    represent unique 'marks' found across the routes. The cells contain
    the SVG depiction of the corresponding mark for that route.

    Args:
        subcluster (dict): A dictionary containing subcluster data, expected
                           to have a 'routes_data' key mapping route IDs to
                           dictionaries of marks and their associated data
                           (where the first element is a depictable object).
        routes_to_display (list, optional): A list of specific route IDs to
                                           include in the table. If empty,
                                           all routes in `subcluster["routes_data"]`
                                           are included. Defaults to [].
        if_display (bool, optional): If True, the generated HTML is
                                     displayed directly using `display(HTML())`.
                                     Defaults to True.

    Returns:
        str: The generated HTML string for the table.
    """
    # Create HTML table header
    if routes_to_display is None:
        routes_to_display = []
    html = "<table style='border-collapse: collapse;'><tr><th style='border: 1px solid black; padding: 4px;'>Route ID</th>"

    # Extract all unique marks across all routes to form consistent columns
    all_marks = set()
    for route_data in subcluster["routes_data"].values():
        all_marks.update(route_data.keys())
    all_marks = sorted(all_marks)  # sort for consistent ordering

    # Add marks as headers
    for mark in all_marks:
        html += f"<th style='border: 1px solid black; padding: 4px;'>{mark}</th>"
    html += "</tr>"

    # Fill in the rows
    if len(routes_to_display) == 0:
        for route_id, route_data in subcluster["routes_data"].items():
            html += f"<tr><td style='border: 1px solid black; padding: 4px;'>{route_id}</td>"
            for mark in all_marks:
                html += "<td style='border: 1px solid black; padding: 4px;'>"
                if mark in route_data:
                    svg = route_data[mark][0].depict()  # Get SVG data as string
                    html += svg
                html += "</td>"
            html += "</tr>"
    else:
        for route_id in routes_to_display:
            # Check if the route_id exists in the subcluster data
            if route_id in subcluster["routes_data"]:
                route_data = subcluster["routes_data"][route_id]
                html += f"<tr><td style='border: 1px solid black; padding: 4px;'>{route_id}</td>"
                for mark in all_marks:
                    html += "<td style='border: 1px solid black; padding: 4px;'>"
                    if mark in route_data:
                        svg = route_data[mark][0].depict()  # Get SVG data as string
                        html += svg
                    html += "</td>"
                html += "</tr>"
            else:
                # Optionally, you can note that the route_id was not found
                html += f"<tr><td colspan='{len(all_marks)+1}' style='border: 1px solid black; padding: 4px; color:red;'>Route ID {route_id} not found.</td></tr>"

    html += "</table>"

    if if_display:
        display(HTML(html))

    return html


def group_lg_table_2_html_fixed(
    grouped: dict,
    groups_to_display=None,
    if_display=False,
    max_group_col_width: int = 200,
) -> str:
    """
    Generates an HTML table visualizing leaving groups X 'marks' for representative routes in grouped data.

    This function takes a dictionary of grouped data, where each key represents
    a group (e.g., a collection of route IDs of routes) and the value is a representative
    dictionary of 'marks' for that group. It generates an HTML table with a
    fixed layout, where each row corresponds to a group, and columns show the
    SVG depiction or string representation of the 'marks' for the group's
    representative.

    Args:
        grouped (dict): A dictionary where keys are group identifiers (e.g.,
                        tuples of route IDs of routes) and values are dictionaries
                        representing the 'marks' for the representative of
                        that group. The 'marks' dictionary should map mark
                        names (str) to objects that have a `.depict()` method
                        or are convertible to a string.
        groups_to_display (list, optional): A list of specific group
                                            identifiers to include in the table.
                                            If None, all groups in the `grouped`
                                            dictionary are included. Defaults to None.
        if_display (bool, optional): If True, the generated HTML is
                                     displayed directly using `display(HTML())`.
                                     Defaults to False.
        max_group_col_width (int, optional): The maximum width (in pixels)
                                             for the column displaying the
                                             group identifiers. Defaults to 200.

    Returns:
        str: The generated HTML string for the table.
    """
    # 1) pick which groups to show
    if groups_to_display is None:
        groups = list(grouped.keys())
    else:
        groups = [g for g in groups_to_display if g in grouped]

    # 2) collect all marks for the header
    all_marks = sorted({m for rep in grouped.values() for m in rep})

    # 3) build table start with auto layout
    html = [
        "<table style='width:100%; table-layout:auto; border-collapse: collapse;'>",
        "<thead><tr>",
        "<th style='border:1px solid #ccc; padding:4px;'>Route IDs</th>",
    ]
    # numeric headers
    html += [
        f"<th style='border:1px solid #ccc; padding:4px; text-align:center;'>X<small>{mark}</small></th>"
        for mark in all_marks
    ]
    html.append("</tr></thead><tbody>")

    # 4) each row
    group_td_style = (
        f"border:1px solid #ccc; padding:4px; "
        "white-space: normal; overflow-wrap: break-word; "
        f"max-width:{max_group_col_width}px;"
    )
    img_td_style = (
        "border:1px solid #ccc; padding:4px; text-align:center; vertical-align:middle;"
    )

    for group in groups:
        rep = grouped[group]
        label = ",".join(str(n) for n in group)
        # start row
        row = [f"<td style='{group_td_style}'>{label}</td>"]
        # fill in each mark column
        for mark in all_marks:
            cell = ["<td style='" + img_td_style + "'>"]
            if mark in rep:
                val = rep[mark]
                cell.append(val.depict() if hasattr(val, "depict") else str(val))
            cell.append("</td>")
            row.append("".join(cell))
        html.append("<tr>" + "".join(row) + "</tr>")

    html.append("</tbody></table>")
    out = "".join(html)
    if if_display:
        display(HTML(out))

    return out


def routes_subclustering_report(
    source: Tree | dict,
    subcluster: dict,
    group_index: str,
    cluster_num: int,
    sb_cgrs_dict: dict,
    if_lg_group: bool = False,
    aam: bool = False,
    html_path: str | None = None,
) -> str:
    """
    Generates an HTML report visualizing a specific subcluster of retrosynthetic routes.

    This function takes a source of retrosynthetic routes (either a Tree object
    or a dictionary representing routes in JSON format), data for a specific
    subcluster, and a dictionary of SB-CGRs. It produces a detailed HTML report
    for the subcluster, including general cluster information, a representative
    SB-CGR, a synthon pseudo reaction, a table of leaving groups (either per
    route or grouped), and SVG visualizations of each valid route within the
    subcluster.

    Args:
        source (Union[Tree, dict]): The source of retrosynthetic routes.
                                     Can be a Tree object containing the full
                                     search tree, or a dictionary loaded from
                                     a routes JSON file.
        subcluster (dict): A dictionary containing data for the specific
                           subcluster. Expected keys include 'routes_data'
                           (mapping route IDs to mark data), 'synthon_reaction',
                           and optionally 'group_lgs' if `if_lg_group` is True.
        group_index (str): The index of the main cluster to which this
                           subcluster belongs. Used for report titling.
        cluster_num (int): The number or identifier of the subcluster within
                           its main group. Used for report titling.
        sb_cgrs_dict (dict): A dictionary mapping route IDs (integers) to
                             SB-CGR objects. Used to display a representative
                             SB-CGR for the cluster.
        if_lg_group (bool, optional): If True, the leaving groups table will
                                     display grouped leaving groups from
                                     `subcluster['group_lgs']`. If False, it
                                     will display leaving groups per individual
                                     route from `subcluster['routes_data']`.
                                     Defaults to False.
        aam (bool, optional): Whether to enable atom-atom mapping visualization
                              in molecule depictions. Defaults to False.
        html_path (str, optional): The file path where the generated HTML
                                   report should be saved. If provided, the
                                   function saves the report to this file and
                                   returns a confirmation message. If None,
                                   the function returns the HTML string
                                   directly. Defaults to None.

    Returns:
        str: The generated HTML report as a string, or a string confirming
             the file path where the report was saved if `html_path` is
             provided. Returns a minimal HTML page indicating no valid routes
             if the subcluster contains no valid/solved routes. Returns an
             error message string if the input `source` or `subcluster` are
             invalid.
    """
    # --- Depict Settings ---
    with contextlib.suppress(Exception):
        depict_settings(aam=bool(aam))

    # --- Figure out what `source` is ---
    using_tree = False
    if hasattr(source, "nodes") and hasattr(source, "route_to_node"):
        tree = source
        using_tree = True
    elif isinstance(source, dict):
        routes_json = source
        tree = None
    else:
        return "<html><body>Error: first argument must be a Tree or a routes_json dict.</body></html>"

    # --- Validate groups ---
    if not isinstance(subcluster, dict):
        return "<html><body>Error: groups must be a dict.</body></html>"

    subcluster_route_ids = list(subcluster["routes_data"].keys())
    # Filter valid routes
    valid_routes = []

    if using_tree:
        for nid in subcluster_route_ids:
            if nid in tree.nodes and tree.nodes[nid].is_solved():
                valid_routes.append(nid)
    else:
        # JSON mode: just keep those IDs present in the JSON
        for nid in subcluster_route_ids:
            if nid in routes_json:
                valid_routes.append(nid)
        routes_dict = make_dict(routes_json)

    if not valid_routes:
        # Return a minimal HTML page indicating no valid routes
        return f"""
        <!doctype html><html lang="en"><head><meta charset="utf-8">
        <title>Cluster {group_index}.{cluster_num} Report</title></head><body>
        <h3>Cluster {group_index}.{cluster_num} Report</h3>
        <p>No valid/solved routes found for this cluster.</p>
        </body></html>"""

    # --- Boilerplate HTML head/tail omitted for brevity ---
    template_begin = (
        """<!doctype html><html><head>…</head><body><div class="container">"""
    )
    template_end = """</div></body></html>"""

    table = f"""
      <table class="table">
        <caption><h3>Cluster {group_index} Routes</h3></caption>
        <tbody>
    """

    # show target
    if using_tree:
        try:
            target_smiles = str(tree.nodes[1].curr_precursor)
        except Exception:
            target_smiles = "N/A"
    else:
        # JSON mode: take the root smiles of the first route
        try:
            key = valid_routes[0]
            target_smiles = routes_json.get(key, routes_json.get(str(key), {})).get(
                "smiles", "N/A"
            )
        except Exception:
            target_smiles = "N/A"

    # legend row omitted…

    # --- HTML Templates & Tags ---
    td = '<td style="text-align: left; border: 1px solid black; border-spacing: 0">'
    font_normal = "<font style='font-weight: normal; font-size: 18px'>"
    font_close = "</font>"

    template_begin = f"""
    <!doctype html>
    <html lang="en">
    <head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
    rel="stylesheet"
    integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3"
    crossorigin="anonymous">
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>SubCluster {group_index}.{cluster_num} Routes Report</title>
    <style>
        /* Optional: Add some basic styling */
        .table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        tr:nth-child(even) {{ background-color: #ffffff; }}
        caption {{ caption-side: top; font-size: 1.5em; margin: 1em 0; }}
        svg {{ max-width: 100%; height: auto; }}
    </style>
    </head>
    <body>
    <div class="container"> """

    template_end = """
    </div> <script
    src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"
    integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p"
    crossorigin="anonymous">
    </script>
    </body>
    </html>
    """

    box_mark = """
    <svg width="30" height="30" viewBox="0 0 1 1" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 5px;">
    <circle cx="0.5" cy="0.5" r="0.5" fill="rgb()" fill-opacity="0.35" />
    </svg>
    """

    # --- Build HTML Table ---
    table = f"""
    <table class="table table-hover caption-top">
    <caption><h3>Retrosynthetic Routes Report - Cluster {group_index}.{cluster_num}</h3></caption>
    <tbody>"""

    table += (
        f"<tr>{td}{font_normal}Target Molecule: {target_smiles}{font_close}</td></tr>"
    )
    table += f"<tr>{td}{font_normal}Group index: {group_index}{font_close}</td></tr>"
    table += f"<tr>{td}{font_normal}Cluster Number: {cluster_num}{font_close}</td></tr>"
    table += f"<tr>{td}{font_normal}Size of Cluster: {len(valid_routes)} routes{font_close} </td></tr>"

    # --- Add SB-CGR Image ---
    first_route_id = valid_routes[0] if valid_routes else None

    if first_route_id and sb_cgrs_dict:
        try:
            sb_cgr = sb_cgrs_dict[first_route_id]
            sb_cgr.clean2d()
            sb_cgr_svg = cgr_display(sb_cgr)

            if sb_cgr_svg.strip().startswith("<svg"):
                table += f"<tr>{td}{font_normal}Identified Strategic Bonds{font_close}<br>{sb_cgr_svg}</td></tr>"
            else:
                table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR (from Route {first_route_id}):{font_close}<br><i>Invalid SVG format retrieved.</i></td></tr>"
                print(
                    f"Warning: Expected SVG for SB-CGR of route {first_route_id}, but got: {sb_cgr_svg[:100]}..."
                )
        except Exception as e:
            table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR (from Route {first_route_id}):{font_close}<br><i>Error retrieving/displaying SB-CGR: {e}</i></td></tr>"
    else:
        if first_route_id:
            table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR (from Route {first_route_id}):{font_close}<br><i>Not found in provided SB-CGR dictionary.</i></td></tr>"
        else:
            table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR:{font_close}<br><i>No valid routes in cluster to select from.</i></td></tr>"

    try:
        synthon_reaction = subcluster["synthon_reaction"]
        synthon_reaction.clean2d()
        synthon_svg = depict_custom_reaction(synthon_reaction)

        extra_synthon = f"<tr>{td}{font_normal}Synthon pseudo reaction:{font_close}<br>{synthon_svg}</td></tr>"
        table += extra_synthon
    except Exception as e:
        table += f"<tr><td colspan='1' style='color: red;'>Error displaying synthon reaction: {e}</td></tr>"

    try:
        if if_lg_group:
            grouped_lgs = subcluster["group_lgs"]
            lg_table_html = group_lg_table_2_html_fixed(grouped_lgs, if_display=False)
        else:
            lg_table_html = lg_table_2_html(subcluster, if_display=False)
        extra_lg = f"<tr>{td}{font_normal}Leaving Groups table:{font_close}<br>{lg_table_html}</td></tr>"
        table += extra_lg
    except Exception as e:
        table += f"<tr><td colspan='1' style='color: red;'>Error displaying leaving groups: {e}</td></tr>"

    table += f"""
    <tr>{td}
        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 15px;">
            <span>{box_mark.replace("rgb()", "rgb(152, 238, 255)")} Target Molecule</span>
            <span>{box_mark.replace("rgb()", "rgb(240, 171, 144)")} Molecule Not In Stock</span>
            <span>{box_mark.replace("rgb()", "rgb(155, 250, 179)")} Molecule In Stock</span>
        </div>
    </td></tr>
    """
    for route_id in valid_routes:
        if using_tree:
            # 1) SVG from Tree
            svg = get_route_svg(tree, route_id)
            # 2) Reaction steps & score
            steps = tree.synthesis_route(route_id)
            score = round(tree.route_score(route_id), 3)
            # build reaction list
            reac_html = "".join(
                f"<b>Step {i+1}:</b> {r!s}<br>" for i, r in enumerate(steps)
            )
            header = f"Route {route_id} — {len(steps)} steps, score={score}"
            table += f"<tr><td><b>{header}</b></td></tr>"
            table += f"<tr><td>{svg}</td></tr>"
            table += f"<tr><td>{reac_html}</td></tr>"

        else:
            # 1) SVG from JSON
            svg = get_route_svg_from_json(routes_json, route_id)
            steps = routes_dict[route_id]
            reac_html = "".join(
                f"<b>Step {i+1}:</b> {r!s}<br>" for i, r in steps.items()
            )

            header = f"Route {route_id} — {len(steps)} steps"
            table += f"<tr><td><b>{header}</b></td></tr>"
            table += f"<tr><td>{svg}</td></tr>"
            table += f"<tr><td>{reac_html}</td></tr>"

    table += "</tbody></table>"

    html = template_begin + table + template_end

    if html_path:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return f"Written to {html_path}"
    return html
