"""Module containing functions for analysis and visualization of the built tree."""

import base64
from itertools import count, islice
from collections import deque
from typing import Any, Dict, List, Union
from datetime import datetime

from CGRtools.containers.molecule import MoleculeContainer
from CGRtools import smiles as read_smiles

from synplan.chem.reaction_routes.visualisation import (
    cgr_display,
    depict_custom_reaction,
)
from synplan.chem.reaction_routes.io import make_dict
from synplan.mcts.tree import Tree

from IPython.display import display, HTML


def get_child_nodes(
    tree: Tree,
    molecule: MoleculeContainer,
    graph: Dict[MoleculeContainer, List[MoleculeContainer]],
) -> Dict[str, Any]:
    """Extracts the child nodes of the given molecule.

    :param tree: The built tree.
    :param molecule: The molecule in the tree from which to extract child nodes.
    :param graph: The relationship between the given molecule and child nodes.
    :return: The dict with extracted child nodes.
    """

    nodes = []
    try:
        graph[molecule]
    except KeyError:
        return []
    for precursor in graph[molecule]:
        temp_obj = {
            "smiles": str(precursor),
            "type": "mol",
            "in_stock": str(precursor) in tree.building_blocks,
        }
        node = get_child_nodes(tree, precursor, graph)
        if node:
            temp_obj["children"] = [node]
        nodes.append(temp_obj)
    return {"type": "reaction", "children": nodes}


def extract_routes(
    tree: Tree, extended: bool = False, min_mol_size: int = 0
) -> List[Dict[str, Any]]:
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
            nodes = tree.route_to_node(winning_node)
            graph, pred = {}, {}
            for before, after in zip(nodes, nodes[1:]):
                before = before.curr_precursor.molecule
                graph[before] = after = [x.molecule for x in after.new_precursors]
                for x in after:
                    pred[x] = before

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


def render_svg(pred, columns, box_colors):
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

    Returns:
        str: A string containing the complete SVG code for the retrosynthetic
             route visualization.
    """
    x_shift = 0.0
    c_max_x = 0.0
    c_max_y = 0.0
    render = []
    cx = count()
    cy = count()
    arrow_points = {}
    for ms in columns:
        heights = []
        for m in ms:
            m.clean2d()
            # X-shift for target
            min_x = min(x for x, y in m._plane.values()) - x_shift
            min_y = min(y for x, y in m._plane.values())
            m._plane = {n: (x - min_x, y - min_y) for n, (x, y) in m._plane.items()}
            max_x = max(x for x, y in m._plane.values())

            c_max_x = max(c_max_x, max_x)

            arrow_points[next(cx)] = [x_shift, max_x]
            heights.append(max(y for x, y in m._plane.values()))

        x_shift = c_max_x + 5.0  # between columns gap
        # calculate Y-shift
        y_shift = sum(heights) + 3.0 * (len(heights) - 1)

        c_max_y = max(c_max_y, y_shift)

        y_shift /= 2.0
        for m, h in zip(ms, heights):
            m._plane = {n: (x, y - y_shift) for n, (x, y) in m._plane.items()}

            # calculate coordinates for boxes
            max_x = max(x for x, y in m._plane.values()) + 0.9  # max x
            min_x = min(x for x, y in m._plane.values()) - 0.6  # min x
            max_y = -(max(y for x, y in m._plane.values()) + 0.45)  # max y
            min_y = -(min(y for x, y in m._plane.values()) - 0.45)  # min y
            x_delta = abs(max_x - min_x)
            y_delta = abs(max_y - min_y)
            box = (
                f'<rect x="{min_x}" y="{max_y}" rx="{y_delta * 0.1}" ry="{y_delta * 0.1}" width="{x_delta}" height="{y_delta}"'
                f' stroke="black" stroke-width=".0025" fill="{box_colors[m.meta["status"]]}" fill-opacity="0.30"/>'
            )
            arrow_points[next(cy)].append(y_shift - h / 2.0)
            y_shift -= h + 3.0
            depicted_molecule = list(m.depict(embedding=True))[:3]
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
            s_min_x, s_max, s_y = arrow_points[s][:3]  # s
            p_min_x, p_max, p_y = arrow_points[p][:3]  # p
            p_max += 1
            mid = p_max + (s_min_x - p_max) / 3
            mid_x = max(mid_x, mid)
        for p in ps:
            arrow_points[p].append(mid_x)

    config = MoleculeContainer._render_config
    font_size = config["font_size"]
    font125 = 1.25 * font_size
    width = c_max_x + 4.0 * font_size  # 3.0 by default
    height = c_max_y + 3.5 * font_size  # 2.5 by default
    box_y = height / 2.0
    svg = [
        f'<svg width="{0.6 * width:.2f}cm" height="{0.6 * height:.2f}cm" '
        f'viewBox="{-font125:.2f} {-box_y:.2f} {width:.2f} '
        f'{height:.2f}" xmlns="http://www.w3.org/2000/svg" version="1.1">',
        '  <defs>\n    <marker id="arrow" markerWidth="10" markerHeight="10" '
        'refX="0" refY="3" orient="auto">\n      <path d="M0,0 L0,6 L9,3"/>\n    </marker>\n  </defs>',
    ]

    for s, p in pred:
        s_min_x, s_max, s_y = arrow_points[s][:3]
        p_min_x, p_max, p_y = arrow_points[p][:3]
        p_max += 1
        mid_x = arrow_points[p][-1]  # p_max + (s_min_x - p_max) / 3
        arrow = f"""  <polyline points="{p_max:.2f} {p_y:.2f}, {mid_x:.2f} {p_y:.2f}, {mid_x:.2f} {s_y:.2f}, {s_min_x - 1.:.2f} {s_y:.2f}"
                fill="none" stroke="black" stroke-width=".04" marker-end="url(#arrow)"/>"""
        if p_y != s_y:
            arrow += f'  <circle cx="{mid_x}" cy="{p_y}" r="0.1"/>'
        svg.append(arrow)
    for atoms, bonds, masks, box in render:
        molecule_svg = MoleculeContainer._graph_svg(
            atoms, bonds, masks, -font125, -box_y, width, height
        )
        molecule_svg.insert(1, box)
        svg.extend(molecule_svg)
    svg.append("</svg>")
    return "\n".join(svg)


def get_route_svg(tree: Tree, node_id: int) -> str:
    """Visualizes the retrosynthetic route.

    :param tree: The built tree.
    :param node_id: The id of the node from which to visualize the route.
    :return: The SVG string.
    """
    if node_id not in tree.winning_nodes:
        return None
    nodes = tree.route_to_node(node_id)
    # Set up node_id types for different box colors
    for n in nodes:
        for precursor in n.new_precursors:
            precursor.molecule.meta["status"] = (
                "instock"
                if precursor.is_building_block(tree.building_blocks)
                else "mulecule"
            )
    nodes[0].curr_precursor.molecule.meta["status"] = "target"
    # Box colors
    box_colors = {
        "target": "#98EEFF",  # 152, 238, 255
        "mulecule": "#F0AB90",  # 240, 171, 144
        "instock": "#9BFAB3",  # 155, 250, 179
    }

    # first column is target
    # second column are first new precursor_to_expand
    columns = [
        [nodes[0].curr_precursor.molecule],
        [x.molecule for x in nodes[1].new_precursors],
    ]
    pred = {x: 0 for x in range(1, len(columns[1]) + 1)}
    cx = [
        n
        for n, x in enumerate(nodes[1].new_precursors, 1)
        if not x.is_building_block(tree.building_blocks)
    ]
    size = len(cx)
    nodes = iter(nodes[2:])
    cy = count(len(columns[1]) + 1)
    while size:
        layer = []
        for s in islice(nodes, size):
            n = cx.pop(0)
            for x in s.new_precursors:
                layer.append(x)
                m = next(cy)
                if not x.is_building_block(tree.building_blocks):
                    cx.append(m)
                pred[m] = n
        size = len(cx)
        columns.append([x.molecule for x in layer])

    columns = [
        columns[::-1] for columns in columns[::-1]
    ]  # Reverse array to make retrosynthetic graph
    pred = tuple(  # Change dict to tuple to make multiple precursor_to_expand available
        (abs(source - len(pred)), abs(target - len(pred)))
        for target, source in pred.items()
    )
    svg = render_svg(pred, columns, box_colors)
    return svg


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

    Returns (levels, parent_of) where:
      - levels[d] is a list of mol dicts at depth d
      - parent_of[node_id] = parent_mol_dict or None for root
    """
    levels = []
    parent_of = {}
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

        # enqueue next-layer molecule children
        for reaction in node.get("children") or []:
            if not isinstance(reaction, dict) or reaction.get("type") != "reaction":
                continue
            for mol_child in reaction.get("children") or []:
                if isinstance(mol_child, dict) and mol_child.get("type") == "mol":
                    queue.append((mol_child, depth + 1, node))

    return levels, parent_of


def get_route_svg_from_json(routes_json: dict, route_id: int) -> str:
    """
    Visualize the retrosynthetic route for routes_json[route_id] as an SVG.
    """
    # 1) Locate the root tree for this route
    root = _get_root(routes_json, route_id)

    # 2) Build per-depth molecule lists & parent mapping
    levels, parent_of = _extract_levels_and_parents(root)

    # 3) Create MoleculeContainer instances and set statuses
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
            mol_container[id(mol)] = container

    # 4) Mirror the columns (reverse depth order)
    json_columns = list(reversed(levels))

    # 5) Flatten and index node IDs for layout ordering
    flat_ids = [id(m) for lvl in json_columns for m in lvl]
    index_map = {nid: idx for idx, nid in enumerate(flat_ids)}

    # 6) Build predecessor edges (parent -> child) in flattened indices
    pred = []
    for node_id, parent in parent_of.items():
        if parent is not None:
            pred.append((index_map[id(parent)], index_map[node_id]))
    pred = tuple(pred)

    # 7) Map JSON columns to MoleculeContainer columns
    columns = [[mol_container[id(m)] for m in lvl] for lvl in json_columns]

    # 8) Render SVG with status color coding
    box_colors = {
        "target": "#98EEFF",
        "mulecule": "#F0AB90",
        "instock": "#9BFAB3",
    }
    return render_svg(pred, columns, box_colors)


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
        MoleculeContainer.depict_settings(aam=True)
    else:
        MoleculeContainer.depict_settings(aam=False)

    routes = []
    if extended:
        # Gather paths
        for idx, node in tree.nodes.items():
            if node.is_solved():
                routes.append(idx)
    else:
        routes = tree.winning_nodes
    # HTML Tags
    th = '<th style="text-align: left; background-color:#978785; border: 1px solid black; border-spacing: 0">'
    td = '<td style="text-align: left; border: 1px solid black; border-spacing: 0">'
    font_red = "<font color='red' style='font-weight: bold'>"
    font_green = "<font color='light-green' style='font-weight: bold'>"
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
    table += f"<tr>{td}{font_normal}Target Molecule: {str(tree.nodes[1].curr_precursor)}{font_close}</td></tr>"
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
            reactions += f"<b>Step {step}:</b> {str(synth_step)}<br>"
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
    clusters: dict, tree: Tree, target_smiles: str, html_path: str = None
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
    source: Union[Tree, dict],
    clusters: dict,
    group_index: str,
    sb_cgrs_dict: dict,
    aam: bool = False,
    html_path: str = None,
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
    try:
        MoleculeContainer.depict_settings(aam=bool(aam))
    except Exception:
        pass

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
            if nid in routes_dict.keys():
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
            target_smiles = routes_json[valid_routes[0]]["smiles"]
        except:
            target_smiles = routes_json[valid_routes[0]]["smiles"]

    # --- HTML Templates & Tags ---
    th = '<th style="text-align: left; background-color:#978785; border: 1px solid black; border-spacing: 0">'
    td = '<td style="text-align: left; border: 1px solid black; border-spacing: 0">'
    font_head = "<font style='font-weight: bold; font-size: 18px'>"
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
                f"<b>Step {i+1}:</b> {str(r)}<br>" for i, r in enumerate(steps)
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
                f"<b>Step {i+1}:</b> {str(r)}<br>" for i, r in steps.items()
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


def lg_table_2_html(subcluster, routes_to_display=[], if_display=True):
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
    all_marks = sorted({m for rep in grouped.values() for m in rep.keys()})

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
    source: Union[Tree, dict],
    subcluster: dict,
    group_index: str,
    cluster_num: int,
    sb_cgrs_dict: dict,
    if_lg_group: bool = False,
    aam: bool = False,
    html_path: str = None,
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
    try:
        MoleculeContainer.depict_settings(aam=bool(aam))
    except Exception:
        pass

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
        target_smiles = routes_json[valid_routes[0]]["smiles"]

    # legend row omitted…

    # --- HTML Templates & Tags ---
    th = '<th style="text-align: left; background-color:#978785; border: 1px solid black; border-spacing: 0">'
    td = '<td style="text-align: left; border: 1px solid black; border-spacing: 0">'
    font_head = "<font style='font-weight: bold; font-size: 18px'>"
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
                f"<b>Step {i+1}:</b> {str(r)}<br>" for i, r in enumerate(steps)
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
                f"<b>Step {i+1}:</b> {str(r)}<br>" for i, r in steps.items()
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
