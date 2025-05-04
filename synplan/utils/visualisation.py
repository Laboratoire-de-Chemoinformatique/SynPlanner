"""Module containing functions for analysis and visualization of the built tree."""

from itertools import count, islice
from typing import Any, Dict, List

from CGRtools.containers.molecule import MoleculeContainer

from synplan.mcts.tree import Tree


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


def get_route_svg(tree: Tree, node_id: int) -> str:
    """Visualizes the retrosynthetic route.

    :param tree: The built tree.
    :param node_id: The id of the node from which to visualize the route.
    :return: The SVG string.
    """
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

    # now we have columns for visualizing
    # lets start recalculate XY
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
