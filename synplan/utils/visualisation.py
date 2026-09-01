"""Module containing functions for analysis and visualization of the built tree."""

from __future__ import annotations

import base64
import contextlib
from datetime import datetime
from html import escape
from typing import TYPE_CHECKING, Any

from chython import depict_settings
from chython.containers.molecule import MoleculeContainer
from IPython.display import HTML, display

from synplan.chem.precursor import is_purchasable
from synplan.chem.reaction.routes.io import make_dict
from synplan.chem.reaction.routes.representation.depiction import (
    cgr_display,
    depict_custom_reaction,
)
from synplan.chem.reaction.routes.route import Route, Step
from synplan.chem.reaction.rules.priority import POLICY_SOURCE_NAME
from synplan.utils.frames import depict_value
from synplan.utils.routedraw import (
    ARROW_DEFS,
    ROLE_STYLE,
    ROUTE_CSS,
    drawable_copy,
    molecule_svg,
)
from synplan.utils.svgslim import Doc, hidden_defs

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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
            "in_stock": is_purchasable(
                precursor,
                tree.building_blocks,
                min_mol_size=0,
                key=str(precursor),
                building_block_candidates=getattr(
                    tree, "building_block_candidates", None
                ),
                use_full_inchikey=getattr(tree, "use_full_inchikey", False),
            ),
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
        tree.building_blocks,
        min_mol_size,
        building_block_candidates=getattr(tree, "building_block_candidates", None),
        use_full_inchikey=getattr(tree, "use_full_inchikey", False),
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

            for before_node, after_node in tree.route_steps(winning_node):
                before = before_node.curr_precursor.molecule
                graph[before] = {
                    "children": [
                        precursor.molecule for precursor in after_node.new_precursors
                    ],
                    "rule_key": after_node.rule_key,
                    "policy_rank": after_node.policy_rank,
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


def _priority_rule(tree: Tree, node) -> Any | None:
    """The curated rule that produced `node`, or None for a policy step.

    Everything is fetched defensively: the lookup is also driven by duck-typed
    stand-ins that carry neither priority rules nor rule ids.
    """
    rules = getattr(tree, "priority_rules", {}).get(
        getattr(node, "rule_source", None) or "", ()
    )
    rule_id = getattr(node, "rule_id", None)
    if rule_id is None or rule_id >= len(rules):
        return None
    return rules[rule_id]


def _json_route(routes_json: dict, route_id: int) -> Route:
    """One route out of a ``make_json`` mapping, by id.

    A mapping that has been through a JSON file spells its ids as strings, so both
    are tried before giving up.
    """
    node = routes_json.get(route_id, routes_json.get(str(route_id)))
    if node is None:
        raise ValueError(f"Route ID {route_id} not found in routes_json.")
    return Route.from_json(node)


def route_rule_labels(tree: Tree, node_id: int) -> dict[int, str]:
    """`{tree node id: label}` for the steps of the route ending at `node_id`.

    Priority rules carry a chemistry name (`rule_name` stamped by their loader); the policy has
    none, so its steps stay unlabelled rather than being given a meaningless id. Keyed by node
    id, so a step finds its label whatever order the route is read in.
    """
    labels = {}
    for route_node_id in tree.route_node_ids(node_id)[1:]:
        rule = _priority_rule(tree, tree.nodes[route_node_id])
        name = getattr(rule, "rule_name", None)
        labels[route_node_id] = f"{rule.rule_id} — {name}" if name else ""
    return labels


#: Page chrome for :func:`routes_report_html`. The route drawings bring their own
#: rules through :data:`synplan.utils.routedraw.ROUTE_CSS`.
_REPORT_CSS = """
:root{--ink:#0f1419;--ink2:#38414a;--ink3:#6b7480;--ink4:#9ba3ad;
--rule:#e6e8eb;--surface:#ffffff;--bg:#fafbfc;--accent:#1e3a8a;--ok:#1f4d3d}
*,::before,::after{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:14px;line-height:1.5;
font-family:"Inter Tight",system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:44px 28px 96px}
.eyebrow{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--ink3)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:3px}
header.page{padding:26px 26px 24px}
header.page h1{margin:.35em 0 0;font-size:21px;font-weight:600;letter-spacing:-.01em}
.target{display:flex;gap:20px;align-items:center;margin-top:20px}
.tile{flex:0 0 auto;border:1px solid var(--accent);border-radius:3px;background:#fff;padding:9px 11px}
.tile svg{display:block}
.target .smi{font-size:13px;color:var(--ink2);word-break:break-all;margin-top:5px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);margin-top:24px;
border:1px solid var(--rule);border-radius:3px;overflow:hidden;background:var(--surface)}
.stat{padding:13px 16px 14px;border-left:1px solid var(--rule)}
.stat:first-child{border-left:0}
.stat .v{display:block;margin-top:5px;font-size:27px;font-weight:600;line-height:1.05;letter-spacing:-.02em}
.stat .u{font-size:12px;font-weight:400;color:var(--ink4);letter-spacing:0}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.chip{display:inline-flex;align-items:center;gap:7px;padding:4px 10px;background:var(--surface);
border:1px solid var(--rule);border-radius:3px;font-size:11px;font-weight:600;
text-transform:uppercase;letter-spacing:.1em;color:var(--ink2)}
.sw{width:12px;height:12px;border-radius:2px;flex:0 0 auto}
.route{margin-top:20px}
.rhead{display:flex;flex-wrap:wrap;gap:34px;padding:13px 18px;border-bottom:1px solid var(--rule)}
.kv .v{font-size:15px;font-weight:600;margin-top:2px}
.kv .v.id{color:var(--accent)}
.draw{padding:20px 18px 22px;overflow-x:auto;display:flex;justify-content:center;
background:radial-gradient(circle,#e9ecef .9px,transparent .9px) 0 0/30px 30px,#fff}
.draw > svg{display:block;max-width:100%;height:auto;flex:0 0 auto}
.step{display:grid;grid-template-columns:22px minmax(0,1fr);gap:0 13px;
align-items:start;padding:11px 18px;border-top:1px solid var(--rule)}
.disc{width:22px;height:22px;border-radius:50%;background:#2b3440;color:#fff;
font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;
line-height:1;margin-top:1px}
.lab{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--ok)}
.rxn{font-size:12px;color:var(--ink2);word-break:break-all;line-height:1.6}
"""

#: Every role the drawing tints, in reading order.
_ROLE_LEGEND = (
    ("target", "Target molecule"),
    ("int", "Intermediate"),
    ("oos", "Not in stock"),
    ("bb", "In stock"),
)


def _report_header(routes: Sequence[Route], tile: MoleculeContainer | None) -> str:
    scores = [
        route.provenance.search_score
        for route in routes
        if route.provenance is not None and route.provenance.search_score is not None
    ]
    stats = (
        ("Routes", len(routes), ""),
        ("Solved", sum(route.solved for route in routes), f" of {len(routes)}"),
        ("Longest", max((len(route) for route in routes), default=0), " steps"),
        ("Best score", round(max(scores), 3) if scores else "—", ""),
    )
    target = (
        ""
        if tile is None
        else f'<div class="target"><div class="tile">{molecule_svg(tile)}</div>'
        '<div><div class="eyebrow">Target molecule</div>'
        f'<div class="smi mono">{escape(str(tile))}</div></div></div>'
    )
    return (
        '<header class="page card">'
        '<div class="eyebrow">Retrosynthetic routes report</div>'
        "<h1>Predicted routes</h1>"
        + target
        + '<div class="stats">'
        + "".join(
            f'<div class="stat"><span class="eyebrow">{name}</span>'
            f'<span class="v">{value}<span class="u">{unit}</span></span></div>'
            for name, value, unit in stats
        )
        + '</div><div class="legend">'
        + "".join(
            f'<span class="chip"><span class="sw" style="background:{ROLE_STYLE[role][0]};'
            f'border:1px solid {ROLE_STYLE[role][1]}"></span>{caption}</span>'
            for role, caption in _ROLE_LEGEND
        )
        + "</div></header>"
    )


def _step_label(step: Step) -> str:
    """The curated rule behind a step, or "" for one the policy proposed.

    A policy rule's id says nothing to a chemist, so a policy step stays
    unlabelled; a priority step is named by its ``rule_key`` (``set:id``), the one
    identifier the detached route carries.
    """

    origin = step.origin
    if origin is None or origin.rule_source in (None, POLICY_SOURCE_NAME):
        return ""
    return origin.rule_key or ""


def routes_report_html(
    routes: Iterable[Route], html_path: str | None, aam: bool = False
) -> str | None:
    """Write an HTML page with the given routes drawn and listed as SMILES.

    Draws exactly the routes it is handed, in the order it is handed them --
    solved, unsolved, from one tree, from several, or read back out of a file.
    Which routes are worth a page is the caller's call:
    ``tree.routes(solved_only=False)`` enumerates one unfinished route per unsolved
    leaf, and most of those are one step deep, so filter before you draw
    (``[r for r in tree.routes(solved_only=False) if len(r) > 2]``).

    Each route gets one drawing and one step list; a step's number is the number on
    its disc in the drawing, 1 being the first reaction performed. Unresolved
    leaves -- the molecules the search could not buy -- are drawn in the red ``oos``
    role. The page is self-contained: no scripts, no external stylesheets, no fonts
    to fetch.

    :param routes: The routes to draw.
    :param html_path: The path to the file where to store resulting HTML. When None,
        the page is returned instead of written.
    :param aam: If True, depict atom-to-atom mapping.
    :return: The page when ``html_path`` is None, otherwise None.
    """
    depict_settings(aam=bool(aam))
    routes = list(routes)

    doc = Doc()
    layouts: dict = {}  # one geometry per molecule, so a card matches its neighbours
    body = []
    for index, route in enumerate(routes, 1):
        rows = ""
        for number, step in enumerate(route, 1):
            label = _step_label(step)
            rows += (
                f'<div class="step"><div class="disc">{number}</div><div>'
                + (f'<div class="lab">{escape(label)}</div>' if label else "")
                + f'<div class="rxn mono">{escape(str(step.reaction))}</div></div></div>'
            )
        provenance = route.provenance
        node_id = None if provenance is None else provenance.tree_node_id
        score = None if provenance is None else provenance.search_score
        unresolved = len(route.unresolved)
        body.append(
            '<section class="route card"><div class="rhead">'
            f'<div class="kv"><div class="eyebrow">Route</div>'
            f'<div class="v id">{index if node_id is None else node_id}</div></div>'
            f'<div class="kv"><div class="eyebrow">Steps</div>'
            f'<div class="v">{len(route)}</div></div>'
            f'<div class="kv"><div class="eyebrow">Search score</div>'
            f'<div class="v">{"—" if score is None else round(score, 3)}</div></div>'
            f'<div class="kv"><div class="eyebrow">Not in stock</div>'
            f'<div class="v">{unresolved}</div></div></div>'
            f'<div class="draw">'
            f"{doc.route(route.svg(standalone=False, layouts=layouts))}"
            f"</div>{rows}</section>"
        )

    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Retrosynthetic Routes Report</title>\n"
        f"<style>{ROUTE_CSS}{_REPORT_CSS}</style>\n</head>\n<body>\n"
        + hidden_defs(ARROW_DEFS, doc.defs())
        + '<div class="wrap">'
        + _report_header(
            routes,
            drawable_copy(routes[0].target, layouts) if routes else None,
        )
        + "".join(body)
        + "</div>\n</body>\n</html>\n"
    )

    if html_path is None:
        return page
    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(page)
    return None


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
    layouts: dict = {}  # one geometry per molecule, shared by every route on the page
    for cluster_num, group_data in clusters.items():
        route_ids = group_data.get("route_ids", [])
        if not route_ids:
            continue
        route_id = route_ids[0]
        # Get SVGs
        svg = Route.from_tree(tree, route_id).svg(layouts=layouts)
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
            route_node = routes_json.get(key) or routes_json.get(str(key), {})
            target_smiles = route_node.get("smiles", "N/A")
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

    sb_cgr = group.get("sb_cgr")
    if sb_cgr is None and first_route_id is not None and sb_cgrs_dict:
        sb_cgr = sb_cgrs_dict.get(first_route_id)

    if sb_cgr is not None:
        try:
            sb_cgr.clean2d()
            sb_cgr_svg = cgr_display(sb_cgr)

            if sb_cgr_svg.strip().startswith("<svg"):
                table += f"<tr>{td}{font_normal}Identified Strategic Bonds{font_close}<br>{sb_cgr_svg}</td></tr>"
            else:
                table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR:{font_close}<br><i>Invalid SVG format retrieved.</i></td></tr>"
                print(
                    f"Warning: Expected SVG for SB-CGR of route {first_route_id}, but got: {sb_cgr_svg[:100]}..."
                )
        except Exception as e:
            table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR:{font_close}<br><i>Error retrieving/displaying SB-CGR: {e}</i></td></tr>"
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
    layouts: dict = {}  # one geometry per molecule, shared by every route on the page
    for route_id in valid_routes:
        if using_tree:
            # 1) SVG from Tree
            route = Route.from_tree(tree, route_id)
            svg = route.svg(layouts=layouts)
            # 2) Reaction steps & score; step order is the drawing's disc order
            steps = [step.reaction for step in route]
            score = round(route.provenance.search_score, 3)
            # build reaction list
            reac_html = "".join(
                f"<b>Step {i + 1}:</b> {r!s}<br>" for i, r in enumerate(steps)
            )
            header = f"Route {route_id} — {len(steps)} steps, score={score}"
            table += f"<tr><td><b>{header}</b></td></tr>"
            table += f"<tr><td>{svg}</td></tr>"
            table += f"<tr><td>{reac_html}</td></tr>"
        else:
            # 1) SVG from JSON
            svg = _json_route(routes_json, route_id).svg(layouts=layouts)
            steps = routes_dict[route_id]
            reac_html = "".join(
                f"<b>Step {i + 1}:</b> {r!s}<br>" for i, r in steps.items()
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


def _has_table_values(data: dict) -> bool:
    return any(bool(row) for row in data.values())


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
                    html += depict_value(route_data[mark])
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
                        html += depict_value(route_data[mark])
                    html += "</td>"
                html += "</tr>"
            else:
                # Optionally, you can note that the route_id was not found
                html += f"<tr><td colspan='{len(all_marks) + 1}' style='border: 1px solid black; padding: 4px; color:red;'>Route ID {route_id} not found.</td></tr>"

    html += "</table>"

    if if_display:
        display(HTML(html))

    return html


def supporting_table_2_html(subcluster, routes_to_display=None, if_display=True):
    """Generate an HTML table for supporting pseudo-reactants marked as Y."""

    if routes_to_display is None:
        routes_to_display = []

    supporting_data = subcluster.get("supporting_data", {})
    if not _has_table_values(supporting_data):
        return ""

    all_marks = sorted(
        {mark for route_data in supporting_data.values() for mark in route_data}
    )
    html = "<table style='border-collapse: collapse;'><tr><th style='border: 1px solid black; padding: 4px;'>Route ID</th>"
    for mark in all_marks:
        html += f"<th style='border: 1px solid black; padding: 4px;'>Y<small>{mark}</small></th>"
    html += "</tr>"

    route_ids = routes_to_display or list(supporting_data)
    for route_id in route_ids:
        route_data = supporting_data.get(route_id)
        if route_data is None:
            html += f"<tr><td colspan='{len(all_marks) + 1}' style='border: 1px solid black; padding: 4px; color:red;'>Route ID {route_id} not found.</td></tr>"
            continue

        html += (
            f"<tr><td style='border: 1px solid black; padding: 4px;'>{route_id}</td>"
        )
        for mark in all_marks:
            html += "<td style='border: 1px solid black; padding: 4px;'>"
            if mark in route_data:
                html += depict_value(route_data[mark])
            html += "</td>"
        html += "</tr>"

    html += "</table>"

    if if_display:
        display(HTML(html))

    return html


def group_lg_table_2_html_fixed(
    grouped: dict,
    groups_to_display=None,
    if_display=False,
    max_group_col_width: int = 200,
    mark_prefix: str = "X",
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
        f"<th style='border:1px solid #ccc; padding:4px; text-align:center;'>{mark_prefix}<small>{mark}</small></th>"
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
                cell.append(depict_value(rep[mark]))
            cell.append("</td>")
            row.append("".join(cell))
        html.append("<tr>" + "".join(row) + "</tr>")

    html.append("</tbody></table>")
    out = "".join(html)
    if if_display:
        display(HTML(out))

    return out


def group_supporting_table_2_html_fixed(
    grouped: dict,
    groups_to_display=None,
    if_display=False,
    max_group_col_width: int = 200,
) -> str:
    """Generate a grouped HTML table for supporting pseudo-reactants marked as Y.

    Unlike the X table, an empty or all-empty ``grouped`` renders nothing rather
    than an empty table.
    """

    if not grouped or not _has_table_values(grouped):
        return ""

    return group_lg_table_2_html_fixed(
        grouped,
        groups_to_display,
        if_display,
        max_group_col_width,
        mark_prefix="Y",
    )


def routes_subclustering_report(
    source: Tree | dict,
    subcluster: dict,
    group_index: str | None = None,
    cluster_num: int | str | None = None,
    sb_cgrs_dict: dict | None = None,
    if_lg_group: bool | None = None,
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
                           'sb_cgr', and optionally 'group_lgs' if
                           `if_lg_group` is True.
        group_index (str, optional): Main cluster ID. If omitted, the value is
                                     read from `subcluster['cluster_id']`.
        cluster_num (int | str, optional): Subcluster ID. If omitted, the value
                                           is read from
                                           `subcluster['subcluster_id']`.
        sb_cgrs_dict (dict, optional): Legacy route-ID to SB-CGR mapping. If
                                       omitted, `subcluster['sb_cgr']` is used.
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

    group_index = (
        group_index if group_index is not None else subcluster.get("cluster_id")
    )
    cluster_num = (
        cluster_num if cluster_num is not None else subcluster.get("subcluster_id")
    )
    group_index = group_index or "?"
    cluster_num = cluster_num or "?"
    if if_lg_group is None:
        if_lg_group = bool(subcluster.get("group_lgs"))

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
            if nid in routes_json or str(nid) in routes_json:
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
            route_node = routes_json.get(key) or routes_json.get(str(key), {})
            target_smiles = route_node.get("smiles", "N/A")
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

    sb_cgr = subcluster.get("sb_cgr")
    if sb_cgr is None and first_route_id is not None and sb_cgrs_dict:
        sb_cgr = sb_cgrs_dict.get(first_route_id)

    if sb_cgr is not None:
        try:
            sb_cgr.clean2d()
            sb_cgr_svg = cgr_display(sb_cgr)

            if sb_cgr_svg.strip().startswith("<svg"):
                table += f"<tr>{td}{font_normal}Identified Strategic Bonds{font_close}<br>{sb_cgr_svg}</td></tr>"
            else:
                table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR:{font_close}<br><i>Invalid SVG format retrieved.</i></td></tr>"
                print(
                    f"Warning: Expected SVG for SB-CGR of route {first_route_id}, but got: {sb_cgr_svg[:100]}..."
                )
        except Exception as e:
            table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR:{font_close}<br><i>Error retrieving/displaying SB-CGR: {e}</i></td></tr>"
    else:
        if first_route_id:
            table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR (from Route {first_route_id}):{font_close}<br><i>Not found in provided SB-CGR dictionary.</i></td></tr>"
        else:
            table += f"<tr>{td}{font_normal}Cluster Representative SB-CGR:{font_close}<br><i>No valid routes in cluster to select from.</i></td></tr>"

    try:
        synthon_reaction = subcluster["synthon_reaction"]
        synthon_svg = depict_custom_reaction(synthon_reaction)

        extra_synthon = f"<tr>{td}{font_normal}Synthon pseudo reaction:{font_close}<br>{synthon_svg}</td></tr>"
        table += extra_synthon
    except Exception as e:
        table += f"<tr><td colspan='1' style='color: red;'>Error displaying synthon reaction: {e}</td></tr>"

    try:
        if if_lg_group:
            grouped_lgs = subcluster["group_lgs"]
            lg_table_html = group_lg_table_2_html_fixed(grouped_lgs, if_display=False)
            supporting_table_html = group_supporting_table_2_html_fixed(
                subcluster.get("group_supporting", {}), if_display=False
            )
            if not supporting_table_html:
                supporting_table_html = supporting_table_2_html(
                    subcluster, if_display=False
                )
        else:
            lg_table_html = lg_table_2_html(subcluster, if_display=False)
            supporting_table_html = supporting_table_2_html(
                subcluster, if_display=False
            )
        table_sections = [
            (
                "Leaving Groups table:",
                lg_table_html,
            )
        ]
        if supporting_table_html:
            table_sections.append(("Supporting Groups table:", supporting_table_html))
        sections_html = "".join(
            f"<div style='flex:1 1 360px; min-width:300px;'>{font_normal}{title}{font_close}<br>{section_html}</div>"
            for title, section_html in table_sections
        )
        extra_lg = f"<tr>{td}<div style='display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap;'>{sections_html}</div></td></tr>"
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
    layouts: dict = {}  # one geometry per molecule, shared by every route on the page
    for route_id in valid_routes:
        if using_tree:
            # 1) SVG from Tree
            route = Route.from_tree(tree, route_id)
            svg = route.svg(layouts=layouts)
            # 2) Reaction steps & score; step order is the drawing's disc order
            steps = [step.reaction for step in route]
            score = round(route.provenance.search_score, 3)
            # build reaction list
            reac_html = "".join(
                f"<b>Step {i + 1}:</b> {r!s}<br>" for i, r in enumerate(steps)
            )
            header = f"Route {route_id} — {len(steps)} steps, score={score}"
            table += f"<tr><td><b>{header}</b></td></tr>"
            table += f"<tr><td>{svg}</td></tr>"
            table += f"<tr><td>{reac_html}</td></tr>"

        else:
            # 1) SVG from JSON
            svg = _json_route(routes_json, route_id).svg(layouts=layouts)
            steps = routes_dict[route_id]
            reac_html = "".join(
                f"<b>Step {i + 1}:</b> {r!s}<br>" for i, r in steps.items()
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
