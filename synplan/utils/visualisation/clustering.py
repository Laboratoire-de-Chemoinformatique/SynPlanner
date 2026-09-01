"""Reports over clustered routes: one page per cluster, one per subcluster."""

from __future__ import annotations

import base64
import contextlib
from datetime import datetime
from typing import TYPE_CHECKING

from chython import depict_settings
from IPython.display import HTML, display

from synplan.chem.reaction.routes.io import make_dict
from synplan.chem.reaction.routes.representation.depiction import (
    cgr_display,
    depict_custom_reaction,
)
from synplan.chem.reaction.routes.route import Route
from synplan.utils.frames import depict_value
from synplan.utils.visualisation.assets import (
    BOOTSTRAP_PAGE_HEAD,
    BOOTSTRAP_PAGE_TAIL,
    BOX_MARK,
)

if TYPE_CHECKING:
    from synplan.mcts.tree import Tree


def _json_route(routes_json: dict, route_id: int) -> Route:
    """One route out of a ``make_json`` mapping, by id.

    A mapping that has been through a JSON file spells its ids as strings, so both
    are tried before giving up.
    """
    node = routes_json.get(route_id, routes_json.get(str(route_id)))
    if node is None:
        raise ValueError(f"Route ID {route_id} not found in routes_json.")
    return Route.from_json(node)


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

    page_head = BOOTSTRAP_PAGE_HEAD.format(title=f"Cluster {group_index} Routes Report")

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
            <span>{BOX_MARK.replace("rgb()", "rgb(152, 238, 255)")} Target Molecule</span>
            <span>{BOX_MARK.replace("rgb()", "rgb(240, 171, 144)")} Molecule Not In Stock</span>
            <span>{BOX_MARK.replace("rgb()", "rgb(155, 250, 179)")} Molecule In Stock</span>
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

    html = page_head + table + BOOTSTRAP_PAGE_TAIL

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

    page_head = BOOTSTRAP_PAGE_HEAD.format(
        title=f"SubCluster {group_index}.{cluster_num} Routes Report"
    )

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
            <span>{BOX_MARK.replace("rgb()", "rgb(152, 238, 255)")} Target Molecule</span>
            <span>{BOX_MARK.replace("rgb()", "rgb(240, 171, 144)")} Molecule Not In Stock</span>
            <span>{BOX_MARK.replace("rgb()", "rgb(155, 250, 179)")} Molecule In Stock</span>
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

    html = page_head + table + BOOTSTRAP_PAGE_TAIL

    if html_path:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return f"Written to {html_path}"
    return html
