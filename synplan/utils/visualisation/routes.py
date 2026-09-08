"""Routes read out of a search tree, and the HTML report that draws them."""

from __future__ import annotations

import json
from html import escape
from typing import TYPE_CHECKING, Any

from chython.containers.molecule import MoleculeContainer

from synplan.chem.building_blocks import vendor_names
from synplan.chem.precursor import is_purchasable
from synplan.chem.reaction.routes.representation.depiction import (
    _temporary_render_config,
)
from synplan.chem.reaction.routes.route import Route, Step
from synplan.chem.reaction.rules.priority import POLICY_SOURCE_NAME
from synplan.utils.routedraw import (
    ARROW_DEFS,
    ROLE_STYLE,
    ROUTE_CSS,
    drawable_copy,
    molecule_svg,
)
from synplan.utils.svgslim import Doc, hidden_defs
from synplan.utils.visualisation.assets import (
    REPORT_CSS,
    REPORT_JS,
    ROLE_LEGEND,
)

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
                key=str(precursor),
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
    :param min_mol_size: Retained for API compatibility; stock membership is size-independent.
    :return: A list of dictionaries. Each dictionary contains a target, a list of
        children, and a boolean indicating whether the target is in building_blocks.
    """
    from synplan.mcts.tree import Tree

    if isinstance(tree, Tree):
        node_ids = (
            [i for i, node in tree.nodes.items() if i != 1 and node.is_solved()]
            if extended
            else tree.winning_nodes
        )
        if node_ids:
            return [Route.from_tree(tree, node_id).to_json() for node_id in node_ids]
    target = tree.nodes[1].precursors_to_expand[0].molecule
    target_in_stock = tree.nodes[1].curr_precursor.is_purchasable(tree.building_blocks)

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


def _seconds(value: float | None) -> str:
    """Whole seconds, tenths below one. A search timed to 0.1 s reads as 0.4, not 0."""
    if value is None:
        return "—"
    return f"{value:.0f}" if value >= 1 else f"{value:.1f}"


def _report_header(
    routes: Sequence[Route],
    tile: MoleculeContainer | None,
    stats: dict | None = None,
) -> str:
    scores = [
        route.provenance.search_score
        for route in routes
        if route.provenance is not None and route.provenance.search_score is not None
    ]
    search_time = None if stats is None else stats.get("search_time")
    tiles = (
        ("Routes", len(routes), ""),
        ("Search time", _seconds(search_time), " s" if search_time is not None else ""),
        ("Shortest route", min((len(route) for route in routes), default=0), " steps"),
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
        "<h1>SynPlanner retrosynthesis results</h1>"
        + target
        + '<div class="stats">'
        + "".join(
            f'<div class="stat"><span class="eyebrow">{name}</span>'
            f'<span class="v">{value}<span class="u">{unit}</span></span></div>'
            for name, value, unit in tiles
        )
        + '</div><div class="legend">'
        + "".join(
            f'<span class="chip"><span class="sw" style="background:{ROLE_STYLE[role][0]};'
            f'border:1px solid {ROLE_STYLE[role][1]}"></span>{caption}</span>'
            for role, caption in ROLE_LEGEND
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


def _route_prices(
    route: Route, names: dict[str, str]
) -> tuple[dict[int, tuple[str, str]], str]:
    """A pill per terminal leaf, and the route's own cost per gram.

    A leaf is priced from the catalogue record the search selected for it, so the
    cheapest offer here is the cheapest offer for that exact isomer. The route
    figure is :meth:`Route.calculate_cost`'s ``cost_per_gram`` computed from those
    same records: one molar equivalent per leaf occurrence at 100% yield, over the
    target's own weight. One leaf without a price and the route has no figure.

    A leaf the search only assumed away for being small gets a pill saying so.
    Its box is drawn in the green ``bb`` role like any other terminal, and that
    pill is the page's one place saying no catalogue backs it.
    """

    pills: dict[int, tuple[str, str]] = {}
    per_mol, complete = 0.0, True
    for leaf in route.leaves():
        selected = leaf.meta.get("selected_stock") or {}
        offers = sorted(
            (selected.get("vendors") or {}).items(), key=lambda item: (item[1], item[0])
        )
        if offers:
            per_mol += float(leaf.molecular_mass) * offers[0][1]
            rows = [[names.get(code, code), f"{price:g}"] for code, price in offers]
            note = "Per gram of this material, catalogue units."
        elif cutoff := leaf.meta.get("assumed_trivial"):
            rows = [["Assumed trivial", f"at most {int(cutoff)} atoms"]]
            note = (
                "No catalogue record: the search assumed a molecule this small "
                "is available."
            )
        elif selected:
            rows = [["Price", "unavailable"]]
            note = "The catalogue lists this material with no offer."
        else:
            complete = False
            continue
        complete = complete and bool(offers)
        pills[id(leaf)] = (
            # One label on every chip, so a row of them reads as one control rather
            # than as a row of numbers competing with the drawing.
            "show price",
            json.dumps(
                {"title": selected.get("inchikey", ""), "rows": rows, "note": note}
            ),
        )
    weight = float(route.target.molecular_mass)
    if not complete or weight <= 0.0:
        return pills, "—"
    return pills, f"{per_mol / weight:g}"


def routes_report_html(
    routes: Iterable[Route],
    html_path: str | None,
    aam: bool = False,
    *,
    stats: dict | None = None,
    prices: bool = True,
    building_blocks: Any = None,
) -> str | None:
    """Write an HTML page with the given routes drawn.

    Draws exactly the routes it is handed, in the order it is handed them --
    solved, unsolved, from one tree, from several, or read back out of a file.
    Which routes are worth a page is the caller's call:
    ``tree.routes(solved_only=False)`` enumerates one unfinished route per unsolved
    leaf, and most of those are one step deep, so filter before you draw
    (``[r for r in tree.routes(solved_only=False) if len(r) > 2]``).

    Each route gets one drawing; the disc a step is drawn on carries its number, 1
    being the first reaction performed. Unresolved leaves -- the molecules the
    search could not buy -- are drawn in the red ``oos`` role. Each route header
    carries Zoom (wheel to scale, drag to pan once open) and SVG/PNG export, which
    write the one route as a standalone file. Only those buttons open anything:
    clicking the drawing itself does nothing, so the chips on it stay clickable.
    The page is self-contained: nothing is fetched, and the only script is the
    inline one those buttons and chips run.

    :param routes: The routes to draw.
    :param html_path: The path to the file where to store resulting HTML. When None,
        the page is returned instead of written.
    :param aam: If True, depict atom-to-atom mapping.
    :param stats: ``Tree.to_stats_dict()`` (or ``SearchRecord.stats``), for the one
        fact the routes cannot carry: how long the search took. Routes read back
        from a v1 JSON file have no search behind them, so they have no time.
    :param prices: Put a pill under every leaf the search could price, and the
        route's own cost per gram beside its search score. Turn it off for a page
        that is about the chemistry, or for routes carrying no catalogue records.
    :param building_blocks: The catalogue the search ran against, read only for the
        vendor names its release metadata carries. Without it a pill names its
        vendors by the short codes the records hold.
    :return: The page when ``html_path`` is None, otherwise None.
    """
    vendors = (
        vendor_names(building_blocks) if prices and building_blocks is not None else {}
    )
    with _temporary_render_config(mapping=bool(aam)):
        routes = list(routes)
        # A legacy SMILES stock carries no vendors at all, so a page drawn against
        # one would wear a Price per gram column that can never hold anything.
        prices = prices and any(
            leaf.meta.get("selected_stock") or leaf.meta.get("assumed_trivial")
            for route in routes
            for leaf in route.leaves()
        )

        doc = Doc()
        layouts: dict = {}  # one geometry per molecule, so a card matches its neighbours
        body = []
        for index, route in enumerate(routes, 1):
            rows = ""
            stereo_status = route.stereo_status
            step_by_node = {
                s.origin.tree_node_id: i for i, s in enumerate(route.steps) if s.origin
            }
            issues_by_step = {}
            for obligation in (route.stereo or {}).get("obligations", ()):
                responsible = obligation.get("step")
                if responsible is None:
                    responsible = step_by_node.get(obligation.get("tree_node_id"))
                issues_by_step.setdefault(responsible, []).append(
                    {
                        "requires_stereo_forming_step": "Required stereochemistry must be established at this step; selectivity is unassessed.",
                        "requires_explicit_resolution_or_inversion_strategy": "An explicit stereo inversion or separation strategy is needed.",
                    }.get(obligation.get("reason"))
                    or obligation.get("detail")
                    or obligation.get("reason", "Stereo assessment needed")
                )
            for number, step in enumerate(route, 1):
                notes = list(issues_by_step.get(number - 1, ()))
                if stereo_status == "needs_reassessment" and (
                    notes or step.reaction.meta.get("stereo_events")
                ):
                    notes = ["Stereo needs reassessment after edits."]
                stereo_note = "".join(
                    f'<div class="rxn">Stereo: {escape(detail.replace("_", " "))}</div>'
                    for detail in dict.fromkeys(notes)
                )
                label = _step_label(step)
                rows += (
                    f'<div class="step"><div class="disc">{number}</div><div>'
                    + (f'<div class="lab">{escape(label)}</div>' if label else "")
                    + stereo_note
                    + f'<div class="rxn mono">{escape(str(step.reaction))}</div>'
                    + "</div></div>"
                )
            provenance = route.provenance
            pills, per_gram = _route_prices(route, vendors) if prices else ({}, "")
            cost = (
                '<div class="kv"><div class="eyebrow">Price per g of target</div>'
                f'<div class="v">{per_gram}</div></div>'
                if prices
                else ""
            )
            node_id = None if provenance is None else provenance.tree_node_id
            score = None if provenance is None else provenance.search_score
            body.append(
                '<section class="route card"><div class="rhead">'
                f'<div class="kv"><div class="eyebrow">Route ID</div>'
                f'<div class="v id">{index if node_id is None else node_id}</div></div>'
                f'<div class="kv"><div class="eyebrow">Steps</div>'
                f'<div class="v">{len(route)}</div></div>'
                f'<div class="kv"><div class="eyebrow">Search score</div>'
                f'<div class="v">{"—" if score is None else round(score, 3)}</div></div>'
                + cost
                + '<div class="acts"><button class="act" data-act="zoom">Zoom</button>'
                '<button class="act" data-act="svg">SVG</button>'
                '<button class="act" data-act="png">PNG</button></div></div>'
                f'<div class="draw">'
                f"{doc.route(route.svg(standalone=False, layouts=layouts, prices=pills))}"
                f"</div>{rows}</section>"
            )

        page = (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            "<title>Retrosynthetic Routes Report</title>\n"
            f"<style>{ROUTE_CSS}{REPORT_CSS}</style>\n</head>\n<body>\n"
            + hidden_defs(ARROW_DEFS, doc.defs())
            + '<div class="wrap">'
            + _report_header(
                routes,
                drawable_copy(routes[0].target, layouts) if routes else None,
                stats,
            )
            + "".join(body)
            + "</div>\n<script>"
            + REPORT_JS
            + "</script>\n</body>\n</html>\n"
        )

        if html_path is None:
            return page
        with open(html_path, "w", encoding="utf-8") as html_file:
            html_file.write(page)
        return None
