"""The routes report page: self-contained, one drawing per route, numbers that agree.

The page is handed routes, so the tests hand it routes -- no tree stands in.
Everything is read back out of the rendered page, never out of the values the
renderer happened to return.
"""

from __future__ import annotations

import json
import re
from html import unescape

import pytest
from chython import smiles as read_smiles

from synplan.chem.reaction.reactor import Reaction
from synplan.chem.reaction.routes.route import Route, RouteProvenance, Step, StepOrigin
from synplan.chem.utils import molecule_key
from synplan.utils.routedraw import ROLE_STYLE
from synplan.utils.visualisation import routes_report_html

# The two namespace URIs an SVG must declare. They name a standard, nothing is fetched.
_NAMESPACES = ("http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink")

_NUM = r"(-?[\d.]+)"
_DISC = re.compile(
    rf'<circle cx="{_NUM}" cy="{_NUM}" r="10.5"[^>]*/>'
    rf'<text x="{_NUM}" y="{_NUM}" class="sp-num">(\d+)</text>'
)
_RECT = re.compile(
    rf'<rect x="{_NUM}" y="{_NUM}" width="{_NUM}" height="{_NUM}"[^>]*/>'
)
_TARGET_CAPTION = re.compile(
    rf'<text x="{_NUM}" y="{_NUM}" class="sp-tag" fill="[^"]*">TARGET</text>'
)
_STEP_NUMBER = re.compile(r'<div class="disc">(\d+)</div>')
_STEP_LABEL = re.compile(r'<div class="lab">([^<]*)</div>')
#: A priced leaf's pill: the offers it carries, and the price it shows.
_PILL = re.compile(
    r'<g class="sp-price" data-offers="([^"]*)">.*?<text[^>]*>([^<]*)</text>'
)
#: Every id a drawing reaches for: a pooled molecule, or the shared arrowhead.
_REFERENCE = re.compile(r'(?:xlink:)?href="#([^"]+)"|url\(#([^"]+)\)')
_DEPICTION = re.compile(
    rf'<svg x="{_NUM}" y="{_NUM}"[^>]*><use xlink:href="#([^"]+)"/>'
)


def route_of(
    reactions,
    node_id: int | None = None,
    score: float | None = None,
    unresolved=(),
    origins=None,
) -> Route:
    """One route out of reactions, the way a caller hands the report one."""

    if origins is None:
        origins = [StepOrigin(tree_node_id=node_id)] * len(reactions)
    steps = tuple(
        Step(reaction, reaction.products[0], origin)
        for reaction, origin in zip(reactions, origins)
    )
    return Route(
        steps=steps,
        unresolved=frozenset(molecule_key(mol) for mol in unresolved),
        provenance=RouteProvenance(score, node_id),
    )


def acetanilide_routes() -> list[Route]:
    """Two routes to one target, whose precursors are real substructures of it.

    A substructure keeps its parent's atom numbers and coordinates, which is what
    chython's reactor hands back for a real disconnection — so alignment, depiction
    and layout all see the same shapes they see in a planning run.
    """
    target = read_smiles("CC(=O)Nc1ccccc1CCO")
    target.clean2d()
    acid = target.substructure([1, 2, 3])
    amine = target.substructure([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    aniline = amine.substructure([4, 5, 6, 7, 8, 9, 10])
    alcohol = amine.substructure([11, 12, 13])
    ring = target.substructure([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    tail = target.substructure([11, 12, 13])

    return [
        route_of(
            (
                Reaction([aniline, alcohol], [amine]),
                Reaction([acid, amine], [target]),
            ),
            node_id=3,
            score=0.5,
        ),
        route_of((Reaction([ring, tail], [target]),), node_id=5, score=0.25),
    ]


@pytest.fixture(scope="module")
def routes() -> list[Route]:
    return acetanilide_routes()


@pytest.fixture(scope="module")
def page(tmp_path_factory, routes) -> str:
    path = tmp_path_factory.mktemp("report") / "report.html"
    assert routes_report_html(routes, str(path)) is None
    return path.read_text(encoding="utf-8")


def offers_by_key(page: str) -> dict[str, list[list[str]]]:
    """Every chip's offer rows on the page, keyed by the record the chip names."""
    payloads = (json.loads(unescape(data)) for data, _ in _PILL.findall(page))
    return {data["title"]: data["rows"] for data in payloads}


def sections(page: str) -> list[str]:
    return page.split('<section class="route card">')[1:]


def drawing(section: str) -> str:
    return section.split('<div class="draw">')[1].split("</div>")[0]


def test_page_is_self_contained(page):
    """No stylesheet, script or font to fetch: the file works with no network."""
    for namespace in _NAMESPACES:
        page = page.replace(namespace, "")
    assert "http://" not in page
    assert "https://" not in page
    assert "<script src" not in page


def test_each_route_offers_zoom_and_export_as_buttons(page):
    """The drawing carries chips, so every control it had is a button of its own."""
    for section in sections(page):
        (acts,) = re.findall(r'<div class="acts">.*?</div>', section)
        assert re.findall(r'data-act="(\w+)"', acts) == ["zoom", "svg", "png"]
    assert "cursor:zoom-in" not in page  # the drawing is not the control any more


def test_cluster_report_reuses_standalone_drawings_without_parsing(routes, monkeypatch):
    from synplan.utils.visualisation import routes_clustering_report

    rendered = {}
    routes_report_html(routes, None, rendered_routes=rendered)
    source = {str(r.provenance.tree_node_id): r.to_json() for r in routes}
    ids = list(source)
    clusters = {"test": {"route_ids": ids}}
    for value in rendered.values():
        defined = set(re.findall(r'\bid="([^"]+)"', value["svg"]))
        used = {a or b for a, b in _REFERENCE.findall(value["svg"])}
        assert used <= defined
        assert not value["aam"] and '-mapping"' not in value["svg"]

    def unexpected(*args, **kwargs):
        raise AssertionError("A cached cluster report reparsed or redrew a route")

    monkeypatch.setattr(Route, "from_json", unexpected)
    monkeypatch.setattr(Route, "svg", unexpected)
    html = routes_clustering_report(source, clusters, "test", {}, rendered_routes=rendered)
    for rid in ids:
        assert rendered[rid]["svg"] in html
        assert f"Route {rid} — {len(rendered[rid]['steps'])} steps" in html
    with pytest.raises(AssertionError, match="reparsed"):
        routes_clustering_report(source, clusters, "test", {}, aam=True, rendered_routes=rendered)


def test_every_drawing_reference_is_defined_on_the_page(page):
    """Export re-inlines what a drawing points at, so nothing may point off-page."""
    defined = set(re.findall(r'<(?:g|marker)[^>]*\bid="([^"]+)"', page))
    for section in sections(page):
        found = {a or b for a, b in _REFERENCE.findall(drawing(section))}
        assert found
        assert found <= defined


def test_page_summarises_the_routes_it_was_handed(page, routes):
    assert "Retrosynthetic Routes Report" in page
    assert "<h1>SynPlanner retrosynthesis results</h1>" in page
    assert str(routes[0].target) in page
    for label, value in (
        ("Routes", len(routes)),
        ("Shortest route", min(len(route) for route in routes)),
        ("Best score", 0.5),
    ):
        assert label in page
        assert f">{value}<" in page
    for role in ("Target molecule", "Intermediate", "Not in stock", "In stock"):
        assert role in page


def test_search_time_comes_from_the_search_not_the_routes(routes):
    """Routes carry no clock, so the page shows a time only when it is handed one."""
    assert ">—<" in routes_report_html(routes, None).split('<div class="legend">')[0]
    page = routes_report_html(routes, None, stats={"search_time": 63.4})
    assert ">63<" in page
    assert ">0.4<" in routes_report_html(routes, None, stats={"search_time": 0.42})


def test_one_drawing_per_route(page, routes):
    found = sections(page)
    assert len(found) == len(routes)
    for route, section in zip(routes, found):
        assert f'<div class="v id">{route.provenance.tree_node_id}</div>' in section
        assert drawing(section).startswith("<svg ")
        assert drawing(section).count('viewBox="0 0 ') == 1
        assert f'<div class="v">{len(route)}</div>' in section
        assert f'<div class="v">{route.provenance.search_score}</div>' in section


def test_discs_number_the_steps_of_the_route(page, routes):
    for route, section in zip(routes, sections(page)):
        discs = sorted(int(d[4]) for d in _DISC.findall(drawing(section)))
        assert discs == list(range(1, len(route) + 1))


def test_the_smiles_list_numbers_the_same_steps_the_discs_do(routes):
    """Every step is listed under its drawing, numbered to match its disc."""
    for section in sections(routes_report_html(routes, None)):
        discs = sorted(int(d[4]) for d in _DISC.findall(drawing(section)))
        assert [int(n) for n in _STEP_NUMBER.findall(section)] == discs


def test_the_last_disc_is_the_cut_from_the_target(page, routes):
    """Read from the drawing: the highest-numbered disc's arrow ends on TARGET."""
    for route, section in zip(routes, sections(page)):
        svg = drawing(section)
        caption = _TARGET_CAPTION.search(svg)
        assert caption is not None
        box_x, box_y = float(caption.group(1)) - 1, float(caption.group(2)) + 5

        target_box = [
            r
            for r in _RECT.findall(svg)
            if abs(float(r[0]) - box_x) < 0.11 and abs(float(r[1]) - box_y) < 0.11
        ]
        assert len(target_box) == 1
        _, _, _, height = (float(v) for v in target_box[0])

        cx, cy, _, _, number = max(_DISC.findall(svg), key=lambda d: int(d[4]))
        assert int(number) == len(route)
        assert float(cx) < box_x  # the disc sits in the lane left of the target
        assert abs(float(cy) - (box_y + height / 2)) < 0.11


def test_the_page_draws_whatever_it_is_handed(routes):
    """One route in, one route on the page: the report picks nothing itself."""
    page = routes_report_html(routes[1:], None)
    assert len(sections(page)) == 1
    assert '<div class="v id">5</div>' in page
    assert ">1<" in page  # one route in the summary


def test_a_route_with_nothing_behind_it_still_draws(routes):
    """A route read back out of a file carries no search: no id, no score."""
    bare = Route(steps=routes[1].steps)
    page = routes_report_html([bare], None)
    (section,) = sections(page)
    assert '<div class="v id">1</div>' in section  # its position on the page
    # No search score on the card, and no catalogue behind it to price.
    assert '<div class="eyebrow">Search score</div><div class="v">—</div>' in section
    assert "Price per g of target" not in section
    summary = page.split('<section class="route card">')[0]
    assert summary.count("—") == 2  # nor a best score or a search time


def test_the_report_names_the_curated_rule_behind_a_step(routes):
    """A priority step is labelled by its rule key; a policy step stays unlabelled."""
    reactions = [step.reaction for step in routes[0].steps]
    labelled = route_of(
        reactions,
        node_id=3,
        origins=[
            StepOrigin(rule_key="ugi:7", rule_source="ugi", rule_id=7),
            StepOrigin(rule_key="policy:412", rule_source="policy", rule_id=412),
        ],
    )
    page = routes_report_html([labelled], None)
    assert _STEP_LABEL.findall(page) == ["ugi:7"]


def unsolved_route() -> tuple[Route, tuple]:
    """A one-step route whose two leaves are both dead ends."""
    target = read_smiles("CC(=O)Nc1ccccc1CCO")
    target.clean2d()
    left = target.substructure([1, 2, 3])
    right = target.substructure([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    return route_of(
        (Reaction([left, right], [target]),), node_id=9, score=0.1, unresolved=(right,)
    ), (right,)


@pytest.fixture(scope="module")
def unsolved_page() -> str:
    route, _ = unsolved_route()
    return routes_report_html([route], None)


def test_an_unsolved_route_shows_its_dead_ends(unsolved_page):
    """The drawing is the only place a dead end is reported: red role, red caption."""
    route, unresolved = unsolved_route()
    assert not route.solved
    (section,) = sections(unsolved_page)
    svg = drawing(section)
    red = ROLE_STYLE["oos"][1]
    assert svg.count(f'stroke="{red}"') == len(unresolved)
    assert svg.count(">NOT IN STOCK</text>") == len(unresolved)
    assert svg.count(">IN STOCK</text>") == 1  # the other leaf is purchasable


def _cut(mol, n: int, m: int) -> list:
    """The two fragments the bond ``n-m`` splits ``mol`` into.

    Substructures, so the atom numbers and coordinates are the target's own, which is
    what chython's reactor hands back for a real disconnection.
    """
    rest = mol.copy()
    rest.delete_bond(n, m)
    parts = rest.connected_components
    assert len(parts) == 2
    return [mol.substructure(part) for part in parts]


@pytest.fixture(scope="module")
def big_target_page() -> str:
    """Three one-step cuts of a target big enough to expose the layout lottery.

    chython lays this molecule out a fresh way nearly every time it is asked, so a
    page that lays every card out on its own shows three different targets.
    """
    target = read_smiles("CC(C)(C)OC(=O)NC1(C(=O)O)CCN(C(=O)OCc2ccccc2)CC1")
    target.clean2d()
    routes = [
        route_of((Reaction(_cut(target, *bond), [target]),), node_id=node_id)
        for node_id, bond in ((3, (6, 8)), (5, (2, 5)), (7, (16, 18)))
    ]
    return routes_report_html(routes, None)


def test_every_card_draws_the_target_the_same_way(big_target_page):
    """One layout per molecule, shared by the whole report.

    Read from the page: the depiction sitting inside the TARGET box is the same
    pooled one on every card, so a chemist scanning the cards compares like with
    like.
    """
    used = set()
    for section in sections(big_target_page):
        svg = drawing(section)
        caption = _TARGET_CAPTION.search(svg)
        assert caption is not None
        box_x, box_y = float(caption.group(1)) - 1, float(caption.group(2)) + 5
        box = [
            r
            for r in _RECT.findall(svg)
            if abs(float(r[0]) - box_x) < 0.11 and abs(float(r[1]) - box_y) < 0.11
        ]
        assert len(box) == 1
        width, height = float(box[0][2]), float(box[0][3])
        inside = [
            pool_id
            for x, y, pool_id in _DEPICTION.findall(svg)
            if box_x <= float(x) <= box_x + width
            and box_y <= float(y) <= box_y + height
        ]
        assert len(inside) == 1  # the boxes do not overlap
        used.add(inside[0])
    assert len(sections(big_target_page)) == 3
    assert len(used) == 1


def priced_route() -> Route:
    """A two-leaf route whose leaves carry the stock records a search selects."""
    (route,) = acetanilide_routes()[1:]
    (first, second) = route.leaves()
    first.meta["selected_stock"] = {
        "inchikey": "AAAAAAAAAAAAAA-UHFFFAOYSA-N",
        "smiles": str(first),
        "sources": [{"vendor": "MP", "ppg": "3.0"}, {"vendor": "CS", "ppg": "11.0"}],
    }
    second.meta["selected_stock"] = {
        "inchikey": "BBBBBBBBBBBBBB-UHFFFAOYSA-N",
        "smiles": str(second),
        "sources": [{"vendor": "MC", "ppg": "2.0"}],
    }
    return route


def test_every_chip_reads_the_same_and_carries_its_own_offers():
    """One label on all of them; what differs is behind the click, not on the chip."""
    page = routes_report_html([priced_route()], None)
    pills = _PILL.findall(page)
    assert [label for _, label in pills] == ["show price"] * 2
    assert offers_by_key(page) == {
        "AAAAAAAAAAAAAA-UHFFFAOYSA-N": [["MP", "3"], ["CS", "11"]],
        "BBBBBBBBBBBBBB-UHFFFAOYSA-N": [["MC", "2"]],
    }


def test_the_route_price_is_its_leaves_over_the_target_weight():
    """Cost per gram of target, one equivalent of each leaf at its cheapest offer."""
    route = priced_route()
    expected = sum(
        float(leaf.molecular_mass)
        * min(float(s["ppg"]) for s in leaf.meta["selected_stock"]["sources"])
        for leaf in route.leaves()
    ) / float(route.target.molecular_mass)
    page = routes_report_html([route], None)
    (shown,) = re.findall(
        r'<div class="eyebrow">Price per g of target</div><div class="v">([^<]*)</div>',
        page,
    )
    assert shown == f"{expected:g}"


def test_an_unpriced_leaf_leaves_the_route_without_a_price():
    """One leaf the search could not price and the route figure is not built at all."""
    route = priced_route()
    next(iter(route.leaves())).meta.pop("selected_stock")
    page = routes_report_html([route], None)
    assert (
        '<div class="eyebrow">Price per g of target</div><div class="v">—</div>' in page
    )
    assert len(_PILL.findall(page)) == 1  # the other leaf still wears its own


def test_prices_off_removes_the_pills_and_the_tile():
    """The switch takes out both halves, so the page never half-prices a route."""
    page = routes_report_html([priced_route()], None, prices=False)
    assert _PILL.findall(page) == []
    assert "Price per g of target" not in page


def test_vendor_names_come_from_the_catalogue_that_was_searched():
    """Given the catalogue, a pill names MolPort rather than repeating its code."""

    class Catalogue(dict):
        metadata = {"source_metadata": "vendors:\n  MP:\n    name: MolPort\n"}

    page = routes_report_html([priced_route()], None, building_blocks=Catalogue())
    page = routes_report_html([priced_route()], None, building_blocks=Catalogue())
    assert offers_by_key(page)["AAAAAAAAAAAAAA-UHFFFAOYSA-N"] == [
        ["MolPort", "3"],
        ["CS", "11"],
    ]


def test_a_price_under_one_keeps_its_leading_zero_through_the_page():
    """svgslim tightens geometry, not payloads: .25 is not what a catalogue said."""
    route = priced_route()
    next(iter(route.leaves())).meta["selected_stock"]["sources"] = [
        {"vendor": "MP", "ppg": "0.25"}
    ]
    page = routes_report_html([route], None)
    assert offers_by_key(page)["AAAAAAAAAAAAAA-UHFFFAOYSA-N"] == [["MP", "0.25"]]


def test_a_catalogue_record_with_no_offer_still_gets_a_pill():
    """The page never drops a selected material for having no price behind it."""
    route = priced_route()
    leaf = next(iter(route.leaves()))
    leaf.meta["selected_stock"]["sources"] = []
    page = routes_report_html([route], None)
    key = leaf.meta["selected_stock"]["inchikey"]
    assert offers_by_key(page)[key] == [["Price", "unavailable"]]
    assert (
        '<div class="eyebrow">Price per g of target</div><div class="v">—</div>' in page
    )


def test_a_pill_names_the_exact_material_it_priced():
    """The drawing shows a structure; the popup pins which catalogue record it is."""
    assert set(offers_by_key(routes_report_html([priced_route()], None))) == {
        "AAAAAAAAAAAAAA-UHFFFAOYSA-N",
        "BBBBBBBBBBBBBB-UHFFFAOYSA-N",
    }


def test_a_stock_that_can_never_price_anything_grows_no_price_column():
    """A legacy SMILES stock leaves no records, so the page drops the tile entirely."""
    page = routes_report_html(acetanilide_routes(), None)
    assert "Price per g of target" not in page


def test_the_page_stylesheet_parses(page):
    """An unbalanced brace silently swallows the rule after it.

    One stray ``}`` before ``:root`` costs the page every custom property, and with
    them every border and tint on it, while the text keeps rendering — so nothing
    looks broken enough to notice except that the page has lost all its lines.
    """
    style = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
    depth = 0
    for offset, character in enumerate(style):
        depth += (character == "{") - (character == "}")
        assert depth >= 0, f"stray }} at {offset}: ...{style[offset - 70 : offset + 1]}"
    assert depth == 0, "unclosed rule"
    # The properties every border and background on the page is written against.
    assert re.search(r":root\{[^}]*--rule:", style)
