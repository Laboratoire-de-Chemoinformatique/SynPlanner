"""The routes report page: self-contained, one drawing per route, numbers that agree.

The page is handed routes, so the tests hand it routes -- no tree stands in.
Everything is read back out of the rendered page, never out of the values the
renderer happened to return.
"""

from __future__ import annotations

import re

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
    assert "<script" not in page


def test_page_summarises_the_routes_it_was_handed(page, routes):
    assert "Retrosynthetic Routes Report" in page
    assert "Retrosynthetic routes report" in page
    assert str(routes[0].target) in page
    for label, value in (
        ("Routes", len(routes)),
        ("Solved", sum(route.solved for route in routes)),
        ("Longest", max(len(route) for route in routes)),
        ("Best score", 0.5),
    ):
        assert label in page
        assert f">{value}<" in page
    for role in ("Target molecule", "Intermediate", "Not in stock", "In stock"):
        assert role in page


def test_one_drawing_per_route(page, routes):
    found = sections(page)
    assert len(found) == len(routes)
    for route, section in zip(routes, found):
        assert f'<div class="v id">{route.provenance.tree_node_id}</div>' in section
        assert drawing(section).startswith("<svg ")
        assert drawing(section).count('viewBox="0 0 ') == 1
        assert f'<div class="v">{len(route)}</div>' in section
        assert f'<div class="v">{route.provenance.search_score}</div>' in section


def test_step_numbers_match_the_discs(page, routes):
    for route, section in zip(routes, sections(page)):
        discs = sorted(int(d[4]) for d in _DISC.findall(drawing(section)))
        assert discs == list(range(1, len(route) + 1))
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
    assert section.count("—") == 1  # no search score on the card
    assert page.count("—") == 2  # nor in the summary


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
    assert _STEP_LABEL.findall(routes_report_html([labelled], None)) == ["ugi:7"]


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
    """The leaf the search could not buy is drawn in the red role and counted."""
    route, unresolved = unsolved_route()
    assert not route.solved
    (section,) = sections(unsolved_page)
    svg = drawing(section)
    red = ROLE_STYLE["oos"][1]
    assert svg.count(f'stroke="{red}"') == len(unresolved)
    assert svg.count(">NOT IN STOCK</text>") == len(unresolved)
    assert svg.count(">IN STOCK</text>") == 1  # the other leaf is purchasable
    assert f'<div class="v">{len(unresolved)}</div>' in section  # "Not in stock"
    assert ">0<" in unsolved_page  # nothing solved, in the summary


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
