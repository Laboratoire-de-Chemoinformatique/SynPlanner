"""The planning report page: self-contained, one drawing per route, numbers that agree.

Everything is read back out of the rendered page, never out of the values the
renderer happened to return.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from chython import smiles as read_smiles

from synplan.chem.reaction.reactor import Reaction
from synplan.utils.visualisation import generate_results_html

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


class _FakeTree:
    """A two-route tree whose precursors are real substructures of the target.

    A substructure keeps its parent's atom numbers and coordinates, which is what
    chython's reactor hands back for a real disconnection — so alignment, depiction
    and layout all see the same shapes they see in a planning run.
    """

    def __init__(self) -> None:
        target = read_smiles("CC(=O)Nc1ccccc1CCO")
        target.clean2d()
        acid = target.substructure([1, 2, 3])
        amine = target.substructure([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        aniline = amine.substructure([4, 5, 6, 7, 8, 9, 10])
        alcohol = amine.substructure([11, 12, 13])
        ring = target.substructure([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        tail = target.substructure([11, 12, 13])

        self.target = target
        self._routes = {
            3: (
                Reaction([aniline, alcohol], [amine]),
                Reaction([acid, amine], [target]),
            ),
            5: (Reaction([ring, tail], [target]),),
        }
        self.building_blocks = frozenset(str(m) for m in (acid, aniline, alcohol, tail))
        self.nodes = {i: _FakeNode(target if i == 1 else None) for i in (1, 2, 3, 5)}
        self.parents = {1: 0, 2: 1, 3: 2, 5: 1}
        self.winning_nodes = [3, 5]
        self.visited_nodes = [1, 2, 3, 5]
        self.curr_time = 1.2345
        self.config = SimpleNamespace(min_mol_size=0)

    def __len__(self) -> int:
        return len(self.nodes)

    def synthesis_route(self, node_id: int):
        return self._routes[node_id]

    def route_score(self, node_id: int) -> float:
        return 0.5


class _FakeNode:
    def __init__(self, molecule) -> None:
        self.curr_precursor = molecule

    def is_solved(self) -> bool:
        return True


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("report") / "report.html"
    assert generate_results_html(_FakeTree(), str(path)) is None
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


def test_page_keeps_the_run_summary(page):
    tree = _FakeTree()
    assert "Predicted Paths Report" in page
    assert "Retrosynthetic routes report" in page
    assert str(tree.target) in page
    for label, value in (
        ("Tree size", len(tree)),
        ("Visited nodes", len(tree.visited_nodes)),
        ("Found paths", len(tree.winning_nodes)),
        ("Time", round(tree.curr_time, 4)),
    ):
        assert label in page
        assert f">{value}<" in page
    for role in ("Target molecule", "Intermediate", "Not in stock", "In stock"):
        assert role in page


def test_one_drawing_per_route(page):
    tree = _FakeTree()
    found = sections(page)
    assert len(found) == len(tree.winning_nodes)
    for node_id, section in zip(tree.winning_nodes, found):
        assert f'<div class="v id">{node_id}</div>' in section
        assert drawing(section).startswith("<svg ")
        assert drawing(section).count('viewBox="0 0 ') == 1
        assert f'<div class="v">{len(tree.synthesis_route(node_id))}</div>' in section
        assert "Cumulated nodes&#39; value" in section


def test_step_numbers_match_the_discs(page):
    tree = _FakeTree()
    for node_id, section in zip(tree.winning_nodes, sections(page)):
        n_steps = len(tree.synthesis_route(node_id))
        discs = _DISC.findall(drawing(section))
        assert [int(d[4]) for d in discs] == list(range(1, n_steps + 1))
        assert [int(n) for n in _STEP_NUMBER.findall(section)] == [
            int(d[4]) for d in discs
        ]


def test_disc_one_is_the_cut_from_the_target(page):
    """Read from the drawing: disc 1's arrow ends on the box captioned TARGET."""
    for section in sections(page):
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

        cx, cy, _, _, number = next(iter(_DISC.findall(svg)))
        assert int(number) == 1
        assert float(cx) < box_x  # the disc sits in the lane left of the target
        assert abs(float(cy) - (box_y + height / 2)) < 0.11
