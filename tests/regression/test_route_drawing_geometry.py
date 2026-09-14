"""A route is drawn from a layout of its own, and the tree keeps its coordinates.

A route's molecules are the search tree's own objects, carrying whatever the reactor
left on them: a precursor sits where its atoms sat in the product, with every atom the
disconnection added piled on the origin. Drawn as they are, that is a collapsed cage --
a bond three times its neighbours', two atoms on one point, a Boc group folded into a
triangle. The drawer must lay out a copy, and may not write the result back.

The fixture is the coordinate handoff itself, not a search: a Boc-piperidine whose
product is laid out and whose Boc chloride precursor keeps the product's coordinates
with its new chlorine at the origin.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from math import hypot

import pytest
from chython import smiles as read_smiles
from chython.containers.molecule import MoleculeContainer

from synplan.chem.reaction.reactor import Reaction
from synplan.chem.reaction.routes.route import Route, Step

_SVG = "{http://www.w3.org/2000/svg}"

#: chython lays a bond out at 0.825 units and draws it shorter still wherever an atom
#: label clears the line. Anything past this is a molecule that was never laid out.
MAX_BOND = 1.5


def _elements(node):
    for child in node:
        yield child
        yield from _elements(child)


def bond_lengths(svg: str) -> list[float]:
    """Every visible bond line, in depiction units.

    Each molecule sits in its own nested ``<svg viewBox>``, so the numbers inside are
    the molecule's own coordinates whatever the page scales them to. The invisible
    spacers chython emits behind atom labels carry ``stroke="none"``.
    """

    return [
        hypot(
            float(el.get("x2")) - float(el.get("x1")),
            float(el.get("y2")) - float(el.get("y1")),
        )
        for el in _elements(ET.fromstring(svg))
        if el.tag == f"{_SVG}line" and el.get("stroke") != "none"
    ]


def coordinates(mol: MoleculeContainer) -> list[tuple[float, float]]:
    return [(atom.x, atom.y) for _, atom in mol.atoms()]


@pytest.fixture
def boc_piperidine() -> tuple[MoleculeContainer, MoleculeContainer, MoleculeContainer]:
    """``(product, Boc chloride, piperidine)`` with the reactor's coordinates.

    Cutting the carbamate of a laid-out product and hanging a chlorine off the
    carbonyl is exactly what the retro reactor hands back: the two precursors inherit
    the product's plane, and the atom the rule introduced has no place in it.
    """

    product = read_smiles("CC(C)(C)OC(=O)N1CCCCC1")
    product.clean2d()

    cut = product.copy()
    cut.delete_bond(6, 8)  # the C(=O)-N of the carbamate
    cut.add_bond(6, cut.add_atom("Cl"), 1)
    boc_chloride, piperidine = sorted(cut.split(), key=len, reverse=True)
    return product, boc_chloride, piperidine


@pytest.fixture
def route(boc_piperidine) -> Route:
    product, boc_chloride, piperidine = boc_piperidine
    return Route([Step(Reaction([boc_chloride, piperidine], [product]), product)])


def test_the_fixture_really_is_a_collapsed_precursor(boc_piperidine):
    """Without this the rest of the file would pass on a route that was never broken."""

    _, boc_chloride, _ = boc_piperidine
    stretched = [
        hypot(
            boc_chloride.atom(n).x - boc_chloride.atom(m).x,
            boc_chloride.atom(n).y - boc_chloride.atom(m).y,
        )
        for n, m, _ in boc_chloride.bonds()
    ]
    assert max(stretched) > MAX_BOND
    assert len(set(coordinates(boc_chloride))) < len(boc_chloride)  # two on one point


@pytest.mark.parametrize("align", [True, False])
def test_a_drawn_route_has_bonds_of_one_length(route, align):
    lengths = bond_lengths(route.svg(align=align))
    assert lengths, "no bonds were drawn"
    assert max(lengths) <= MAX_BOND


@pytest.mark.parametrize("compact", [False, True])
def test_drawing_leaves_the_route_molecules_where_the_reactor_put_them(route, compact):
    """The tree holds these very objects; the drawer may not relayout them."""

    molecules = [
        mol
        for step in route
        for mol in (*step.reaction.reactants, *step.reaction.products)
    ]
    before = {id(mol): coordinates(mol) for mol in molecules}
    route.svg(compact=compact)
    assert {id(mol): coordinates(mol) for mol in molecules} == before


def test_compact_spacing_preserves_molecule_scale_and_default_rendering(route):
    from synplan.utils.svgslim import content_key

    layouts = {}
    default = route.svg(layouts=layouts)
    compact = route.svg(layouts=layouts, compact=True)
    normal_root, compact_root = ET.fromstring(default), ET.fromstring(compact)
    if not all(
        float(compact_root.get(axis)) < float(normal_root.get(axis))
        for axis in ("width", "height")
    ):
        pytest.fail("Compact rendering did not reduce the canvas")
    normal_mols = normal_root.find(f"{_SVG}g").findall(f"{_SVG}svg")
    compact_mols = compact_root.find(f"{_SVG}g").findall(f"{_SVG}svg")
    for normal, smaller in zip(normal_mols, compact_mols, strict=True):
        for dimension in ("viewBox", "width", "height"):
            if normal.get(dimension) != smaller.get(dimension):
                pytest.fail("Compact rendering resized a molecule")
    if content_key(default) != content_key(route.svg(layouts=layouts)):
        pytest.fail("Compact settings leaked into default rendering")


def test_explicit_spacing_overrides_compact_preset(route):
    from synplan.utils.svgslim import content_key

    layouts = {}
    default = route.svg(layouts=layouts)
    overridden = route.svg(
        layouts=layouts, compact=True, box_pad=9, row_gap=34, col_gap=74
    )
    if content_key(default) != content_key(overridden):
        pytest.fail("Explicit spacing did not override the compact preset")


@pytest.mark.parametrize("prices_enabled", [False, True])
def test_compact_boxes_and_captions_do_not_overlap(route, prices_enabled):
    prices = (
        {id(leaf): ("$10/g", "{}") for leaf in route.leaves()}
        if prices_enabled
        else None
    )
    drawing = ET.fromstring(route.svg(compact=True, prices=prices))
    group = drawing.find(f"{_SVG}g")
    boxes = [
        tuple(float(rect.get(value)) for value in ("x", "y", "width", "height"))
        for rect in group.findall(f"{_SVG}rect")
    ]
    # Caption and optional price space are part of each box's vertical footprint.
    band = 31 if prices_enabled else 0
    for index, (x, y, width, height) in enumerate(boxes):
        for xx, yy, ww, hh in boxes[index + 1 :]:
            if (
                x < xx + ww
                and xx < x + width
                and y - 14 < yy + hh + band
                and yy - 14 < y + height + band
            ):
                pytest.fail("Compact boxes or their caption bands overlap")
