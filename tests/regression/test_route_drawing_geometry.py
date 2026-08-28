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


def test_drawing_leaves_the_route_molecules_where_the_reactor_put_them(route):
    """The tree holds these very objects; the drawer may not relayout them."""

    molecules = [
        mol
        for step in route
        for mol in (*step.reaction.reactants, *step.reaction.products)
    ]
    before = {id(mol): coordinates(mol) for mol in molecules}
    route.svg()
    assert {id(mol): coordinates(mol) for mol in molecules} == before
