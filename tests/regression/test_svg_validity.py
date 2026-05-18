"""SVG-output invariants for ``cgr_display`` / CGRContainer depiction.

Two bug classes are guarded:

Malformed SVG: a typo in an f-string in ``visualisation.py`` (missing
closing quote after a colour interpolation in the ``order==2, p_order==3``
bond branch) produces unparseable XML that breaks the entire ``<svg>``
document. Invariant: any depiction we produce must parse via
``xml.etree.ElementTree``.

Class-level state leak: ``cgr_display`` patches ``CGRContainer``
bond-rendering methods at the class level with no ``try/finally`` restore.
After the first call, every subsequent ``CGRContainer.depict()`` in the
same process uses the wide-bond style, including unrelated callers.
Invariant: a plain ``.depict()`` call before any ``cgr_display`` call must
produce the same XML as a ``.depict()`` call after ``cgr_display`` has been
invoked.

The first test depicts a small CGR derived from a conftest reaction; the
second test calls ``cgr_display`` once on an unrelated CGR and compares
``cgr.depict()`` output before vs. after.

These tests do not inspect specific bond colours, paths, or coordinates —
only that the SVG is parseable and that depiction is deterministic given the
same CGR. They survive any depiction style refresh.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest


@pytest.fixture
def two_cgrs(simple_cgr, diels_alder_cgr):
    """Two unrelated CGRs so we can test a state leak across calls."""
    return simple_cgr, diels_alder_cgr


def test_depict_produces_parseable_svg(simple_cgr):
    """``CGRContainer.depict()`` must return parseable XML.

    Catches the broken f-string in ``visualisation.py`` (the missing closing
    quote after ``{formed}`` in the wide-bond DynamicBond branch) and any
    future SVG-formatting regression that emits malformed XML.
    """
    simple_cgr.clean2d()
    svg = simple_cgr.depict()
    assert isinstance(svg, str) and svg.strip(), "depict() returned empty"
    # ElementTree.fromstring raises ParseError on malformed XML.
    try:
        ET.fromstring(svg)
    except ET.ParseError as e:
        pytest.fail(
            f"CGRContainer.depict() produced unparseable XML: {e}. Likely "
            "cause: an f-string in visualisation.py emits malformed "
            "attribute syntax (e.g. missing closing quote)."
        )


def test_cgr_display_produces_parseable_svg(simple_cgr):
    """``cgr_display`` must also return parseable XML; its wide-bond branch
    is where the f-string bug lives."""
    from synplan.chem.reaction_routes.visualisation import cgr_display

    svg = cgr_display(simple_cgr)
    assert isinstance(svg, str) and svg.strip()
    try:
        ET.fromstring(svg)
    except ET.ParseError as e:
        pytest.fail(
            f"cgr_display produced unparseable XML: {e}. Likely the broken "
            "f-string at visualisation.py:203 in the order==2/p_order==3 "
            "DynamicBond branch fired for this CGR."
        )


def test_cgr_display_does_not_leak_state_into_depict(two_cgrs):
    """``cgr_display`` must not permanently alter ``CGRContainer`` methods.

    The bug is structural: ``cgr_display`` assigns to ``CGRContainer``'s
    class-level ``_render_bonds`` and ``__render_aromatic_bond`` attributes.
    Without a try/finally restore, every subsequent ``.depict()`` call
    anywhere in the process inherits the wide-bond renderer.

    The invariant checked here is class-attribute identity *before* and
    *after* the call: direct and deterministic, unaffected by chython's
    coordinate generation.
    """
    from chython.containers import CGRContainer

    from synplan.chem.reaction_routes.visualisation import cgr_display

    _, cgr_b = two_cgrs

    attrs_to_check = (
        "_render_bonds",
        "_CGRContainer__render_aromatic_bond",
        "_WideBondDepictCGR__render_aromatic_bond",
    )
    before = {a: CGRContainer.__dict__.get(a) for a in attrs_to_check}
    _ = cgr_display(cgr_b)
    after = {a: CGRContainer.__dict__.get(a) for a in attrs_to_check}

    leaks = [a for a in attrs_to_check if before[a] is not after[a]]
    assert not leaks, (
        f"cgr_display left CGRContainer class-level attributes patched: "
        f"{leaks}. Every subsequent CGRContainer.depict() in the process "
        "will inherit the wide-bond renderer. Wrap the patches in "
        "try/finally and restore the originals."
    )
