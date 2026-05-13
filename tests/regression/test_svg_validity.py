"""SVG-output invariants for ``cgr_display`` / CGRContainer depiction.

Two bug classes are guarded:

1. **Malformed SVG.** A typo in an f-string in
   ``visualisation.py`` (missing closing quote after a colour interpolation
   in the ``order==2, p_order==3`` bond branch) produces unparseable XML
   that breaks the entire ``<svg>`` document. Invariant: any depiction we
   produce must parse via ``xml.etree.ElementTree``.

2. **Class-level state leak.** ``cgr_display`` patches ``CGRContainer``
   bond-rendering methods at the *class* level with no ``try/finally``
   restore. After the first call, every subsequent ``CGRContainer.depict()``
   in the same process uses the wide-bond style — including unrelated
   callers. Invariant: a plain ``.depict()`` call before any ``cgr_display``
   call must produce the same XML as a ``.depict()`` call after
   ``cgr_display`` has been invoked.

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
    """``cgr_display`` must also return parseable XML — its wide-bond branch
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
    """``cgr_display`` must not permanently alter ``CGRContainer.depict``.

    Sequence:
        1. depict A normally  →  svg_a_before
        2. cgr_display(B)     (this monkey-patches CGRContainer methods)
        3. depict A normally  →  svg_a_after
    Invariant: svg_a_before == svg_a_after (modulo non-determinism the
    caller controls — coordinate seeds are reset before each depict).

    If they differ, ``cgr_display``'s class-level patches are still in
    effect during step 3, meaning every unrelated CGR depiction in the
    process is using the wide-bond renderer.
    """
    from synplan.chem.reaction_routes.visualisation import cgr_display

    cgr_a, cgr_b = two_cgrs
    cgr_a.clean2d()
    svg_a_before = cgr_a.depict()

    _ = cgr_display(cgr_b)

    cgr_a.clean2d()
    svg_a_after = cgr_a.depict()

    assert svg_a_before == svg_a_after, (
        "CGRContainer.depict() output changed for the same CGR after "
        "cgr_display() was called on a different CGR. cgr_display patches "
        "CGRContainer._render_bonds at the class level without a "
        "try/finally restore (visualisation.py:511-517), so all subsequent "
        "depictions in the process inherit the wide-bond renderer."
    )
