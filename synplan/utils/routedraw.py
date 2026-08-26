"""Draws one retrosynthetic route as a single SVG.

Boxed chython depictions on a tidy-tree layout, pale role tints, hairline connectors
and a numbered disc per disconnection, carrying the step's position in the route. A
page showing these must also carry :data:`ROUTE_CSS` and one copy of
:data:`ARROW_DEFS`.
"""

from __future__ import annotations

import re
from math import cos, radians, sin
from typing import TYPE_CHECKING, Any

from synplan.utils.align2d import align_molecule
from synplan.utils.routelayout import Node, edges, layout, walk

if TYPE_CHECKING:
    from chython.containers.molecule import MoleculeContainer

__all__ = [
    "ARROW_DEFS",
    "ROLE_STYLE",
    "ROUTE_CSS",
    "draw_route",
    "molecule_svg",
    "orient_route",
]

#: Depiction units to px, and the white margin between a depiction and its frame.
PX_PER_UNIT = 26.0
BOX_PAD = 9.0
ROW_GAP = 34.0  # 18 for air plus 16 for the caption band above a box

#: fill, stroke, stroke width per role.
ROLE_STYLE = {
    "target": ("#f3f6fb", "#94a6cb", 1.6),
    "bb": ("#f3f8f5", "#9dbcae", 1.0),
    "int": ("#ffffff", "#edeff1", 1.0),
    "oos": ("#fdf2f2", "#d06a6a", 1.4),
}
ROLE_CAPTION = {
    "target": ("TARGET", "#41598f"),
    "bb": ("IN STOCK", "#4a7a66"),
    "oos": ("NOT IN STOCK", "#b13b3b"),
}

_FONT = "Inter Tight,system-ui,sans-serif"
ROUTE_CSS = (
    ".sp-link{fill:none;stroke:#c4cbd3;stroke-width:1.5;stroke-linecap:round;"
    "stroke-linejoin:round}"
    ".sp-lead{fill:none;stroke:#c4cbd3;stroke-width:1.5;stroke-linecap:round}"
    f".sp-tag{{font-family:{_FONT};font-size:8.5px;font-weight:700;letter-spacing:.5px}}"
    ".sp-num{text-anchor:middle;dominant-baseline:central;font-weight:700;fill:#fff;"
    f"font-family:{_FONT};font-size:11px}}"
)
ARROW_DEFS = (
    '<marker id="sp-arrow" markerWidth="7" markerHeight="7" refX="5.4" refY="2.6" '
    'orient="auto"><path d="M0.6,0.7 L5.2,2.6 L0.6,4.5" fill="none" stroke="#c4cbd3" '
    'stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"/></marker>'
)

_VIEWBOX = re.compile(r'viewBox="(-?[\d.]+) (-?[\d.]+) ([\d.]+) ([\d.]+)"')


def _is_degenerate(mol: MoleculeContainer) -> bool:
    """chython's own test for "never laid out": every atom on one point."""
    if len(mol) < 2:
        return False
    xs = [atom.x for _, atom in mol.atoms()]
    ys = [atom.y for _, atom in mol.atoms()]
    return max(xs) - min(xs) < 0.01 and max(ys) - min(ys) < 0.01


def _depiction(mol: MoleculeContainer) -> tuple[str, list[float]]:
    if _is_degenerate(mol):
        mol.clean2d()
    svg = mol.depict()
    match = _VIEWBOX.search(svg)
    if match is None:
        raise ValueError(f"chython depiction of {mol} carries no viewBox")
    return svg, [float(v) for v in match.groups()]


def molecule_svg(mol: MoleculeContainer, px: float = 22.0, cap: float = 300.0) -> str:
    """A standalone depiction of ``mol`` at ``px`` per bond, capped at ``cap`` wide."""
    svg, box = _depiction(mol)
    inner = svg[svg.index(">", svg.index("<svg")) + 1 : svg.rindex("</svg>")]
    scale = min(px, cap / box[2]) if box[2] else px
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{box[2] * scale:.0f}" '
        f'height="{box[3] * scale:.0f}" viewBox="{box[0]} {box[1]} {box[2]} {box[3]}">'
        f"{inner}</svg>"
    )


def orient_route(steps: Any, step_deg: int = 5) -> int:
    """Rotate every molecule of the route by one shared angle, in place.

    Alignment fixes each precursor relative to its product; the absolute angle of the
    whole route is still free, so spend it on the shape this layout stacks best —
    smallest summed height, then smallest column width.
    """
    mols, seen = [], set()
    for reaction in steps:
        for mol in (*reaction.reactants, *reaction.products):
            if id(mol) not in seen:
                seen.add(id(mol))
                mols.append(mol)

    coords = [[(atom.x, atom.y) for _, atom in mol.atoms()] for mol in mols]
    best = None
    for deg in range(0, 180, step_deg):
        angle = radians(deg)
        c, s = cos(angle), sin(angle)
        height = width = 0.0
        for points in coords:
            xs = [x * c - y * s for x, y in points]
            ys = [x * s + y * c for x, y in points]
            height += max(ys) - min(ys)
            width = max(width, max(xs) - min(xs))
        if best is None or (height, width) < best[0]:
            best = ((height, width), deg)

    deg = best[1]
    if deg:
        angle = radians(deg)
        c, s = cos(angle), sin(angle)
        for mol, points in zip(mols, coords):
            for (_, atom), (x, y) in zip(mol.atoms(), points):
                atom.xy = (x * c - y * s, x * s + y * c)
            # wedges are derived from the coordinates and cached
            mol.__dict__.pop("_wedge_map", None)
            mol.__dict__.pop("__cached_method__repr_svg_", None)
    return deg


def _route_tree(steps: Any, unresolved: Any, align: bool) -> tuple[Node, dict]:
    if align:
        # leaf-first order, so each disconnection inherits the layout above it
        for step in reversed(steps):
            for precursor in step.reaction.reactants:
                align_molecule(precursor, step.product)
        orient_route([step.reaction for step in steps])

    by_product = {
        id(step.product): (number, step) for number, step in enumerate(steps, 1)
    }
    red = {id(mol) for mol in unresolved}
    depicts = {}

    def build(mol: MoleculeContainer) -> Node:
        svg, box = _depiction(mol)
        depicts[id(mol)] = (svg, box)
        node = Node(
            key=str(mol),
            w=box[2] * PX_PER_UNIT + 2 * BOX_PAD,
            h=box[3] * PX_PER_UNIT + 2 * BOX_PAD,
        )
        node.mol = mol
        made = by_product.get(id(mol))
        if made is None:
            node.role = "oos" if id(mol) in red else "bb"
        else:
            node.number, step = made
            node.role = "int"
            node.children = [build(m) for m in step.reaction.reactants]
        return node

    root = build(steps[-1].product)
    root.role = "target"
    return root, depicts


def _to_svg(root: Node, depicts: dict, links: list[dict], w: float, h: float) -> str:
    top = 16.0  # room for the role caption drawn above a box
    margin = 14.0  # frame strokes are centred on the box edge; keep them on the canvas
    width, height = w + 2 * margin, h + top + 2 * margin
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        f'<g transform="translate({margin:.0f},{top + margin:.0f})">',
    ]

    for edge in links:
        lane, (px, py) = edge["lane"], edge["parent"]
        for cx, cy in edge["children"]:
            dy = py - cy
            if abs(dy) < 0.5:
                path = f"M{cx:.1f} {cy:.1f} H{lane:.1f}"
            else:
                sign = 1 if dy > 0 else -1
                r = min(16.0, abs(dy) / 2, abs(lane - cx) / 2)
                path = (
                    f"M{cx:.1f} {cy:.1f} H{lane - r:.1f} Q{lane:.1f} {cy:.1f} "
                    f"{lane:.1f} {cy + sign * r:.1f} V{py:.1f}"
                )
            parts.append(f'<path d="{path}" class="sp-link"/>')
        parts.append(
            f'<path d="M{lane:.1f} {py:.1f} H{px:.1f}" class="sp-lead" '
            f'marker-end="url(#sp-arrow)"/>'
        )

    for node in [root, *walk(root)]:
        fill, stroke, stroke_width = ROLE_STYLE[node.role]
        parts.append(
            f'<rect x="{node.x:.1f}" y="{node.y:.1f}" width="{node.w:.1f}" '
            f'height="{node.h:.1f}" rx="5" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"/>'
        )
        svg, box = depicts[id(node.mol)]
        inner = svg[svg.index(">", svg.index("<svg")) + 1 : svg.rindex("</svg>")]
        parts.append(
            f'<svg x="{node.x + BOX_PAD:.1f}" y="{node.y + BOX_PAD:.1f}" '
            f'width="{max(1.0, node.w - 2 * BOX_PAD):.1f}" '
            f'height="{max(1.0, node.h - 2 * BOX_PAD):.1f}" '
            f'viewBox="{box[0]} {box[1]} {box[2]} {box[3]}">{inner}</svg>'
        )
        if node.role in ROLE_CAPTION:
            caption, colour = ROLE_CAPTION[node.role]
            parts.append(
                f'<text x="{node.x + 1:.1f}" y="{node.y - 5:.1f}" class="sp-tag" '
                f'fill="{colour}">{caption}</text>'
            )

    for edge in links:
        lane, (_, py) = edge["lane"], edge["parent"]
        parts.append(
            f'<circle cx="{lane:.1f}" cy="{py:.1f}" r="10.5" fill="#2b3440" '
            f'stroke="#ffffff" stroke-width="2"/>'
            f'<text x="{lane:.1f}" y="{py:.1f}" class="sp-num">{edge["number"]}</text>'
        )

    parts.append("</g></svg>")
    return "".join(parts)


def draw_route(steps: Any, unresolved: Any = (), align: bool = True) -> str:
    """Draw one route.

    Whether a leaf is purchasable is decided by the caller, never here. Which step
    feeds which is read from object identity, so two steps sharing a product SMILES
    still get a disc each.

    :param steps: The route's steps, deepest first; each carries the ``reaction``
        it runs and the ``product`` it disconnects. The disc over the
        disconnection of ``steps[i]`` carries ``i + 1``.
    :param unresolved: The terminal precursors that are not purchasable; they are
        drawn in the ``oos`` role, every other leaf in ``bb``.
    :param align: If True, give every precursor its product's orientation.
    :return: The SVG, without ``ROUTE_CSS`` or ``ARROW_DEFS``.
    """
    root, depicts = _route_tree(steps, unresolved, align)
    width, height, col_x, col_w = layout(root, row_gap=ROW_GAP)
    links = edges(root, col_x, col_w)
    for edge in links:
        edge["number"] = edge["node"].number
    return _to_svg(root, depicts, links, width, height)
