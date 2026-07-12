"""CGR and reaction SVG depiction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

from chython.algorithms.depict import DepictCGR, _render_config
from chython.containers import CGRContainer, ReactionContainer

TRANSIENT_BOND_COLOR = "blue"
DYNAMIC_BOND_WIDTH_FACTOR = 2.5

_MISSING = object()


def _unwrap_cgr(value: Any) -> CGRContainer:
    """Accept raw CGRs and historical composition result wrappers."""

    if isinstance(value, Mapping) and "cgr" in value:
        value = value["cgr"]
    else:
        cgr = getattr(value, "cgr", _MISSING)
        if cgr is not _MISSING:
            value = cgr

    if value is None:
        raise ValueError("Cannot depict an empty RouteCGR result.")
    return value


__all__ = [
    "WideBondDepictCGR",
    "cgr_display",
    "depict_custom_reaction",
    "depict_route_cgr",
    "wide_cgr_renderer",
]


@contextmanager
def _temporary_class_attrs(cls, updates):
    saved = {name: cls.__dict__.get(name) for name in updates}
    for name, value in updates.items():
        setattr(cls, name, value)

    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, value)


@contextmanager
def _temporary_render_config(**updates):
    saved = {name: _render_config.get(name) for name in updates}
    _render_config.update(updates)

    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                _render_config.pop(name, None)
            else:
                _render_config[name] = value


@contextmanager
def _hidden_bonds(cgr: CGRContainer, pairs: list[tuple[int, int]]):
    removed = []
    for n, m in pairs:
        removed.append((n, m, cgr._bonds[n].pop(m), cgr._bonds[m].pop(n)))

    try:
        yield
    finally:
        for n, m, bond_nm, bond_mn in removed:
            cgr._bonds[n][m] = bond_nm
            cgr._bonds[m][n] = bond_mn


def _dynamic_bond_width() -> float:
    return _render_config.get("bond_width", 0.04) * DYNAMIC_BOND_WIDTH_FACTOR


def _transient_bond_pairs(cgr: CGRContainer) -> list[tuple[int, int]]:
    return [
        (n, m)
        for n, m, bond in cgr.bonds()
        if bond.order is None and bond.p_order is None
    ]


def _render_transient_bond(cgr: CGRContainer, n: int, m: int) -> str:
    nx, ny = cgr._plane[n]
    mx, my = cgr._plane[m]
    return (
        f'      <line x1="{nx:.2f}" y1="{-ny:.2f}" '
        f'x2="{mx:.2f}" y2="{-my:.2f}" stroke="{TRANSIENT_BOND_COLOR}" '
        f'stroke-width="{_render_config["dynamic_bond_width"]:.2f}"/>'
    )


def _wide_aromatic_bond(self, n_x, n_y, m_x, m_y, c_x, c_y, color):
    svg = DepictCGR._DepictCGR__render_aromatic_bond(
        self, n_x, n_y, m_x, m_y, c_x, c_y, color
    )
    if svg and color:
        return (
            svg[:-2] + f' stroke-width="{_render_config["dynamic_bond_width"]:.2f}"/>'
        )
    return svg


class WideBondDepictCGR(DepictCGR):
    """Use Chython's CGR renderer with wider dynamic and blue transient bonds."""

    __slots__ = ()

    _DepictCGR__render_aromatic_bond = _wide_aromatic_bond

    def _render_bonds(self):
        transient_pairs = _transient_bond_pairs(self)
        with _hidden_bonds(self, transient_pairs):
            svg = DepictCGR._render_bonds(self)

        svg.extend(_render_transient_bond(self, n, m) for n, m in transient_pairs)
        return svg


@contextmanager
def wide_cgr_renderer(container_cls=CGRContainer):
    """Temporarily render CGR dynamic bonds with SynPlanner route styling."""

    updates = {
        "_render_bonds": WideBondDepictCGR._render_bonds,
        "_DepictCGR__render_aromatic_bond": (
            WideBondDepictCGR._DepictCGR__render_aromatic_bond
        ),
    }
    with (
        _temporary_render_config(dynamic_bond_width=_dynamic_bond_width()),
        _temporary_class_attrs(container_cls, updates),
    ):
        yield


def depict_route_cgr(
    cgr: CGRContainer | Mapping[str, Any], *args, **kwargs
) -> str:
    """Render a RouteCGR with SynPlanner's wider transient-bond style.

    Raw CGRs and historical {"cgr": ...}/typed result wrappers are accepted
    for compatibility with older route-export callers.
    """

    cgr = _unwrap_cgr(cgr)
    with wide_cgr_renderer(cgr.__class__):
        return CGRContainer.depict(cgr, *args, **kwargs)


def cgr_display(cgr: CGRContainer | Mapping[str, Any]) -> str:
    """Return an SVG string for a CGR with wide dynamic route bonds.

    Historical {"cgr": ...} and typed build-result wrappers are also accepted.
    """

    cgr = _unwrap_cgr(cgr)
    with wide_cgr_renderer(CGRContainer):
        cgr.clean2d()
        return CGRContainer.depict(cgr, clean2d=False)


def depict_custom_reaction(reaction: ReactionContainer):
    """Return Chython's SVG depiction for a reaction.

    This helper preserves SynPlanner's public import path while delegating the
    rendering work to Chython.
    """

    reaction.clean2d()
    return reaction.depict()
