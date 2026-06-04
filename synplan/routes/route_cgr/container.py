"""RouteCGR container class and conversion helpers."""

from __future__ import annotations

from chython.containers import CGRContainer


class RouteCGRContainer(CGRContainer):
    """CGRContainer subclass used for composed synthetic route CGRs.

    RouteCGRs may contain the route-only transient bond marker
    ``DynamicBond(None, None)``. Chython's default CGR formatter does not know
    how to serialize that state, so this subclass provides a stable textual
    representation (`[.>.]`). SVG renderer wiring is delegated lazily to
    ``route_cgr_depiction.py`` when ``depict()`` is called.
    """

    __slots__ = ()

    def depict(self, *args, **kwargs):
        from synplan.routes.route_cgr.depiction import depict_route_cgr

        return depict_route_cgr(self, *args, **kwargs)

    def _format_bond(self, n, m, adjacency, **kwargs):
        bond = self._bonds[n][m]
        if bond.order is None and bond.p_order is None:
            return "[.>.]"
        return super()._format_bond(n, m, adjacency, **kwargs)

    def __getstate__(self):
        return {
            slot: getattr(self, slot)
            for slot in CGRContainer.__slots__
            if hasattr(self, slot)
        }


def enable_route_cgr_container(cgr: CGRContainer) -> RouteCGRContainer:
    """Convert a CGRContainer instance in-place to RouteCGRContainer."""

    cgr.__class__ = RouteCGRContainer
    return cgr
