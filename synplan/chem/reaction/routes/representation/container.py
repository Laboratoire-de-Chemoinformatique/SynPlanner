"""RouteCGR container class and conversion helpers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

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

    def flush_cache(self):
        stereo = self.__dict__.get("route_stereo_steps")
        super().flush_cache()
        if stereo is not None:
            self.route_stereo_steps = stereo

    def copy(self, **kwargs):
        result = super().copy(**kwargs)
        if hasattr(self, "route_stereo_steps"):
            result.route_stereo_steps = deepcopy(self.route_stereo_steps)
        return result

    def substructure(self, atoms, **kwargs):
        result = super().substructure(atoms, **kwargs)
        if hasattr(self, "route_stereo_steps"):
            result.route_stereo_steps = deepcopy(self.route_stereo_steps)
        return result

    def remap(self, mapping, *, copy=False):
        snapshots = None
        if hasattr(self, "route_stereo_steps"):
            from .deconvolution import reactions_from_route_cgr
            from .stereo import snapshot

            snapshots = {}
            for step, reaction in reactions_from_route_cgr(self).items():
                for molecule in reaction.molecules():
                    molecule.remap(
                        {n: m for n, m in mapping.items() if n in molecule._atoms}
                    )
                reaction.flush_cache()
                snapshots[str(step + 1)] = snapshot(reaction, reaction.compose())
        result = super().remap(mapping, copy=copy)
        if snapshots is not None:
            (result if copy else self).route_stereo_steps = snapshots
        return result

    def depict(self, *args, **kwargs):
        from synplan.chem.reaction.routes.representation.depiction import (
            depict_route_cgr,
        )

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


def unwrap_cgr(value):
    """The CGR inside a composition result, or ``value`` if it is one already."""

    if isinstance(value, Mapping) and "cgr" in value:
        return value["cgr"]
    cgr = getattr(value, "cgr", _MISSING)
    return value if cgr is _MISSING else cgr


_MISSING = object()
