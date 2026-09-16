"""RouteCGR container class and conversion helpers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from chython.containers import CGRContainer

_ROUTE_METADATA = ("route_stereo_steps", "route_step_depths")


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
        metadata = {
            name: self.__dict__[name]
            for name in _ROUTE_METADATA
            if name in self.__dict__
        }
        super().flush_cache()
        self.__dict__.update(metadata)

    def copy(self, **kwargs):
        result = super().copy(**kwargs)
        for name in _ROUTE_METADATA:
            if hasattr(self, name):
                setattr(result, name, deepcopy(getattr(self, name)))
        return result

    def substructure(self, atoms, **kwargs):
        result = super().substructure(atoms, **kwargs)
        for name in _ROUTE_METADATA:
            if hasattr(self, name):
                setattr(result, name, deepcopy(getattr(self, name)))
        return result

    def remap(self, mapping, *, copy=False):
        snapshots = {} if hasattr(self, "route_stereo_steps") else None
        depths = deepcopy(getattr(self, "route_step_depths", None))
        if snapshots is not None or depths is not None:
            from .deconvolution import reactions_from_route_cgr
            from .stereo import graph_digest, snapshot

            for step, reaction in reactions_from_route_cgr(self).items():
                key = str(step + 1)
                source_graph = graph_digest(reaction.compose())
                for molecule in reaction.molecules():
                    molecule.remap(
                        {n: m for n, m in mapping.items() if molecule.has_atom(n)}
                    )
                reaction.flush_cache()
                cgr = reaction.compose()
                if snapshots is not None:
                    snapshots[key] = snapshot(reaction, cgr)
                if depths is not None and key in depths:
                    if depths[key]["graph"] == source_graph:
                        depths[key]["graph"] = graph_digest(cgr)
        result = super().remap(mapping, copy=copy)
        target = result if copy else self
        if snapshots is not None:
            target.route_stereo_steps = snapshots
        if depths is not None:
            target.route_step_depths = depths
        return result

    def depict(self, *args, **kwargs):
        from synplan.chem.reaction.routes.representation.depiction import (
            depict_route_cgr,
        )

        return depict_route_cgr(self, *args, **kwargs)

    def _format_bond(self, n, m, adjacency, **kwargs):
        bond = self.bond(n, m)
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
