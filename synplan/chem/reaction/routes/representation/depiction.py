"""Depiction helpers for RouteCGRContainer."""

from __future__ import annotations

from chython.containers import CGRContainer

from synplan.chem.reaction.routes.visualisation import wide_cgr_renderer


def depict_route_cgr(cgr, *args, **kwargs):
    """Render a RouteCGR with SynPlan's wider transient-bond style."""

    with wide_cgr_renderer(cgr.__class__):
        return CGRContainer.depict(cgr, *args, **kwargs)


__all__ = ["depict_route_cgr"]
