"""Typed contracts shared by route representation, export, and clustering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict


class RouteNode(TypedDict, total=False):
    """Version-1 JSON-compatible route node.

    The existing route JSON format deliberately has no top-level envelope.  This
    type documents that stable shape without changing serialized files.
    """

    type: str
    smiles: str
    children: list[RouteNode]
    in_stock: bool
    bb: dict[str, Any]
    meta: dict[str, Any]
    target_protection_restored: bool
    target_protection_sequence_mode: str
    target_protection_variant_index: int
    target_protection_rule_sequence: list[str]
    target_protection_steps: int
    stereo_mismatch: bool
    step_id: int
    tree_node_id: int | None
    rule_id: int | None
    rule_source: str | None
    rule_key: str | None


@dataclass(frozen=True)
class RouteDiagnostic:
    """A recoverable route-processing failure with enough context to audit it."""

    route_id: Any
    stage: str
    message: str
    exception_type: str | None = None


class RouteExportError(RuntimeError):
    """Raised by strict route export when one or more routes cannot be exported."""

    def __init__(self, diagnostics: tuple[RouteDiagnostic, ...]):
        self.diagnostics = diagnostics
        message = "; ".join(
            f"route {diagnostic.route_id} ({diagnostic.stage}): {diagnostic.message}"
            for diagnostic in diagnostics
        )
        super().__init__(message or "Route export failed")


@dataclass(frozen=True)
class RouteCGRBuildResult:
    """Result of RouteCGR composition independent of the legacy dict API."""

    route_id: Any
    cgr: Any | None = None
    reactions_dict: Mapping[int, Any] | None = None
    diagnostic: RouteDiagnostic | None = None

    @property
    def ok(self) -> bool:
        return self.cgr is not None and self.diagnostic is None

    def as_legacy_dict(
        self, *, include_reactions: bool = False
    ) -> dict[str, Any] | None:
        """Return the historical ``compose_route_cgr`` result shape."""

        if not self.ok:
            return None
        result: dict[str, Any] = {"cgr": self.cgr}
        if include_reactions and self.reactions_dict is not None:
            result["reactions_dict"] = dict(self.reactions_dict)
        return result


@dataclass(frozen=True)
class RouteExportResult:
    """Route-tree output and all recoverable diagnostics collected during export."""

    routes: dict[int, RouteNode] | list[RouteNode]
    diagnostics: tuple[RouteDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.diagnostics


@dataclass(frozen=True)
class SubclusterRouteData:
    """Named internal replacement for the historical seven-item tuple."""

    sb_cgr: Any
    unlabeled_reaction: Any
    synthon_cgr: Any
    synthon_reaction: Any
    leaving_groups: Mapping[int, tuple[Any, int]]
    leaving_group_count: int
    supporting_groups: Mapping[int, tuple[Any, Any]]

    def as_legacy_tuple(self) -> tuple[Any, ...]:
        """Return the tuple shape historically exposed by subclustering."""

        return (
            self.sb_cgr,
            self.unlabeled_reaction,
            self.synthon_cgr,
            self.synthon_reaction,
            self.leaving_groups,
            self.leaving_group_count,
            self.supporting_groups,
        )
