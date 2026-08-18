"""Ordered and bounded building-block route postprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from chython import smiles as smiles_parser
from chython.containers import MoleculeContainer

from synplan.chem.building_blocks.catalog import BuildingBlockCatalog
from synplan.chem.building_blocks.config import DeprotectionPolicy
from synplan.chem.building_blocks.deprotection import (
    DeprotectionSequenceMode,
    deprotect_molecule,
)
from synplan.chem.reaction.routes.contracts import RouteDiagnostic
from synplan.chem.reaction.routes.tree_ops import iter_molecule_leaves

from .cost import RouteCostEstimate, estimate_route_cost, estimate_route_costs
from .deprotected_building_blocks import expand_deprotected_building_blocks
from .stereo import (
    RouteStereoError,
    restore_route_stereo,
    route_root_matches_target_stereo,
)
from .target_protection import restore_protected_target, route_root_matches_target

# Compatibility alias; postprocessing now uses the shared route diagnostic.
RoutePostprocessDiagnostic = RouteDiagnostic


@dataclass(frozen=True, slots=True)
class RoutePostprocessConfig:
    """Controls restoration and costing order for a route pool."""

    expand_deprotected: bool = True
    calculate_cost: bool = True
    target_deprotection_policy: DeprotectionPolicy = "conservative"
    max_variants_per_route: int = 100
    target_protection_sequence_mode: DeprotectionSequenceMode = "enumerate"

    def __post_init__(self) -> None:
        if self.max_variants_per_route < 1:
            raise ValueError("max_variants_per_route must be at least 1")
        if self.target_deprotection_policy not in {"conservative", "aggressive"}:
            raise ValueError(
                "target_deprotection_policy must be conservative or aggressive"
            )
        if self.target_protection_sequence_mode not in {
            "enumerate",
            "deterministic",
        }:
            raise ValueError(
                "target_protection_sequence_mode must be enumerate or deterministic"
            )


@dataclass(frozen=True, slots=True)
class PostprocessedRoute:
    """One stable source-route variant and its optional cost estimate."""

    route_id: Any
    variant_index: int
    route: Mapping[str, Any]
    cost: RouteCostEstimate | None = None


@dataclass(frozen=True, slots=True)
class RoutePostprocessResult:
    """Completed variants plus recoverable per-route diagnostics."""

    variants: tuple[PostprocessedRoute, ...]
    diagnostics: tuple[RouteDiagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.diagnostics


def _has_unresolved_deprotected_leaf(node: Mapping[str, Any]) -> bool:
    return any(
        leaf.get("in_stock") is True
        and any(
            record.get("output_origin") == "deprotected"
            for record in (leaf.get("bb") or {}).get("records", [])
        )
        for _path, leaf in iter_molecule_leaves(node)
    )


def _diagnostic(route_id: Any, stage: str, error: Exception) -> RouteDiagnostic:
    return RouteDiagnostic(
        route_id=route_id,
        stage=stage,
        message=str(error),
        exception_type=type(error).__name__,
    )


def postprocess_routes(
    routes: Mapping[Any, Mapping[str, Any]],
    catalog: BuildingBlockCatalog,
    *,
    target_smiles: str | Mapping[Any, str] | None = None,
    preprocessing_provenance: Mapping[str, object] | None = None,
    protected_targets: Mapping[Any, str] | None = None,
    target_provenance: Mapping[Any, Mapping[str, object]] | None = None,
    config: RoutePostprocessConfig | None = None,
) -> RoutePostprocessResult:
    """Infer target restoration needs, restore BBs, then calculate route cost.

    Route failures are isolated and reported in ``diagnostics``. Costing is
    never run on a provenance-marked deprotected leaf when restoration is
    disabled.

    Supply the original standardized target through ``target_smiles``. For
    each route, graph comparison determines whether protection restoration is
    needed, while molecular stereo descriptors determine whether stereo
    restoration is needed. A single string applies to every route; a mapping
    supports route-specific targets. Target protection always precedes BB
    expansion, stereo restoration, and costing. The variant bound applies to
    the Cartesian product of target sequences and protected-BB alternatives.

    ``protected_targets`` and ``target_provenance`` are retained as
    compatibility inputs for callers that select those stages explicitly.
    """
    if not isinstance(routes, Mapping):
        raise TypeError("routes must be a mapping of identifiers to route trees")
    if target_smiles is not None and not isinstance(target_smiles, (str, Mapping)):
        raise TypeError("target_smiles must be a SMILES string or mapping by route id")
    if isinstance(target_smiles, str) and not target_smiles.strip():
        raise ValueError("target_smiles must be non-empty")
    if target_smiles is not None and protected_targets is not None:
        raise ValueError("target_smiles and protected_targets are mutually exclusive")
    if preprocessing_provenance is not None and not isinstance(
        preprocessing_provenance, Mapping
    ):
        raise TypeError("preprocessing_provenance must be a mapping")
    if preprocessing_provenance is not None and target_provenance is not None:
        raise ValueError(
            "preprocessing_provenance and target_provenance are mutually exclusive"
        )
    if protected_targets is not None and not isinstance(protected_targets, Mapping):
        raise TypeError("protected_targets must be a mapping keyed by route id")
    target_source = target_smiles if target_smiles is not None else protected_targets
    automatic_target_inference = target_smiles is not None
    if preprocessing_provenance is not None and target_source is None:
        raise ValueError("preprocessing_provenance requires target_smiles")
    options = config or RoutePostprocessConfig()
    expanded: list[tuple[Any, int, Mapping[str, Any]]] = []
    diagnostics: list[RouteDiagnostic] = []
    if target_provenance is not None and not isinstance(target_provenance, Mapping):
        raise TypeError("target_provenance must be a mapping keyed by route id")

    for route_id, route in routes.items():
        if (
            options.calculate_cost
            and not options.expand_deprotected
            and _has_unresolved_deprotected_leaf(route)
        ):
            diagnostics.append(
                RouteDiagnostic(
                    route_id=route_id,
                    stage="ordering",
                    message=(
                        "cost calculation requires deprotected BB restoration first"
                    ),
                )
            )
            continue
        if isinstance(target_source, str):
            target = target_source
        elif target_source is not None:
            target = target_source.get(route_id)
        else:
            target = None
        target_record = preprocessing_provenance if target is not None else None
        if target_record is None and target_provenance is not None:
            target_record = target_provenance.get(route_id)
        try:
            if target is not None and (
                not isinstance(target, str) or not target.strip()
            ):
                raise ValueError(
                    f"target for route {route_id!r} must be a non-empty SMILES string"
                )
            if target_record is not None and target is None:
                raise ValueError("target provenance requires a target for the route")
            protection_needed = bool(
                target is not None
                and (
                    not automatic_target_inference
                    or not route_root_matches_target(route, target)
                )
            )
            stereo_source = target
            if automatic_target_inference and protection_needed:
                parsed_target = smiles_parser(target)
                if not isinstance(parsed_target, MoleculeContainer):
                    raise ValueError("target must describe one molecule")
                stereo_source = str(
                    deprotect_molecule(
                        parsed_target,
                        policy=options.target_deprotection_policy,
                    )
                )
            stereo_needed = bool(
                target is not None
                and (
                    not automatic_target_inference
                    or not route_root_matches_target_stereo(route, stereo_source)
                )
            )
            if (
                automatic_target_inference
                and target_record is not None
                and not protection_needed
            ):
                raise ValueError(
                    "preprocessing_provenance was supplied, but the target and "
                    "route root have the same connectivity"
                )
            if protection_needed and stereo_needed:
                target_scenario = "stereo_and_protection"
            elif protection_needed:
                target_scenario = "protection"
            elif stereo_needed:
                target_scenario = "stereo"
            else:
                target_scenario = "none"
            restoration_options: dict[str, object] = {
                "sequence_mode": options.target_protection_sequence_mode,
                "max_variants": options.max_variants_per_route,
            }
            if options.target_deprotection_policy != "conservative":
                restoration_options["policy"] = options.target_deprotection_policy
            if target_record is not None:
                restoration_options["preprocessing_provenance"] = target_record
            target_variants = (
                restore_protected_target(
                    route,
                    target,
                    **restoration_options,
                )
                if protection_needed
                else [dict(route)]
            )
        except Exception as error:
            diagnostics.append(_diagnostic(route_id, "target_protection", error))
            continue

        try:
            if options.expand_deprotected:
                first_variants = expand_deprotected_building_blocks(
                    target_variants[0],
                    catalog,
                    max_variants=options.max_variants_per_route,
                )
                required_variants = len(target_variants) * len(first_variants)
                if required_variants > options.max_variants_per_route:
                    raise ValueError(
                        "target-protection and deprotected-BB expansion requires "
                        f"{required_variants} variants; limit is "
                        f"{options.max_variants_per_route}"
                    )
                variants = list(first_variants)
                for target_variant in target_variants[1:]:
                    current_variants = expand_deprotected_building_blocks(
                        target_variant,
                        catalog,
                        max_variants=len(first_variants),
                    )
                    if len(current_variants) != len(first_variants):
                        raise ValueError(
                            "deprotected-BB expansion count changed between "
                            "target-protection sequences"
                        )
                    variants.extend(current_variants)
            else:
                variants = target_variants
        except Exception as error:
            diagnostics.append(_diagnostic(route_id, "bb_restoration", error))
            continue

        if stereo_needed:
            stereo_variants = []
            for target_variant_index, variant in enumerate(variants):
                try:
                    stereo_variants.append(
                        restore_route_stereo(
                            variant,
                            target,
                            catalog=catalog,
                        )
                    )
                except Exception as error:
                    contextual_error = RouteStereoError(
                        f"variant {target_variant_index}: {error}"
                    )
                    diagnostics.append(
                        _diagnostic(
                            route_id,
                            "stereo_restoration",
                            contextual_error,
                        )
                    )
            variants = stereo_variants

        if automatic_target_inference and target is not None:
            for variant in variants:
                variant["target_postprocessing_scenario"] = target_scenario

        expanded.extend(
            (route_id, variant_index, variant)
            for variant_index, variant in enumerate(variants)
        )

    costs: dict[tuple[Any, int], RouteCostEstimate] = {}
    if options.calculate_cost and expanded:
        keyed_routes = {
            (route_id, variant_index): route
            for route_id, variant_index, route in expanded
        }
        try:
            costs = estimate_route_costs(keyed_routes, catalog)
        except Exception:
            # Preserve route-local failure isolation if a malformed variant makes
            # the optimized shared catalogue calculation fail.
            for key, route in keyed_routes.items():
                try:
                    costs[key] = estimate_route_cost(route, catalog)
                except Exception as error:
                    diagnostics.append(_diagnostic(key[0], "cost", error))

    return RoutePostprocessResult(
        variants=tuple(
            PostprocessedRoute(
                route_id=route_id,
                variant_index=variant_index,
                route=route,
                cost=costs.get((route_id, variant_index)),
            )
            for route_id, variant_index, route in expanded
        ),
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "PostprocessedRoute",
    "RoutePostprocessConfig",
    "RoutePostprocessDiagnostic",
    "RoutePostprocessResult",
    "postprocess_routes",
]
