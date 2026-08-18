"""Post-processing transformations for JSON-compatible synthesis routes."""

from .cost import (
    BuildingBlockCost,
    RouteCostError,
    RouteCostEstimate,
    estimate_route_cost,
    estimate_route_costs,
)
from .deprotected_building_blocks import (
    RouteExpansionLimitError,
    expand_deprotected_building_blocks,
)
from .pipeline import (
    PostprocessedRoute,
    RoutePostprocessConfig,
    RoutePostprocessDiagnostic,
    RoutePostprocessResult,
    postprocess_routes,
)
from .stereo import RouteStereoError, restore_route_stereo
from .target_protection import (
    TargetProtectionError,
    restore_protected_target,
)

__all__ = [
    "BuildingBlockCost",
    "PostprocessedRoute",
    "RouteCostError",
    "RouteCostEstimate",
    "RouteExpansionLimitError",
    "RoutePostprocessConfig",
    "RoutePostprocessDiagnostic",
    "RoutePostprocessResult",
    "RouteStereoError",
    "TargetProtectionError",
    "estimate_route_cost",
    "estimate_route_costs",
    "expand_deprotected_building_blocks",
    "postprocess_routes",
    "restore_protected_target",
    "restore_route_stereo",
]
