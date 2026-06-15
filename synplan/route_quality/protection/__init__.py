"""Compatibility wrapper for :mod:`synplan.routes.quality.protection`.

Import from ``synplan.routes.quality.protection`` in new code.
"""

from synplan.routes.quality.protection import (
    CompetingInteraction,
    CompetingScanResult,
    CompetingSitesScore,
    FunctionalGroupDetector,
    FunctionalGroupMatch,
    HalogenDetector,
    HalogenMatch,
    IncompatibilityMatrix,
    ProtectionAction,
    ProtectionConfig,
    ProtectionFragmentCatalog,
    ProtectionRevisionConfig,
    ProtectionRevisionDiagnostic,
    ProtectionRouteReviser,
    RevisedRoute,
    RouteScanner,
    classify_reaction_type,
    classify_reaction_type_broad,
    classify_reaction_type_detailed,
    get_reaction_center_atoms,
)

__all__ = [
    "CompetingInteraction",
    "CompetingScanResult",
    "CompetingSitesScore",
    "FunctionalGroupDetector",
    "FunctionalGroupMatch",
    "HalogenDetector",
    "HalogenMatch",
    "IncompatibilityMatrix",
    "ProtectionAction",
    "ProtectionConfig",
    "ProtectionFragmentCatalog",
    "ProtectionRevisionConfig",
    "ProtectionRevisionDiagnostic",
    "ProtectionRouteReviser",
    "RevisedRoute",
    "RouteScanner",
    "classify_reaction_type",
    "classify_reaction_type_broad",
    "classify_reaction_type_detailed",
    "get_reaction_center_atoms",
]
