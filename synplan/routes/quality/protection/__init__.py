"""Protection-strategy route quality scoring."""

from synplan.routes.quality.protection.config import (
    ProtectionConfig,
    ProtectionRevisionConfig,
)
from synplan.routes.quality.protection.functional_groups import (
    FunctionalGroupDetector,
    FunctionalGroupMatch,
    HalogenDetector,
    HalogenMatch,
)
from synplan.routes.quality.protection.reaction_classifier import (
    classify_reaction_type,
    classify_reaction_type_broad,
    classify_reaction_type_detailed,
    get_reaction_center_atoms,
)
from synplan.routes.quality.protection.scanner import (
    CompetingInteraction,
    CompetingScanResult,
    IncompatibilityMatrix,
    RouteScanner,
)
from synplan.routes.quality.protection.scorer import CompetingSitesScore
from synplan.routes.quality.protection.revision import (
    ProtectionAction,
    ProtectionFragmentCatalog,
    ProtectionRouteReviser,
    ProtectionRevisionDiagnostic,
    RevisedRoute,
)

__all__ = [
    "CompetingScanResult",
    "CompetingInteraction",
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
