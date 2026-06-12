"""Compatibility wrapper for :mod:`synplan.chem.reaction.routes.quality.protection`.

Import from ``synplan.chem.reaction.routes.quality.protection`` in new code.
"""

from synplan.chem.reaction.routes.quality.protection import (
    CompetingInteraction,
    CompetingSitesScore,
    FunctionalGroupDetector,
    FunctionalGroupMatch,
    HalogenDetector,
    HalogenMatch,
    IncompatibilityMatrix,
    ProtectionConfig,
    RouteScanner,
    classify_reaction_type,
    classify_reaction_type_broad,
    classify_reaction_type_detailed,
    get_reaction_center_atoms,
)

__all__ = [
    "CompetingInteraction",
    "CompetingSitesScore",
    "FunctionalGroupDetector",
    "FunctionalGroupMatch",
    "HalogenDetector",
    "HalogenMatch",
    "IncompatibilityMatrix",
    "ProtectionConfig",
    "RouteScanner",
    "classify_reaction_type",
    "classify_reaction_type_broad",
    "classify_reaction_type_detailed",
    "get_reaction_center_atoms",
]
