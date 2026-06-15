"""Compatibility wrapper for :mod:`synplan.routes.quality.protection.scanner`."""

from chython.containers import ReactionContainer

from synplan.routes.quality.protection.functional_groups import (
    FunctionalGroupDetector,
    HalogenDetector,
)
from synplan.routes.quality.protection.reaction_classifier import (
    get_reaction_center_atoms,
)
from synplan.routes.quality.protection.scanner import (
    CompetingInteraction,
    CompetingScanResult,
    IncompatibilityMatrix,
    RouteScanner,
)

__all__ = [
    "CompetingInteraction",
    "CompetingScanResult",
    "FunctionalGroupDetector",
    "HalogenDetector",
    "IncompatibilityMatrix",
    "ReactionContainer",
    "RouteScanner",
    "get_reaction_center_atoms",
]
