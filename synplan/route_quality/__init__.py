"""Route quality assessment module for synthetic route analysis.

Provides functional group detection, reaction classification, and
incompatibility scoring to identify synthesis steps that may require
protecting group strategies.

This module is inspired by the work of Westerlund et al.:

    Westerlund, A. M.; Sigmund, L. M.; Kannas, C.; Genheden, S.; Kabeshov, M.
    "Toward lab-ready AI synthesis plans with protection strategies and route scoring."
    *ChemRxiv*, 2025. https://doi.org/10.26434/chemrxiv-2025-gdrr8

The competing-sites score S(T) and the functional-group incompatibility
framework follow the methodology described in that paper.
"""

from synplan.route_quality.protection.config import ProtectionConfig
from synplan.route_quality.protection.functional_groups import (
    FunctionalGroupDetector,
    FunctionalGroupMatch,
    HalogenDetector,
    HalogenMatch,
)
from synplan.route_quality.protection.reaction_classifier import (
    classify_reaction_type,
    classify_reaction_type_broad,
    classify_reaction_type_detailed,
    get_reaction_center_atoms,
)
from synplan.route_quality.protection.scanner import (
    CompetingInteraction,
    IncompatibilityMatrix,
    RouteScanner,
)
from synplan.route_quality.protection.scorer import CompetingSitesScore

__all__ = [
    "ProtectionConfig",
    "FunctionalGroupDetector",
    "FunctionalGroupMatch",
    "HalogenDetector",
    "HalogenMatch",
    "classify_reaction_type",
    "classify_reaction_type_broad",
    "classify_reaction_type_detailed",
    "get_reaction_center_atoms",
    "CompetingInteraction",
    "IncompatibilityMatrix",
    "RouteScanner",
    "CompetingSitesScore",
]
