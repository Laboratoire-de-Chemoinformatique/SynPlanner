"""Compatibility wrapper for :mod:`synplan.chem.reaction.routes.quality`.

Import from ``synplan.chem.reaction.routes.quality`` in new code.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    "CompetingInteraction": (
        "synplan.chem.reaction.routes.quality.protection.scanner",
        "CompetingInteraction",
    ),
    "CompetingSitesScore": (
        "synplan.chem.reaction.routes.quality.protection.scorer",
        "CompetingSitesScore",
    ),
    "FunctionalGroupDetector": (
        "synplan.chem.reaction.routes.quality.protection.functional_groups",
        "FunctionalGroupDetector",
    ),
    "FunctionalGroupMatch": (
        "synplan.chem.reaction.routes.quality.protection.functional_groups",
        "FunctionalGroupMatch",
    ),
    "HalogenDetector": (
        "synplan.chem.reaction.routes.quality.protection.functional_groups",
        "HalogenDetector",
    ),
    "HalogenMatch": (
        "synplan.chem.reaction.routes.quality.protection.functional_groups",
        "HalogenMatch",
    ),
    "IncompatibilityMatrix": (
        "synplan.chem.reaction.routes.quality.protection.scanner",
        "IncompatibilityMatrix",
    ),
    "ProtectionConfig": (
        "synplan.chem.reaction.routes.quality.protection.config",
        "ProtectionConfig",
    ),
    "ProtectionRouteScorer": (
        "synplan.chem.reaction.routes.quality.scorer",
        "ProtectionRouteScorer",
    ),
    "RouteScanner": (
        "synplan.chem.reaction.routes.quality.protection.scanner",
        "RouteScanner",
    ),
    "RouteScorer": ("synplan.chem.reaction.routes.quality.scorer", "RouteScorer"),
    "classify_reaction_type": (
        "synplan.chem.reaction.routes.quality.protection.reaction_classifier",
        "classify_reaction_type",
    ),
    "classify_reaction_type_broad": (
        "synplan.chem.reaction.routes.quality.protection.reaction_classifier",
        "classify_reaction_type_broad",
    ),
    "classify_reaction_type_detailed": (
        "synplan.chem.reaction.routes.quality.protection.reaction_classifier",
        "classify_reaction_type_detailed",
    ),
    "get_reaction_center_atoms": (
        "synplan.chem.reaction.routes.quality.protection.reaction_classifier",
        "get_reaction_center_atoms",
    ),
}

__all__ = [
    "CompetingInteraction",
    "CompetingSitesScore",
    "FunctionalGroupDetector",
    "FunctionalGroupMatch",
    "HalogenDetector",
    "HalogenMatch",
    "IncompatibilityMatrix",
    "ProtectionConfig",
    "ProtectionRouteScorer",
    "RouteScanner",
    "RouteScorer",
    "classify_reaction_type",
    "classify_reaction_type_broad",
    "classify_reaction_type_detailed",
    "get_reaction_center_atoms",
]


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'synplan.route_quality' has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_LAZY_EXPORTS])
