"""Compatibility facade for the historical route-CGR module.

The route package was reorganized under ``routes.representation``. Keep the
module path used by ``main`` importers while resolving each symbol lazily from
its canonical implementation.
"""

from __future__ import annotations

import importlib
import warnings

_EXPORTS = {
    "CGRContainer": ("chython.containers", "CGRContainer"),
    "DynamicBond": ("chython.containers.bonds", "DynamicBond"),
    "MoleculeContainer": ("chython.containers", "MoleculeContainer"),
    "ReactionContainer": ("chython.containers", "ReactionContainer"),
    "Tree": ("synplan.mcts.tree", "Tree"),
    "compose_all_route_cgrs": (
        "synplan.chem.reaction.routes.representation.route_cgr",
        "compose_all_route_cgrs",
    ),
    "compose_all_sb_cgrs": (
        "synplan.chem.reaction.routes.representation.sb_cgr",
        "compose_all_sb_cgrs",
    ),
    "compose_route_cgr": (
        "synplan.chem.reaction.routes.representation.route_cgr",
        "compose_route_cgr",
    ),
    "compose_sb_cgr": (
        "synplan.chem.reaction.routes.representation.sb_cgr",
        "compose_sb_cgr",
    ),
    "extract_reactions": (
        "synplan.chem.reaction.routes.representation.route_cgr",
        "extract_reactions",
    ),
    "find_next_atom_num": (
        "synplan.chem.reaction.routes.representation.route_cgr",
        "find_next_atom_num",
    ),
    "get_clean_mapping": (
        "synplan.chem.reaction.routes.representation.route_cgr",
        "get_clean_mapping",
    ),
    "get_leaving_groups": (
        "synplan.chem.reaction.routes.representation.route_cgr",
        "get_leaving_groups",
    ),
    "process_first_reaction": (
        "synplan.chem.reaction.routes.representation.route_cgr",
        "process_first_reaction",
    ),
    "process_target_blocks": (
        "synplan.chem.reaction.routes.representation.route_cgr",
        "process_target_blocks",
    ),
    "update_reaction_dict": (
        "synplan.chem.reaction.routes.representation.route_cgr",
        "update_reaction_dict",
    ),
    "validate_molecule_components": (
        "synplan.chem.reaction.routes.representation.route_cgr",
        "validate_molecule_components",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"{__name__}.{name} is deprecated; import it from {target[0]} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__():
    return __all__
