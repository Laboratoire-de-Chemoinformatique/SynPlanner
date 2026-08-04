"""Compatibility facade for the historical route-clustering module.

Clustering now lives in ``routes.clustering`` and its CLI in ``routes.cli``.
This module preserves the import path and legacy tuple-returning adapters used
by the ``main`` branch.
"""

from __future__ import annotations

import importlib
import warnings

_EXPORTS = {
    "CGRContainer": ("chython.containers", "CGRContainer"),
    "DynamicBond": ("chython.containers.bonds", "DynamicBond"),
    "DynamicX": (
        "synplan.chem.reaction.routes.clustering.pseudo_atoms",
        "DynamicX",
    ),
    "MarkedAt": (
        "synplan.chem.reaction.routes.clustering.pseudo_atoms",
        "MarkedAt",
    ),
    "MarkedY": (
        "synplan.chem.reaction.routes.clustering.pseudo_atoms",
        "MarkedY",
    ),
    "MoleculeContainer": ("chython.containers", "MoleculeContainer"),
    "ReactionContainer": ("chython.containers", "ReactionContainer"),
    "SubclusterError": (
        "synplan.chem.reaction.routes.clustering",
        "SubclusterError",
    ),
    "all_lg_collect": (
        "synplan.chem.reaction.routes.clustering",
        "all_lg_collect",
    ),
    "cluster_route_from_csv": (
        "synplan.chem.reaction.routes.clustering",
        "cluster_route_from_csv",
    ),
    "cluster_route_from_json": (
        "synplan.chem.reaction.routes.clustering",
        "cluster_route_from_json",
    ),
    "cluster_routes": (
        "synplan.chem.reaction.routes.clustering",
        "cluster_routes",
    ),
    "compose_all_route_cgrs": (
        "synplan.chem.reaction.routes.representation",
        "compose_all_route_cgrs",
    ),
    "compose_all_sb_cgrs": (
        "synplan.chem.reaction.routes.representation",
        "compose_all_sb_cgrs",
    ),
    "extract_strat_bonds": (
        "synplan.chem.reaction.routes.clustering",
        "extract_strat_bonds",
    ),
    "group_by_identical_values": (
        "synplan.chem.reaction.routes.clustering",
        "group_by_identical_values",
    ),
    "group_routes_by_synthon_detail": (
        "synplan.chem.reaction.routes.clustering",
        "group_routes_by_synthon_detail",
    ),
    "lg_process_reset": (
        "synplan.chem.reaction.routes.clustering",
        "lg_process_reset",
    ),
    "lg_reaction_replacer": (
        "synplan.chem.reaction.routes.clustering",
        "lg_reaction_replacer",
    ),
    "lg_replacer": (
        "synplan.chem.reaction.routes.clustering",
        "lg_replacer",
    ),
    "make_dict": ("synplan.chem.reaction.routes.io", "make_dict"),
    "make_json": ("synplan.chem.reaction.routes.io", "make_json"),
    "new_lg_reaction_replacer": (
        "synplan.chem.reaction.routes.clustering",
        "new_lg_reaction_replacer",
    ),
    "post_process_subgroup": (
        "synplan.chem.reaction.routes.clustering",
        "post_process_subgroup",
    ),
    "read_routes_csv": (
        "synplan.chem.reaction.routes.io",
        "read_routes_csv",
    ),
    "read_routes_json": (
        "synplan.chem.reaction.routes.io",
        "read_routes_json",
    ),
    "remove_and_shift": (
        "synplan.chem.reaction.routes.clustering",
        "remove_and_shift",
    ),
    "replace_leaving_groups_in_synthon": (
        "synplan.chem.reaction.routes.clustering",
        "replace_leaving_groups_in_synthon",
    ),
    "replace_supporting_reactants_with_y": (
        "synplan.chem.reaction.routes.clustering",
        "replace_supporting_reactants_with_y",
    ),
    "run_cluster_cli": (
        "synplan.chem.reaction.routes.cli",
        "run_cluster_cli",
    ),
    "subcluster_all_clusters": (
        "synplan.chem.reaction.routes.clustering",
        "subcluster_all_clusters",
    ),
    "subcluster_one_cluster": (
        "synplan.chem.reaction.routes.clustering",
        "subcluster_one_cluster",
    ),
    "supporting_groups_from_route_cgr": (
        "synplan.chem.reaction.routes.clustering",
        "supporting_groups_from_route_cgr",
    ),
    "write_routes_csv": (
        "synplan.chem.reaction.routes.io",
        "write_routes_csv",
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
