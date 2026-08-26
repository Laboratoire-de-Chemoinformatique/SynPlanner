"""Reaction-route domain code, organized by artifact.

Modules / subpackages::

    representation   route-CGR + strategic-bond CGR representations (route_cgr,
                     sb_cgr) plus container/state/hash/depiction helpers
    clustering       cluster / subcluster logic and pseudo-atom primitives
    quality          route quality scoring (scorer + protection subsystem)
    io               csv/json route loaders + writers, tree export
    cli              ``run_cluster_cli`` command-line entry point
    visualisation    CGR / reaction depiction helpers

This ``__init__`` curates the package's public surface. Members are resolved
**lazily** (PEP 562 ``__getattr__``): importing ``synplan.chem.reaction.routes``
does not eagerly execute the heavy submodules (``cli``/``visualisation`` pull
IPython/chython depiction, ``io`` pulls ``synplan.mcts.tree``). This keeps the
planning path light and prevents import cycles through the back-compat shims.
"""

import importlib
import warnings

# public name -> submodule (relative to this package) that defines it
_EXPORTS = {
    # cli
    "run_cluster_cli": "cli",
    # clustering (core + subclustering, re-exported by the clustering package)
    "SubclusterError": "clustering",
    "all_lg_collect": "clustering",
    "cluster_route_from_csv": "clustering",
    "cluster_route_from_json": "clustering",
    "cluster_routes": "clustering",
    "extract_strat_bonds": "clustering",
    "group_by_identical_values": "clustering",
    "group_routes_by_synthon_detail": "clustering",
    "lg_process_reset": "clustering",
    "lg_reaction_replacer": "clustering",
    "lg_replacer": "clustering",
    "new_lg_reaction_replacer": "clustering",
    "post_process_subgroup": "clustering",
    "remove_and_shift": "clustering",
    "replace_leaving_groups_in_synthon": "clustering",
    "replace_supporting_reactants_with_y": "clustering",
    "subcluster_all_clusters": "clustering",
    "subcluster_one_cluster": "clustering",
    "supporting_groups_from_route_cgr": "clustering",
    # io
    "make_dict": "io",
    "build_route_trees": "io",
    "make_json": "io",
    "read_route_tree": "io",
    "read_routes_csv": "io",
    "read_routes_json": "io",
    "write_routes_csv": "io",
    "write_routes_json": "io",
    # route facade
    "Route": "route",
    "Step": "route",
    # pseudo_atoms (clustering atom primitives)
    "DynamicX": "clustering.pseudo_atoms",
    "Marked": "clustering.pseudo_atoms",
    "MarkedAt": "clustering.pseudo_atoms",
    "MarkedY": "clustering.pseudo_atoms",
    # representation (route_cgr + sb_cgr + container/state/hash/depiction)
    "RouteCGRContainer": "representation",
    "RouteDynamicBond": "representation",
    "build_route_cgr": "representation",
    "compose_all_route_cgrs": "representation",
    "compose_all_sb_cgrs": "representation",
    "compose_route_cgr": "representation",
    "compose_sb_cgr": "representation",
    "depict_route_cgr": "representation.depiction",
    "enable_route_cgr_container": "representation",
    "extract_reactions": "representation",
    "find_next_atom_num": "representation",
    "get_clean_mapping": "representation",
    "get_leaving_groups": "representation",
    "hash_route_cgrs": "representation",
    "process_first_reaction": "representation",
    "process_target_blocks": "representation",
    "remove_transient_bonds": "representation",
    "route_cgr_hash": "representation",
    "route_cgrs_equal": "representation",
    "transient_bond": "representation",
    "update_reaction_dict": "representation",
    "validate_molecule_components": "representation",
    # visualisation
    "WideBondDepictCGR": "representation.depiction",
    "cgr_display": "representation.depiction",
    "depict_custom_reaction": "representation.depiction",
    "wide_cgr_renderer": "representation.depiction",
    # quality
    "ProtectionRouteScorer": "quality.scorer",
    "RouteScorer": "quality.scorer",
}

_LEGACY_EXPORTS = frozenset(
    {
        "all_lg_collect",
        "find_next_atom_num",
        "get_clean_mapping",
        "get_leaving_groups",
        "group_by_identical_values",
        "lg_process_reset",
        "lg_reaction_replacer",
        "new_lg_reaction_replacer",
        "process_first_reaction",
        "process_target_blocks",
        "remove_and_shift",
        "replace_leaving_groups_in_synthon",
        "replace_supporting_reactants_with_y",
        "supporting_groups_from_route_cgr",
        "transient_bond",
        "update_reaction_dict",
        "validate_molecule_components",
    }
)

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _LEGACY_EXPORTS:
        warnings.warn(
            f"{name} is a legacy route helper; import from its concrete module instead",
            DeprecationWarning,
            stacklevel=2,
        )
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value  # cache so __getattr__ runs at most once per name
    return value


def __dir__():
    return __all__
