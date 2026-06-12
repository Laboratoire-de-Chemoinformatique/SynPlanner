"""Reaction-route domain code, organized by artifact.

Modules:
    representation   route-CGR + strategic-bond CGR builders
    clustering       cluster / subcluster logic
    io               csv/json route loaders + writers, tree export, loaders
    leaving_groups   chython LG primitives + route-level LG operations
    cli              ``run_cluster_cli`` command-line entry point
    visualisation    depiction helpers

This ``__init__`` curates the package's public surface. Members are resolved
**lazily** (PEP 562 ``__getattr__``): importing ``synplan.chem.reaction.routes``
does not eagerly execute the heavy submodules (``cli`` pulls IPython/chython
depiction, ``io`` pulls ``synplan.mcts.tree``). This keeps the planning path
light and prevents import cycles through the back-compat shims.
"""

import importlib
from typing import TYPE_CHECKING

# public name -> submodule that defines it
_EXPORTS = {
    # cli
    "run_cluster_cli": "cli",
    # clustering
    "SubclusterError": "clustering",
    "cluster_routes": "clustering",
    "extract_strat_bonds": "clustering",
    "group_by_identical_values": "clustering",
    "group_routes_by_synthon_detail": "clustering",
    "post_process_subgroup": "clustering",
    "replace_leaving_groups_in_synthon": "clustering",
    "subcluster_all_clusters": "clustering",
    "subcluster_one_cluster": "clustering",
    # io
    "cluster_route_from_csv": "io",
    "cluster_route_from_json": "io",
    "make_dict": "io",
    "make_json": "io",
    "read_routes_csv": "io",
    "read_routes_json": "io",
    "write_routes_csv": "io",
    "write_routes_json": "io",
    # leaving_groups
    "DynamicX": "leaving_groups",
    "Marked": "leaving_groups",
    "MarkedAt": "leaving_groups",
    "all_lg_collect": "leaving_groups",
    "lg_process_reset": "leaving_groups",
    "lg_reaction_replacer": "leaving_groups",
    "lg_replacer": "leaving_groups",
    "new_lg_reaction_replacer": "leaving_groups",
    # representation
    "compose_all_route_cgrs": "representation",
    "compose_all_sb_cgrs": "representation",
    "compose_route_cgr": "representation",
    "compose_sb_cgr": "representation",
    "extract_reactions": "representation",
    "find_next_atom_num": "representation",
    "get_clean_mapping": "representation",
    "get_leaving_groups": "representation",
    "process_first_reaction": "representation",
    "process_target_blocks": "representation",
    "update_reaction_dict": "representation",
    "validate_molecule_components": "representation",
    # visualisation
    "CustomDepictMolecule": "visualisation",
    "WideBondDepictCGR": "visualisation",
    "cgr_display": "visualisation",
    "depict_custom_reaction": "visualisation",
    "remove_and_shift": "visualisation",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value  # cache so __getattr__ runs at most once per name
    return value


def __dir__():
    return __all__


if TYPE_CHECKING:  # static re-exports for type checkers / IDEs (no runtime cost)
    from synplan.chem.reaction.routes.cli import run_cluster_cli
    from synplan.chem.reaction.routes.clustering import (
        SubclusterError,
        cluster_routes,
        extract_strat_bonds,
        group_by_identical_values,
        group_routes_by_synthon_detail,
        post_process_subgroup,
        replace_leaving_groups_in_synthon,
        subcluster_all_clusters,
        subcluster_one_cluster,
    )
    from synplan.chem.reaction.routes.io import (
        cluster_route_from_csv,
        cluster_route_from_json,
        make_dict,
        make_json,
        read_routes_csv,
        read_routes_json,
        write_routes_csv,
        write_routes_json,
    )
    from synplan.chem.reaction.routes.leaving_groups import (
        DynamicX,
        Marked,
        MarkedAt,
        all_lg_collect,
        lg_process_reset,
        lg_reaction_replacer,
        lg_replacer,
        new_lg_reaction_replacer,
    )
    from synplan.chem.reaction.routes.representation import (
        compose_all_route_cgrs,
        compose_all_sb_cgrs,
        compose_route_cgr,
        compose_sb_cgr,
        extract_reactions,
        find_next_atom_num,
        get_clean_mapping,
        get_leaving_groups,
        process_first_reaction,
        process_target_blocks,
        update_reaction_dict,
        validate_molecule_components,
    )
    from synplan.chem.reaction.routes.visualisation import (
        CustomDepictMolecule,
        WideBondDepictCGR,
        cgr_display,
        depict_custom_reaction,
        remove_and_shift,
    )
