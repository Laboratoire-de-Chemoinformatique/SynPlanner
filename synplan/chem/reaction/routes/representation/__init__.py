"""Route-level condensed graph of reaction helpers."""

from synplan.chem.reaction.routes.representation.components import (
    route_cgr_pseudo_reactants_by_role,
)
from synplan.chem.reaction.routes.representation.container import (
    RouteCGRContainer,
    enable_route_cgr_container,
)
from synplan.chem.reaction.routes.representation.deconvolution import (
    reactions_from_route_cgr,
    routes_dict_from_route_cgrs,
)
from synplan.chem.reaction.routes.representation.hash import (
    RouteCGRGraph,
    atom_label,
    bond_label,
    compare_route_cgr_dicts,
    hash_route_cgrs,
    route_cgr_bucket_fingerprint,
    route_cgr_bucket_hash,
    route_cgr_fingerprint,
    route_cgr_fingerprint_without_route_order,
    route_cgr_graph,
    route_cgr_hash,
    route_cgr_hash_without_route_order,
    route_cgr_metadata,
    route_cgrs_equal,
    route_order_variant_sets,
)
from synplan.chem.reaction.routes.representation.route_cgr import (
    build_route_cgr,
    compose_all_route_cgrs,
    compose_route_cgr,
    extract_reactions,
    find_next_atom_num,
    get_clean_mapping,
    get_leaving_groups,
    process_first_reaction,
    process_target_blocks,
    update_reaction_dict,
    validate_molecule_components,
)
from synplan.chem.reaction.routes.representation.sb_cgr import (
    compose_all_sb_cgrs,
    compose_sb_cgr,
)
from synplan.chem.reaction.routes.representation.state import (
    RouteDynamicBond,
    remove_transient_bonds,
    route_atom,
    transient_bond,
)

__all__ = [
    "RouteCGRContainer",
    "RouteCGRGraph",
    "RouteDynamicBond",
    "atom_label",
    "bond_label",
    "build_route_cgr",
    "compare_route_cgr_dicts",
    "compose_all_route_cgrs",
    "compose_all_sb_cgrs",
    "compose_route_cgr",
    "compose_sb_cgr",
    "depict_route_cgr",
    "enable_route_cgr_container",
    "extract_reactions",
    "find_next_atom_num",
    "get_clean_mapping",
    "get_leaving_groups",
    "hash_route_cgrs",
    "process_first_reaction",
    "process_target_blocks",
    "reactions_from_route_cgr",
    "remove_transient_bonds",
    "route_atom",
    "route_cgr_bucket_fingerprint",
    "route_cgr_bucket_hash",
    "route_cgr_fingerprint",
    "route_cgr_fingerprint_without_route_order",
    "route_cgr_graph",
    "route_cgr_hash",
    "route_cgr_hash_without_route_order",
    "route_cgr_metadata",
    "route_cgr_pseudo_reactants_by_role",
    "route_cgrs_equal",
    "route_order_variant_sets",
    "routes_dict_from_route_cgrs",
    "transient_bond",
    "update_reaction_dict",
    "validate_molecule_components",
]


def __getattr__(name: str):
    if name != "depict_route_cgr":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from synplan.chem.reaction.routes.representation.depiction import depict_route_cgr

    globals()[name] = depict_route_cgr
    return depict_route_cgr
