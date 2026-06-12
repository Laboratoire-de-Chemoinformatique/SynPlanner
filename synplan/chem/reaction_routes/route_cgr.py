"""Back-compat shim; import from `synplan.chem.reaction.routes.representation` instead."""

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.routes.representation")

from synplan.chem.reaction.routes.representation import *
from synplan.chem.reaction.routes.representation import (
    compose_all_route_cgrs as compose_all_route_cgrs,
    compose_all_sb_cgrs as compose_all_sb_cgrs,
    compose_route_cgr as compose_route_cgr,
    compose_sb_cgr as compose_sb_cgr,
    extract_reactions as extract_reactions,
    find_next_atom_num as find_next_atom_num,
    get_clean_mapping as get_clean_mapping,
    get_leaving_groups as get_leaving_groups,
    process_first_reaction as process_first_reaction,
    process_target_blocks as process_target_blocks,
    update_reaction_dict as update_reaction_dict,
    validate_molecule_components as validate_molecule_components,
)
