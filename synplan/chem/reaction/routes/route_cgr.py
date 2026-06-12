"""Back-compat shim; import from `synplan.chem.reaction.routes.representation` instead."""

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.routes.representation")

from synplan.chem.reaction.routes.representation import *
from synplan.chem.reaction.routes.representation import (
    compose_all_route_cgrs as compose_all_route_cgrs,
)
from synplan.chem.reaction.routes.representation import (
    compose_all_sb_cgrs as compose_all_sb_cgrs,
)
from synplan.chem.reaction.routes.representation import (
    compose_route_cgr as compose_route_cgr,
)
from synplan.chem.reaction.routes.representation import (
    compose_sb_cgr as compose_sb_cgr,
)
from synplan.chem.reaction.routes.representation import (
    extract_reactions as extract_reactions,
)
from synplan.chem.reaction.routes.representation import (
    find_next_atom_num as find_next_atom_num,
)
from synplan.chem.reaction.routes.representation import (
    get_clean_mapping as get_clean_mapping,
)
from synplan.chem.reaction.routes.representation import (
    get_leaving_groups as get_leaving_groups,
)
from synplan.chem.reaction.routes.representation import (
    process_first_reaction as process_first_reaction,
)
from synplan.chem.reaction.routes.representation import (
    process_target_blocks as process_target_blocks,
)
from synplan.chem.reaction.routes.representation import (
    update_reaction_dict as update_reaction_dict,
)
from synplan.chem.reaction.routes.representation import (
    validate_molecule_components as validate_molecule_components,
)
