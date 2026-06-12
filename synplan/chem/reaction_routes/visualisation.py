"""Back-compat shim; import from `synplan.chem.reaction.routes.visualisation` instead."""

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.routes.visualisation")

from synplan.chem.reaction.routes.visualisation import *
from synplan.chem.reaction.routes.visualisation import (
    CustomDepictMolecule as CustomDepictMolecule,
    WideBondDepictCGR as WideBondDepictCGR,
    cgr_display as cgr_display,
    depict_custom_reaction as depict_custom_reaction,
    remove_and_shift as remove_and_shift,
)
