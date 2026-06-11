"""Back-compat shim; import from `synplan.chem.reaction.routes.leaving_groups` instead."""

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.routes.leaving_groups")

from synplan.chem.reaction.routes.leaving_groups import *
from synplan.chem.reaction.routes.leaving_groups import (
    DynamicX as DynamicX,
    Marked as Marked,
    MarkedAt as MarkedAt,
)
