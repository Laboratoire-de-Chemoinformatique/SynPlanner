"""Back-compat shim; import from `synplan.chem.reaction.routes` instead."""

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.routes")

from synplan.chem.reaction.routes import *  # noqa: F403
from synplan.chem.reaction.routes import __all__  # noqa: F401
