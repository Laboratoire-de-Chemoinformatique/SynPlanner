"""Back-compat shim; import from `synplan.chem.reaction.rules.analysis` instead."""

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.rules.analysis")

from synplan.chem.reaction.rules.analysis import *
from synplan.chem.reaction.rules.analysis import RuleSet as RuleSet

__all__ = [
    "RuleSet",
]
