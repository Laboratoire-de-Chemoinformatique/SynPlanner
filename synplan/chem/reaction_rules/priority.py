"""Back-compat shim; import from `synplan.chem.reaction.rules.priority` instead."""

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.rules.priority")

from synplan.chem.reaction.rules.priority import *
from synplan.chem.reaction.rules.priority import (
    POLICY_SOURCE_NAME as POLICY_SOURCE_NAME,
)
from synplan.chem.reaction.rules.priority import (
    PrioritySmartsError as PrioritySmartsError,
)
from synplan.chem.reaction.rules.priority import (
    parse_priority_rules as parse_priority_rules,
)

__all__ = [
    "POLICY_SOURCE_NAME",
    "PrioritySmartsError",
    "parse_priority_rules",
]
