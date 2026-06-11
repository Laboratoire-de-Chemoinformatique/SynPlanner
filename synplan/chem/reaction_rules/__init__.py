"""Back-compat shim; import from `synplan.chem.reaction.rules` instead."""

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.rules")

from synplan.chem.reaction.rules import *
from synplan.chem.reaction.rules import (
    POLICY_SOURCE_NAME as POLICY_SOURCE_NAME,
)
from synplan.chem.reaction.rules import (
    PrioritySmartsError as PrioritySmartsError,
)
from synplan.chem.reaction.rules import (
    RuleSet as RuleSet,
)
from synplan.chem.reaction.rules import (
    parse_priority_rules as parse_priority_rules,
)

__all__ = [
    "POLICY_SOURCE_NAME",
    "PrioritySmartsError",
    "RuleSet",
    "parse_priority_rules",
]
