"""Back-compat shim; import from `synplan.chem.reaction.rules.extraction` instead."""

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.rules.extraction")

from synplan.chem.reaction.rules.extraction import *
from synplan.chem.reaction.rules.extraction import (
    _make_extracted_rule_record as _make_extracted_rule_record,
)
