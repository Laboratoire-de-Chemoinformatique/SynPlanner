"""Back-compat shim; import from `synplan.chem.reaction.rules.extraction` instead."""

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.rules.extraction")

from synplan.chem.reaction.rules.extraction import *
from synplan.chem.reaction.rules.extraction import (
    RuleExtractionConfig as RuleExtractionConfig,
)
from synplan.chem.reaction.rules.extraction import (
    _make_extracted_rule_record as _make_extracted_rule_record,
)
from synplan.chem.reaction.rules.extraction import (
    _process_extraction_result as _process_extraction_result,
)
from synplan.chem.reaction.rules.extraction import (
    add_environment_atoms as add_environment_atoms,
)
from synplan.chem.reaction.rules.extraction import (
    add_functional_groups as add_functional_groups,
)
from synplan.chem.reaction.rules.extraction import (
    add_leaving_incoming_groups as add_leaving_incoming_groups,
)
from synplan.chem.reaction.rules.extraction import (
    add_ring_structures as add_ring_structures,
)
from synplan.chem.reaction.rules.extraction import (
    assemble_final_rule as assemble_final_rule,
)
from synplan.chem.reaction.rules.extraction import (
    clean_atom as clean_atom,
)
from synplan.chem.reaction.rules.extraction import (
    clean_molecules as clean_molecules,
)
from synplan.chem.reaction.rules.extraction import (
    create_rule as create_rule,
)
from synplan.chem.reaction.rules.extraction import (
    create_substructures_and_reagents as create_substructures_and_reagents,
)
from synplan.chem.reaction.rules.extraction import (
    extract_rules as extract_rules,
)
from synplan.chem.reaction.rules.extraction import (
    extract_rules_from_reactions as extract_rules_from_reactions,
)
from synplan.chem.reaction.rules.extraction import (
    molecule_substructure_as_query as molecule_substructure_as_query,
)
from synplan.chem.reaction.rules.extraction import (
    sort_rules as sort_rules,
)
from synplan.chem.reaction.rules.extraction import (
    validate_rule as validate_rule,
)

__all__ = [
    "RuleExtractionConfig",
    "add_environment_atoms",
    "add_functional_groups",
    "add_leaving_incoming_groups",
    "add_ring_structures",
    "assemble_final_rule",
    "clean_atom",
    "clean_molecules",
    "create_rule",
    "create_substructures_and_reagents",
    "extract_rules",
    "extract_rules_from_reactions",
    "molecule_substructure_as_query",
    "sort_rules",
    "validate_rule",
]
