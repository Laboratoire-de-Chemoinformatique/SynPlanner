"""Reaction engine package.

Re-exports the public API of :mod:`synplan.chem.reaction.reactor` so the
historical dotted path ``synplan.chem.reaction`` keeps exposing the same
names after the module was promoted to a package.
"""

from synplan.chem.reaction.reactor import (
    CanonicalRetroReactor,
    Reaction,
    add_small_mols,
    apply_reaction_rule,
)

__all__ = [
    "CanonicalRetroReactor",
    "Reaction",
    "add_small_mols",
    "apply_reaction_rule",
]
