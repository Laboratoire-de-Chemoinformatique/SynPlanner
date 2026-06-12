"""Torch-free file-IO and path helpers for reaction-rule representations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synplan.chem.reaction import CanonicalRetroReactor

_POLICY_DATA_SUFFIX = "_policy_data.tsv"


def rule_smarts_from_reactors(
    reaction_rules: Sequence[CanonicalRetroReactor],
) -> tuple[str, ...]:
    """Return SMARTS strings in the same order as loaded runtime reactors."""
    return tuple(str(rule) for rule in reaction_rules)


def load_rule_smarts(reaction_rules_path: str | Path) -> tuple[str, ...]:
    """Load an ordered rule set and return its runtime SMARTS representation."""
    # Lazy import: ``synplan.utils.loading`` pulls torch, which would break the
    # torch-free guarantee of the representation package.
    from synplan.utils.loading import load_reaction_rules

    return rule_smarts_from_reactors(load_reaction_rules(str(reaction_rules_path)))


def reaction_rules_path_from_policy_data(policy_data_path: str | Path) -> Path:
    """Return the sibling rules TSV emitted with an extracted policy mapping."""
    policy_data_path = Path(policy_data_path)
    if not policy_data_path.name.endswith(_POLICY_DATA_SUFFIX):
        raise ValueError(
            "mhn_ranking policy data must be an extracted '*_policy_data.tsv' file, "
            f"got {policy_data_path.name!r}"
        )
    rules_path = policy_data_path.with_name(
        policy_data_path.name[: -len(_POLICY_DATA_SUFFIX)] + ".tsv"
    )
    if not rules_path.is_file():
        raise FileNotFoundError(
            f"Could not find the reaction rules paired with {policy_data_path}: "
            f"expected {rules_path}"
        )
    return rules_path


__all__ = [
    "load_rule_smarts",
    "reaction_rules_path_from_policy_data",
    "rule_smarts_from_reactors",
]
