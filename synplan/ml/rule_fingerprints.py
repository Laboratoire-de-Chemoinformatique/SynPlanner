"""Chython rule fingerprints for MHN-style ranking policies."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from chython import smarts
from chython.containers import MoleculeContainer, ReactionContainer

from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.reaction_rules.fingerprints import (
    query_cgr_morgan_fingerprint,
    query_reaction_atom_labels,
)
from synplan.chem.utils import reaction_query_to_reaction
from synplan.utils.loading import load_reaction_rules

RuleFingerprintType = Literal["legacy", "query_cgr"]

RULE_FINGERPRINT_SCHEMA_VERSION = "1"
_MAX_RULE_FINGERPRINT_CACHE_SIZE = 8
_RULE_FINGERPRINT_CACHE: OrderedDict[str, torch.Tensor] = OrderedDict()
_POLICY_DATA_SUFFIX = "_policy_data.tsv"
_RULE_FINGERPRINT_TYPES = {"legacy", "query_cgr"}


def _cache_get(cache: OrderedDict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_set(
    cache: OrderedDict[str, torch.Tensor],
    key: str,
    value: torch.Tensor,
    *,
    max_size: int = _MAX_RULE_FINGERPRINT_CACHE_SIZE,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def validate_morgan_settings(
    *,
    fp_size: int,
    min_radius: int,
    max_radius: int,
    active_bits: int,
) -> None:
    """Validate Chython Morgan fingerprint settings."""
    if fp_size <= 0 or fp_size & (fp_size - 1):
        raise ValueError("mhn_rule_fp_size must be a positive power of two")
    if min_radius <= 0:
        raise ValueError("mhn_rule_fp_min_radius must be > 0")
    if max_radius < min_radius:
        raise ValueError("mhn_rule_fp_max_radius must be >= mhn_rule_fp_min_radius")
    if active_bits <= 0:
        raise ValueError("mhn_rule_fp_active_bits must be > 0")


def validate_rule_fingerprint_type(fp_type: str) -> None:
    """Validate the configured rule fingerprint implementation."""
    if fp_type not in _RULE_FINGERPRINT_TYPES:
        expected = "', '".join(sorted(_RULE_FINGERPRINT_TYPES))
        raise ValueError(f"mhn_rule_fp_type must be one of '{expected}'")


@dataclass(frozen=True)
class RuleFingerprintConfig:
    """Configuration that fully identifies an MHN rule fingerprint schema."""

    fp_size: int = 2048
    min_radius: int = 1
    max_radius: int = 4
    active_bits: int = 2
    fp_type: RuleFingerprintType = "query_cgr"
    schema_version: str = RULE_FINGERPRINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_morgan_settings(
            fp_size=self.fp_size,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            active_bits=self.active_bits,
        )
        validate_rule_fingerprint_type(self.fp_type)
        if not self.schema_version:
            raise ValueError("mhn_rule_fp_schema_version must be non-empty")

    def to_digest_payload(self) -> dict[str, int | str]:
        """Return stable serializable values that affect rule fingerprints."""
        return {
            "fp_size": self.fp_size,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "active_bits": self.active_bits,
            "fp_type": self.fp_type,
            "schema_version": self.schema_version,
        }


def rule_smarts_from_reactors(
    reaction_rules: Sequence[CanonicalRetroReactor],
) -> tuple[str, ...]:
    """Return SMARTS strings in the same order as loaded runtime reactors."""
    return tuple(str(rule) for rule in reaction_rules)


def load_rule_smarts(reaction_rules_path: str | Path) -> tuple[str, ...]:
    """Load an ordered rule set and return its runtime SMARTS representation."""
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


def rule_fingerprint_digest(
    rule_smarts: Sequence[str],
    fingerprint_config: RuleFingerprintConfig | None = None,
) -> str:
    """Hash ordered rules and all settings that affect rule fingerprints."""
    fingerprint_config = fingerprint_config or RuleFingerprintConfig()
    payload = {
        "rules": tuple(rule_smarts),
        "fingerprint_config": fingerprint_config.to_digest_payload(),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _side_fingerprint(
    molecules: Sequence[MoleculeContainer],
    fingerprint_config: RuleFingerprintConfig,
) -> torch.Tensor:
    """Max-pool fragment fingerprints for one side of a reaction rule."""
    if not molecules:
        return torch.zeros(fingerprint_config.fp_size, dtype=torch.float)

    fingerprints = [
        torch.as_tensor(
            molecule.morgan_fingerprint(
                min_radius=fingerprint_config.min_radius,
                max_radius=fingerprint_config.max_radius,
                length=fingerprint_config.fp_size,
                number_active_bits=fingerprint_config.active_bits,
            ),
            dtype=torch.float,
        )
        for molecule in molecules
    ]
    return torch.stack(fingerprints).amax(dim=0)


def _legacy_rule_fingerprint(
    rule_query: ReactionContainer,
    fingerprint_config: RuleFingerprintConfig,
) -> torch.Tensor:
    reaction = reaction_query_to_reaction(rule_query)
    target = _side_fingerprint(reaction.reactants, fingerprint_config)
    precursors = _side_fingerprint(reaction.products, fingerprint_config)
    return target - 0.5 * precursors


def _query_cgr_rule_fingerprint(
    rule_query: ReactionContainer,
    fingerprint_config: RuleFingerprintConfig,
) -> torch.Tensor:
    return torch.as_tensor(
        query_cgr_morgan_fingerprint(
            rule_query.compose(dynamic=True),
            atom_labels=query_reaction_atom_labels(rule_query),
            min_radius=fingerprint_config.min_radius,
            max_radius=fingerprint_config.max_radius,
            length=fingerprint_config.fp_size,
            number_active_bits=fingerprint_config.active_bits,
        ),
        dtype=torch.float,
    )


def rule_fingerprints_from_smarts(
    rule_smarts: Sequence[str],
    fingerprint_config: RuleFingerprintConfig | None = None,
) -> torch.Tensor:
    """Build ordered rule fingerprints for retrospective reaction SMARTS.

    ``legacy`` uses the MHN-react-style side delta, ``target - 0.5 * precursors``,
    after converting query rules to ordinary reaction containers. ``query_cgr``
    fingerprints Chython ``QueryCGRContainer`` objects with original query-side
    atom labels so constraints such as hydrogen count and ring size are retained.
    """
    fingerprint_config = fingerprint_config or RuleFingerprintConfig()
    rules = tuple(rule_smarts)
    fingerprint_digest = rule_fingerprint_digest(rules, fingerprint_config)
    cached = _cache_get(_RULE_FINGERPRINT_CACHE, fingerprint_digest)
    if cached is not None:
        return cached

    fingerprints = []
    for index, rule_smarts_text in enumerate(rules):
        try:
            rule_query = smarts(rule_smarts_text)
            if fingerprint_config.fp_type == "legacy":
                fingerprint = _legacy_rule_fingerprint(rule_query, fingerprint_config)
            else:
                fingerprint = _query_cgr_rule_fingerprint(
                    rule_query, fingerprint_config
                )
            fingerprints.append(fingerprint)
        except Exception as err:
            raise ValueError(
                f"Failed to fingerprint reaction rule at index {index}:\n"
                f"  SMARTS: {rule_smarts_text}\n"
                f"  error: {type(err).__name__}: {err}"
            ) from err

    tensor = (
        torch.stack(fingerprints)
        if fingerprints
        else torch.empty((0, fingerprint_config.fp_size), dtype=torch.float)
    )
    _cache_set(_RULE_FINGERPRINT_CACHE, fingerprint_digest, tensor)
    return tensor
