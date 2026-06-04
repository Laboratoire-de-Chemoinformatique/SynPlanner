"""Chython template features for MHN-style ranking policies."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import torch
from chython import smarts
from chython.containers import MoleculeContainer

from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.utils import reaction_query_to_reaction
from synplan.utils.loading import load_reaction_rules

_MAX_TEMPLATE_FEATURE_CACHE_SIZE = 8
_TEMPLATE_FEATURE_CACHE: OrderedDict[str, torch.Tensor] = OrderedDict()
_POLICY_DATA_SUFFIX = "_policy_data.tsv"


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
    max_size: int = _MAX_TEMPLATE_FEATURE_CACHE_SIZE,
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
        raise ValueError("mhn_template_fp_size must be a positive power of two")
    if min_radius < 0:
        raise ValueError("mhn_template_fp_min_radius must be >= 0")
    if max_radius < min_radius:
        raise ValueError(
            "mhn_template_fp_max_radius must be >= mhn_template_fp_min_radius"
        )
    if active_bits <= 0:
        raise ValueError("mhn_template_fp_active_bits must be > 0")


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


def template_feature_digest(
    rule_smarts: Sequence[str],
    *,
    fp_size: int,
    min_radius: int,
    max_radius: int,
    active_bits: int,
) -> str:
    """Hash ordered rules and all settings that affect template features."""
    payload = {
        "rules": tuple(rule_smarts),
        "fp_size": fp_size,
        "min_radius": min_radius,
        "max_radius": max_radius,
        "active_bits": active_bits,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _side_fingerprint(
    molecules: Sequence[MoleculeContainer],
    *,
    fp_size: int,
    min_radius: int,
    max_radius: int,
    active_bits: int,
) -> torch.Tensor:
    """Max-pool fragment fingerprints for one side of a reaction rule."""
    if not molecules:
        return torch.zeros(fp_size, dtype=torch.float)

    fingerprints = [
        torch.as_tensor(
            molecule.morgan_fingerprint(
                min_radius=min_radius,
                max_radius=max_radius,
                length=fp_size,
                number_active_bits=active_bits,
            ),
            dtype=torch.float,
        )
        for molecule in molecules
    ]
    return torch.stack(fingerprints).amax(dim=0)


def template_features_from_smarts(
    rule_smarts: Sequence[str],
    *,
    fp_size: int = 2048,
    min_radius: int = 1,
    max_radius: int = 4,
    active_bits: int = 2,
) -> torch.Tensor:
    """Build Chython side-delta template fingerprints for retrospective rules.

    The left side is the target query and the right side contains precursors.
    Fragment fingerprints are max-pooled on each side before calculating
    ``target - 0.5 * precursors``.
    """
    validate_morgan_settings(
        fp_size=fp_size,
        min_radius=min_radius,
        max_radius=max_radius,
        active_bits=active_bits,
    )
    rules = tuple(rule_smarts)
    digest = template_feature_digest(
        rules,
        fp_size=fp_size,
        min_radius=min_radius,
        max_radius=max_radius,
        active_bits=active_bits,
    )
    cached = _cache_get(_TEMPLATE_FEATURE_CACHE, digest)
    if cached is not None:
        return cached

    features = []
    for index, rule_smarts_text in enumerate(rules):
        try:
            reaction = reaction_query_to_reaction(smarts(rule_smarts_text))
            target = _side_fingerprint(
                reaction.reactants,
                fp_size=fp_size,
                min_radius=min_radius,
                max_radius=max_radius,
                active_bits=active_bits,
            )
            precursors = _side_fingerprint(
                reaction.products,
                fp_size=fp_size,
                min_radius=min_radius,
                max_radius=max_radius,
                active_bits=active_bits,
            )
            features.append(target - 0.5 * precursors)
        except Exception as err:
            raise ValueError(
                f"Failed to fingerprint reaction rule at index {index}:\n"
                f"  SMARTS: {rule_smarts_text}\n"
                f"  error: {type(err).__name__}: {err}"
            ) from err

    tensor = (
        torch.stack(features)
        if features
        else torch.empty((0, fp_size), dtype=torch.float)
    )
    _cache_set(_TEMPLATE_FEATURE_CACHE, digest, tensor)
    return tensor
