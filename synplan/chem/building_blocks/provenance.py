"""Validation helpers for replayable deprotection provenance."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from chython import smiles as smiles_parser
from chython.containers import MoleculeContainer, ReactionContainer

from synplan.chem.utils import safe_canonicalization
from synplan.utils.audit import sha256_file

from .config import DeprotectionPolicy
from .deprotection import (
    MAX_DEPROTECTION_PASSES,
    DeprotectionEvent,
    remove_protective_groups,
)
from .rules import protective_rules_path

EXACT_DEPROTECTION_FIELDS = (
    "standardized_input_smiles",
    "deprotection_policy",
    "protective_rules_sha256",
    "deprotection_events",
    "mapped_deprotection",
)


def current_protective_rules_sha256() -> str:
    """Return the digest identifying the active protecting-group taxonomy."""
    return sha256_file(protective_rules_path())


def deprotect_molecule_with_provenance(
    molecule: MoleculeContainer,
    *,
    policy: DeprotectionPolicy = "conservative",
    max_passes: int = MAX_DEPROTECTION_PASSES,
) -> tuple[MoleculeContainer, dict[str, str] | None]:
    """Deprotect a standardized copy and return its exact replay record.

    The input molecule is not mutated. A None record means that standardization
    and deprotection produced no structural change, so restoration is unnecessary.
    """
    protected = safe_canonicalization(molecule, clean_stereo=False)
    deprotected = protected.copy()
    events: list[DeprotectionEvent] = []
    changed = remove_protective_groups(
        deprotected,
        policy=policy,
        max_passes=max_passes,
        event_collector=events,
    )
    deprotected = safe_canonicalization(deprotected, clean_stereo=False)
    if not changed or str(protected) == str(deprotected):
        return deprotected, None
    reaction = ReactionContainer(
        reactants=[protected.copy()],
        products=[deprotected.copy()],
    )
    record = {
        "standardized_input_smiles": str(protected),
        "canonical_smiles": str(deprotected),
        "deprotection_policy": policy,
        "protective_rules_sha256": current_protective_rules_sha256(),
        "deprotection_events": json.dumps(
            [event.as_dict() for event in events],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "mapped_deprotection": format(reaction, "m"),
    }
    return deprotected, record


def _value(record: Mapping[str, object] | object, field: str) -> str:
    if isinstance(record, Mapping):
        value = record.get(field, "")
    else:
        value = getattr(record, field, "")
    return str(value or "")


def _canonical_smiles(value: str, *, context: str) -> str:
    try:
        molecule = smiles_parser(value)
    except Exception as error:
        raise ValueError(f"{context}: invalid structure") from error
    if not isinstance(molecule, MoleculeContainer):
        raise ValueError(f"{context}: expected one molecule")
    return str(safe_canonicalization(molecule, clean_stereo=False))


def validate_deprotection_provenance(
    record: Mapping[str, object] | object,
    *,
    context: str,
    required: bool = False,
) -> ReactionContainer | None:
    """Validate an exact record and return its mapped transformation.

    A record with none of the exact fields is a supported legacy record unless
    exact provenance is required. A partially populated record is always rejected.
    """
    values = {field: _value(record, field) for field in EXACT_DEPROTECTION_FIELDS}
    if not any(values.values()):
        if required:
            raise ValueError(f"{context}: exact deprotection provenance is required")
        return None
    missing = [field for field, value in values.items() if not value]
    if missing:
        raise ValueError(
            f"{context}: incomplete exact deprotection provenance; missing {missing}"
        )
    policy = values["deprotection_policy"]
    if policy not in {"conservative", "aggressive"}:
        raise ValueError(f"{context}: invalid deprotection_policy {policy!r}")
    digest = values["protective_rules_sha256"]
    try:
        valid_digest = len(digest) == 64 and int(digest, 16) >= 0
    except ValueError:
        valid_digest = False
    if not valid_digest:
        raise ValueError(f"{context}: invalid protective_rules_sha256")
    try:
        events: Any = json.loads(values["deprotection_events"])
    except json.JSONDecodeError as error:
        raise ValueError(f"{context}: invalid deprotection_events JSON") from error
    if not isinstance(events, list) or not events:
        raise ValueError(f"{context}: deprotection_events must be a non-empty list")
    for index, event in enumerate(events):
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("pass_index"), int)
            or not isinstance(event.get("rule_name"), str)
            or not event["rule_name"]
            or not isinstance(event.get("query_mapping"), list)
            or not event["query_mapping"]
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(number, int) for number in pair)
                for pair in event["query_mapping"]
            )
        ):
            raise ValueError(f"{context}: invalid deprotection event at index {index}")
    try:
        reaction = smiles_parser(values["mapped_deprotection"])
    except Exception as error:
        raise ValueError(f"{context}: invalid mapped_deprotection") from error
    if (
        not isinstance(reaction, ReactionContainer)
        or len(reaction.reactants) != 1
        or len(reaction.products) != 1
    ):
        raise ValueError(
            f"{context}: mapped_deprotection must contain one reactant and one product"
        )
    protected = str(safe_canonicalization(reaction.reactants[0], clean_stereo=False))
    deprotected = str(safe_canonicalization(reaction.products[0], clean_stereo=False))
    if protected != _canonical_smiles(
        values["standardized_input_smiles"], context=context
    ):
        raise ValueError(
            f"{context}: mapped deprotection reactant does not match "
            "standardized_input_smiles"
        )
    canonical_smiles = _value(record, "canonical_smiles")
    if not canonical_smiles:
        raise ValueError(f"{context}: exact provenance requires canonical_smiles")
    if deprotected != _canonical_smiles(canonical_smiles, context=context):
        raise ValueError(
            f"{context}: mapped deprotection product does not match canonical_smiles"
        )
    return reaction


__all__ = [
    "EXACT_DEPROTECTION_FIELDS",
    "current_protective_rules_sha256",
    "deprotect_molecule_with_provenance",
    "validate_deprotection_provenance",
]
