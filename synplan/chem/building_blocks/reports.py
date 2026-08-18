"""Structured report contracts for building-block preparation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IdentityReportRow:
    """Identity and provenance for one candidate emitted by preparation."""

    source_index: int
    input_smiles: str
    canonical_smiles: str
    standard_inchi: str
    standard_inchikey: str
    inchi_return_code: int | str
    inchi_warnings: str
    output_origin: str
    status: str
    note: str = ""
    standardized_input_smiles: str = ""
    deprotection_policy: str = ""
    protective_rules_sha256: str = ""
    deprotection_events: str = ""
    mapped_deprotection: str = ""


@dataclass(frozen=True, slots=True)
class DuplicateReportRow:
    """One input or deprotection candidate collapsed by canonical identity."""

    duplicate_kind: str
    canonical_smiles: str
    first_source_index: int
    duplicate_source_index: int
    first_origin: str
    duplicate_origin: str


@dataclass(frozen=True, slots=True)
class CollisionReportRow:
    """Distinct canonical structures sharing one full Standard InChIKey."""

    standard_inchikey: str
    canonical_smiles: str
    source_indexes: str
    source_info: str
    output_origins: str


@dataclass(frozen=True, slots=True)
class StereoReportRow:
    """Stereo audit for a standardized or deprotected candidate."""

    source_index: int
    input_smiles: str
    canonical_smiles: str
    stereo_free_smiles: str
    stereo_present: bool
    output_origin: str
    status: str
    note: str = ""


IDENTITY_FIELDS = tuple(IdentityReportRow.__dataclass_fields__)
DUPLICATE_FIELDS = tuple(DuplicateReportRow.__dataclass_fields__)
COLLISION_FIELDS = tuple(CollisionReportRow.__dataclass_fields__)
STEREO_FIELDS = tuple(StereoReportRow.__dataclass_fields__)


__all__ = [
    "COLLISION_FIELDS",
    "DUPLICATE_FIELDS",
    "IDENTITY_FIELDS",
    "STEREO_FIELDS",
    "CollisionReportRow",
    "DuplicateReportRow",
    "IdentityReportRow",
    "StereoReportRow",
]
