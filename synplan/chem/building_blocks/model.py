"""The public building-block record and immutable index types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from frozendict import frozendict


@dataclass(frozen=True, slots=True)
class BuildingBlock:
    """One purchasable structure and its positive vendor price-per-gram offers."""

    smiles: str
    inchikey: str
    vendors: frozendict[str, float]
    has_stereo: bool


BuildingBlocksByInchiKey: TypeAlias = frozendict[str, BuildingBlock]
BuildingBlockCandidateIndex: TypeAlias = frozendict[str, tuple[BuildingBlock, ...]]


__all__ = [
    "BuildingBlock",
    "BuildingBlockCandidateIndex",
    "BuildingBlocksByInchiKey",
]
