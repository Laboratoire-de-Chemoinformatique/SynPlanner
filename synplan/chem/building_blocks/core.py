"""The public building-block record and immutable runtime catalogue."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from frozendict import frozendict


@dataclass(frozen=True, slots=True)
class BuildingBlock:
    """One material, optional prices, and immutable supplier provenance."""

    smiles: str
    inchikey: str
    vendors: frozendict[str, float]
    has_stereo: bool
    sources: tuple[frozendict[str, str], ...] = ()
    stereo_type: str = ""


BuildingBlockCatalogue: TypeAlias = Mapping[str, tuple[BuildingBlock, ...]]


def match_building_blocks(
    catalogue: BuildingBlockCatalogue,
    inchikey: str,
) -> tuple[BuildingBlock, ...]:
    """Return all catalogue records sharing an InChIKey connectivity block.

    This is candidate retrieval only. Search uses ``compatible_records`` to
    check molecular identity and every specified stereo requirement before
    choosing a full record from this bucket.
    """

    return catalogue.get(inchikey[:14], ())


__all__ = [
    "BuildingBlock",
    "BuildingBlockCatalogue",
    "match_building_blocks",
]
