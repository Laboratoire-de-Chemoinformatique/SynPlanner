"""The public building-block record and immutable runtime catalogue."""

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


BuildingBlockCatalogue: TypeAlias = frozendict[str, tuple[BuildingBlock, ...]]


def match_building_blocks(
    catalogue: BuildingBlockCatalogue,
    inchikey: str,
) -> tuple[BuildingBlock, ...]:
    """Return all catalogue records sharing an InChIKey connectivity block.

    Search is intentionally stereo-agnostic and therefore uses only the first
    14 characters. Complete keys and stereo metadata remain on each record for
    future use.
    """

    return catalogue.get(inchikey[:14], ())


__all__ = [
    "BuildingBlock",
    "BuildingBlockCatalogue",
    "match_building_blocks",
]
