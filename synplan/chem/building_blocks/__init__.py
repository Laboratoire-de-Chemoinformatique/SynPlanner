"""Vendor-aware ordinary building blocks and their immutable runtime catalogue."""

from chython import inchi_key as molecule_to_inchikey

from synplan.chem.stereo import has_stereo as molecule_has_stereo

from .core import (
    BuildingBlock,
    BuildingBlockCatalogue,
    match_building_blocks,
)
from .database import SQLiteBuildingBlockCatalogue
from .io import (
    load_building_block_catalogue,
    load_building_blocks,
    standardize_building_block_catalogue,
    standardize_building_blocks,
)

__all__ = [
    "BuildingBlock",
    "BuildingBlockCatalogue",
    "SQLiteBuildingBlockCatalogue",
    "load_building_block_catalogue",
    "load_building_blocks",
    "match_building_blocks",
    "molecule_has_stereo",
    "molecule_to_inchikey",
    "standardize_building_block_catalogue",
    "standardize_building_blocks",
]
