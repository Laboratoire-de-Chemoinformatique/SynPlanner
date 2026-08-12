"""Typed ordinary building-block stocks used by synthesis planning."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from chython.containers import MoleculeContainer

from .identity import (
    canonical_molecule_smiles,
    inchi_to_inchi_key,
    molecule_to_inchi_key,
    validate_standard_inchi_key,
)

StockIdentityFormat = Literal["smiles", "inchikey"]


def _validate_format(identity_format: str) -> StockIdentityFormat:
    if identity_format not in {"smiles", "inchikey"}:
        raise ValueError("identity_format must be 'smiles' or 'inchikey'")
    return identity_format  # type: ignore[return-value]


def _is_standard_inchi_key(value: str) -> bool:
    try:
        validate_standard_inchi_key(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class BuildingBlockStock:
    """Immutable stock keys together with their lookup representation."""

    keys: frozenset[str]
    identity_format: StockIdentityFormat = "smiles"

    def __post_init__(self) -> None:
        identity_format = _validate_format(self.identity_format)
        try:
            keys = frozenset(self.keys)
        except TypeError as error:
            raise TypeError("building-block stock keys must be iterable strings") from error
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("building-block stock keys must be non-empty strings")
        if identity_format == "inchikey":
            for key in keys:
                validate_standard_inchi_key(key)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "identity_format", identity_format)

    def key_for_molecule(self, molecule: MoleculeContainer) -> str:
        """Generate the stock lookup key for ``molecule``."""
        if self.identity_format == "inchikey":
            return molecule_to_inchi_key(molecule)
        return canonical_molecule_smiles(molecule)

    def contains_molecule(self, molecule: MoleculeContainer) -> bool:
        """Return whether ``molecule`` is explicitly present in this stock."""
        return self.key_for_molecule(molecule) in self.keys

    def without_molecule(self, molecule: MoleculeContainer) -> BuildingBlockStock:
        """Return a stock with the key for ``molecule`` removed."""
        key = self.key_for_molecule(molecule)
        return BuildingBlockStock(self.keys.difference({key}), self.identity_format)

    def __contains__(self, key: object) -> bool:
        return key in self.keys

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys)

    def __len__(self) -> int:
        return len(self.keys)


def coerce_building_block_stock(
    stock: BuildingBlockStock | Iterable[str],
    identity_format: StockIdentityFormat | None = None,
) -> BuildingBlockStock:
    """Coerce a typed or legacy stock, migrating homogeneous legacy InChIs.

    Legacy raw-InChI and full-InChIKey collections are detected only when the
    complete collection is homogeneous.  Mixed identity representations are rejected.
    Other legacy string collections retain the historic canonical-SMILES meaning.
    """
    if identity_format is not None:
        identity_format = _validate_format(identity_format)
    if isinstance(stock, BuildingBlockStock):
        if identity_format is not None and identity_format != stock.identity_format:
            raise ValueError(
                "explicit identity_format conflicts with the typed building-block stock"
            )
        return stock

    values = frozenset(stock)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("legacy building-block stock keys must be non-empty strings")
    if not values:
        return BuildingBlockStock(values, identity_format or "smiles")

    raw_inchi = {value for value in values if value.startswith("InChI=")}
    valid_inchikey = {value for value in values if _is_standard_inchi_key(value)}

    if raw_inchi and len(raw_inchi) != len(values):
        raise ValueError("legacy stock mixes raw InChI with another identity format")
    if valid_inchikey and len(valid_inchikey) != len(values):
        raise ValueError("legacy stock mixes InChIKey with another identity format")
    if raw_inchi:
        if identity_format == "smiles":
            raise ValueError("raw InChI stock conflicts with identity_format='smiles'")
        return BuildingBlockStock(
            frozenset(inchi_to_inchi_key(value) for value in values), "inchikey"
        )
    if valid_inchikey:
        if identity_format == "smiles":
            raise ValueError("InChIKey stock conflicts with identity_format='smiles'")
        return BuildingBlockStock(values, "inchikey")
    if identity_format == "inchikey":
        raise ValueError("identity_format='inchikey' requires Standard InChIKeys")
    return BuildingBlockStock(values, "smiles")


__all__ = [
    "BuildingBlockStock",
    "StockIdentityFormat",
    "coerce_building_block_stock",
]
