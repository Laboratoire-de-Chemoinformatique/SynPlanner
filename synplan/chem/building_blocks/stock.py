"""Typed ordinary building-block stocks used by synthesis planning."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chython import smiles as smiles_parser
from chython.containers import MoleculeContainer

from synplan.chem.utils import safe_canonicalization
from synplan.utils.files import (
    ChemicalRecord,
    iter_chemical_records,
    open_text,
    resolve_chemical_input_format,
)

from .config import BuildingBlockStockInputFormat, BuildingBlockStockLoadConfig
from .identity import (
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
            raise TypeError(
                "building-block stock keys must be iterable strings"
            ) from error
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("building-block stock keys must be non-empty strings")
        if identity_format == "smiles":
            normalized: set[str] = set()
            for key in keys:
                try:
                    molecule = smiles_parser(key, ignore=True)
                except Exception as error:
                    raise ValueError(
                        f"invalid SMILES building-block stock key: {key!r}: {error}"
                    ) from error
                if not isinstance(molecule, MoleculeContainer):
                    raise ValueError(
                        f"invalid SMILES building-block stock key: {key!r}"
                    )
                normalized.add(str(safe_canonicalization(molecule, clean_stereo=False)))
            keys = frozenset(normalized)
        if identity_format == "inchikey":
            for key in keys:
                validate_standard_inchi_key(key)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "identity_format", identity_format)

    @classmethod
    def _from_validated_keys(
        cls, keys: frozenset[str], identity_format: StockIdentityFormat
    ) -> BuildingBlockStock:
        """Build from keys validated by a trusted loader or derived operation.

        General callers and legacy coercion use the public constructor so SMILES keys
        cannot bypass canonicalization.
        """
        stock = object.__new__(cls)
        object.__setattr__(stock, "keys", keys)
        object.__setattr__(stock, "identity_format", _validate_format(identity_format))
        return stock

    def key_for_molecule(self, molecule: MoleculeContainer) -> str:
        """Generate the stock lookup key for ``molecule``."""
        if self.identity_format == "inchikey":
            return molecule_to_inchi_key(molecule)
        return str(safe_canonicalization(molecule, clean_stereo=False))

    def contains_molecule(self, molecule: MoleculeContainer) -> bool:
        """Return whether ``molecule`` is explicitly present in this stock."""
        return self.key_for_molecule(molecule) in self.keys

    def without_molecule(self, molecule: MoleculeContainer) -> BuildingBlockStock:
        """Return a stock with the key for ``molecule`` removed."""
        key = self.key_for_molecule(molecule)
        return self._from_validated_keys(
            self.keys.difference({key}), self.identity_format
        )

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
    """Coerce a typed or legacy canonical-SMILES or full-InChIKey stock.

    Full-InChIKey collections are detected only when the complete collection is
    homogeneous. Raw InChI is deliberately not a stock representation.
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

    if raw_inchi:
        raise ValueError(
            "raw InChI building-block stocks are unsupported; "
            "use canonical SMILES or full Standard InChIKeys"
        )
    if valid_inchikey and len(valid_inchikey) != len(values):
        raise ValueError("legacy stock mixes InChIKey with another identity format")
    if valid_inchikey:
        if identity_format == "smiles":
            raise ValueError("InChIKey stock conflicts with identity_format='smiles'")
        return BuildingBlockStock(values, "inchikey")
    if identity_format == "inchikey":
        raise ValueError("identity_format='inchikey' requires Standard InChIKeys")
    return BuildingBlockStock(values, "smiles")


BuildingBlocksFormat = BuildingBlockStockInputFormat
ResolvedBuildingBlocksFormat = Literal["smiles", "inchikey"]

_STANDARD_TABLE_COLUMNS = {
    "smiles": "smiles",
    "cxsmiles": "smiles",
    "inchikey": "inchikey",
}


def _record_location(path: Path, record: ChemicalRecord) -> str:
    if record.input_format == "sdf":
        return f"{path}: SDF record {record.sequence}"
    return f"{path}:{record.line_number}"


def _validated_record_key(
    path: Path, record: ChemicalRecord
) -> tuple[ResolvedBuildingBlocksFormat, str]:
    location = _record_location(path, record)
    if record.format_error:
        raise ValueError(f"{location}: {record.format_error}")
    if record.molecule is not None:
        return "smiles", str(safe_canonicalization(record.molecule, clean_stereo=False))

    value = record.chemistry
    if value.startswith("InChI="):
        raise ValueError(
            f"{location}: raw InChI stock input is unsupported; "
            "use canonical SMILES or a full Standard InChIKey"
        )

    try:
        return "inchikey", validate_standard_inchi_key(value)
    except ValueError:
        pass

    try:
        molecule = smiles_parser(value, ignore=True)
    except Exception as error:
        raise ValueError(f"{location}: invalid SMILES: {error}") from error
    if not isinstance(molecule, MoleculeContainer):
        raise ValueError(f"{location}: invalid SMILES: {value!r}")
    try:
        key = str(safe_canonicalization(molecule, clean_stereo=False))
    except Exception as error:
        raise ValueError(f"{location}: cannot canonicalize SMILES: {error}") from error
    return "smiles", key


def _read_validated_inchikey_stock(path: Path) -> frozenset[str]:
    """Read a plain InChIKey stock without generic chemistry-record objects."""
    keys: set[str] = set()
    count = 0
    add_key = keys.add
    validate_key = validate_standard_inchi_key
    with open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            key = line.strip()
            if not key or key.startswith("#"):
                continue
            count += 1
            if key.startswith("InChI="):
                raise ValueError(
                    f"{path}:{line_number}: raw InChI stock input is unsupported; "
                    "use canonical SMILES or a full Standard InChIKey"
                )
            try:
                add_key(validate_key(key))
            except ValueError as error:
                if any(character.isspace() for character in key):
                    raise ValueError(
                        f"{path}:{line_number}: identity rows cannot contain whitespace"
                    ) from error
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not count:
        raise ValueError(f"{path}: building-block stock is empty")
    return frozenset(keys)


def _read_validated_stock(
    path: Path, config: BuildingBlockStockLoadConfig
) -> tuple[ResolvedBuildingBlocksFormat, frozenset[str]]:
    file_format = resolve_chemical_input_format(path)
    if file_format == "inchikey":
        if config.identity_format == "smiles":
            raise ValueError(
                f"{path}: requested 'smiles' input but detected 'inchikey'"
            )
        return "inchikey", _read_validated_inchikey_stock(path)

    records = iter_chemical_records(
        path,
        input_format=file_format,
        chemistry_columns=_STANDARD_TABLE_COLUMNS,
        chemistry_column=config.chemistry_column,
        delimiter=config.delimiter,
        skip_comments=True,
    )
    observed: set[ResolvedBuildingBlocksFormat] = set()
    declared: set[str] = set()
    keys: set[str] = set()
    count = 0
    for record in records:
        count += 1
        identity_format, key = _validated_record_key(path, record)
        observed.add(identity_format)
        declared.add(record.chemistry_format)
        keys.add(key)
    if not count:
        raise ValueError(f"{path}: building-block stock is empty")
    if len(observed) != 1:
        raise ValueError(
            f"{path}: mixed building-block identity formats: {sorted(observed)!r}"
        )
    detected = observed.pop()
    if file_format in {"csv", "tsv"} and declared != {detected}:
        raise ValueError(
            f"{path}: header declares {sorted(declared)!r} but rows contain "
            f"{detected!r}"
        )
    extension_format = "inchikey" if file_format == "inchikey" else None
    if extension_format is not None and detected != extension_format:
        raise ValueError(f"{path}: .{file_format} file contains {detected!r} records")
    if config.identity_format != "auto" and detected != config.identity_format:
        raise ValueError(
            f"{path}: requested {config.identity_format!r} input but detected "
            f"{detected!r}"
        )
    return detected, frozenset(keys)


def detect_building_blocks_format(
    building_blocks_path: str | Path,
    *,
    config: BuildingBlockStockLoadConfig | None = None,
) -> ResolvedBuildingBlocksFormat:
    """Resolve and fully validate a building-block file identity representation."""
    path = Path(building_blocks_path).resolve(strict=True)
    detected, _keys = _read_validated_stock(
        path, config or BuildingBlockStockLoadConfig()
    )
    return detected


def load_building_block_stock(
    building_blocks_path: str | Path,
    *,
    config: BuildingBlockStockLoadConfig | None = None,
) -> BuildingBlockStock:
    """Load a validated canonical-SMILES or full-InChIKey stock.

    Each chemistry record is parsed exactly once. Typed SMILES stocks always own
    stereo-preserving canonical keys. File-decoding policy is supplied explicitly
    through a domain-owned load configuration.
    """
    path = Path(building_blocks_path).resolve(strict=True)
    source_format, keys = _read_validated_stock(
        path, config or BuildingBlockStockLoadConfig()
    )
    return BuildingBlockStock._from_validated_keys(keys, source_format)


__all__ = [
    "BuildingBlockStock",
    "BuildingBlocksFormat",
    "ResolvedBuildingBlocksFormat",
    "StockIdentityFormat",
    "coerce_building_block_stock",
    "detect_building_blocks_format",
    "load_building_block_stock",
]
