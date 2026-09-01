"""Strict TSV preparation and streaming JSON loading for building blocks."""

from __future__ import annotations

import csv
import functools
import json
import math
import os
import tempfile
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import ijson
from chython import smiles as smiles_parser
from chython.containers import MoleculeContainer
from frozendict import frozendict

from synplan.chem.utils import safe_canonicalization

from .identity import (
    molecule_has_stereo,
    molecule_to_inchikey,
    validate_standard_inchikey,
)
from .model import (
    BuildingBlock,
    BuildingBlockCandidateIndex,
    BuildingBlocksByInchiKey,
)


def _write_tsv_atomic(path: Path, rows: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("line_number", "error"))
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(records, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def standardize_building_block_catalogue(
    input_file: str | Path, output_file: str | Path
) -> str:
    """Convert a vendor-price TSV into a stereo-preserving Chython JSON catalogue.

    The output is replaced atomically only when every row is valid. All row-level
    failures are collected in ``<output>.errors.tsv``.
    """

    source = Path(input_file)
    output = Path(output_file)
    if source.resolve() == output.resolve():
        raise ValueError("input_file name and output_file name cannot be the same.")
    if output.suffix.lower() != ".json":
        raise ValueError("building-block catalogue output must use the .json extension")

    records: dict[str, dict[str, Any]] = {}
    errors: list[tuple[int, str]] = []
    try:
        stream = source.open(encoding="utf-8", newline="")
    except OSError as error:
        raise ValueError(f"Could not read building-block TSV {source}") from error

    with stream:
        reader = csv.DictReader(stream, delimiter="\t")
        header = tuple(reader.fieldnames or ())
        smiles_columns = [name for name in header if name.casefold() == "smiles"]
        price_columns = [name for name in header if name.casefold().endswith("_ppg")]
        if len(smiles_columns) != 1:
            raise ValueError(f"{source}: expected exactly one SMILES column")
        if not price_columns:
            raise ValueError(f"{source}: expected at least one *_ppg vendor column")
        smiles_column = smiles_columns[0]

        for line_number, row in enumerate(reader, start=2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("row does not match the header column count")
                raw_smiles = row[smiles_column].strip()
                if not raw_smiles:
                    raise ValueError("SMILES is empty")

                vendors: dict[str, float] = {}
                for column in price_columns:
                    raw_price = row[column].strip()
                    if not raw_price:
                        continue
                    try:
                        price = float(raw_price)
                    except ValueError as error:
                        raise ValueError(
                            f"{column}: price {raw_price!r} is not numeric"
                        ) from error
                    if not math.isfinite(price) or price < 0.0:
                        raise ValueError(
                            f"{column}: price {raw_price!r} must be finite and non-negative"
                        )
                    if price > 0.0:
                        vendors[column[: -len("_ppg")]] = price

                molecule = smiles_parser(
                    raw_smiles,
                    ignore=True,
                    ignore_stereo=False,
                )
                if not isinstance(molecule, MoleculeContainer):
                    raise ValueError("SMILES does not describe one molecule")
                molecule = safe_canonicalization(molecule, clean_stereo=False)
                canonical_smiles = str(molecule)
                key = molecule_to_inchikey(molecule)

                existing = records.get(key)
                if existing is None:
                    records[key] = {
                        "smiles": canonical_smiles,
                        "vendors": vendors,
                        "has_stereo": molecule_has_stereo(molecule),
                    }
                else:
                    # Standard InChI deliberately merges some tautomeric spellings.
                    # Keep the first canonical SMILES and merge only vendor offers.
                    existing["has_stereo"] = bool(
                        existing["has_stereo"] or molecule_has_stereo(molecule)
                    )
                    existing_vendors = existing["vendors"]
                    for vendor, price in vendors.items():
                        previous = existing_vendors.get(vendor)
                        if previous is None or price < previous:
                            existing_vendors[vendor] = price
            except Exception as error:
                errors.append((line_number, str(error) or type(error).__name__))

    error_path = Path(f"{output}.errors.tsv")
    if not records and not errors:
        errors.append((1, "catalogue contains no data rows"))
    if errors:
        _write_tsv_atomic(error_path, errors)
        raise ValueError(
            f"{source}: {len(errors)} invalid row(s); details written to {error_path}"
        )

    _write_json_atomic(output, records)
    error_path.unlink(missing_ok=True)
    return str(output)


@functools.cache
def load_building_block_indexes(
    path_value: str | Path,
) -> tuple[BuildingBlocksByInchiKey, BuildingBlockCandidateIndex]:
    """Stream a prepared JSON catalogue and build its two immutable indexes."""

    path = Path(path_value).resolve(strict=True)
    building_blocks_by_inchikey: dict[str, BuildingBlock] = {}
    candidate_groups: dict[str, list[BuildingBlock]] = defaultdict(list)
    try:
        with path.open("rb") as stream:
            for key, raw_record in ijson.kvitems(stream, ""):
                location = f"{path}:{key}"
                key = validate_standard_inchikey(key, context=str(path))
                if key in building_blocks_by_inchikey:
                    raise ValueError(f"{location}: duplicate InChIKey")
                if not isinstance(raw_record, dict):
                    raise ValueError(f"{location}: record must be a JSON object")
                if set(raw_record) != {"smiles", "vendors", "has_stereo"}:
                    raise ValueError(
                        f"{location}: expected smiles, vendors, and has_stereo fields"
                    )
                canonical_smiles = raw_record["smiles"]
                if not isinstance(canonical_smiles, str) or not canonical_smiles:
                    raise ValueError(f"{location}: smiles must be a non-empty string")
                has_stereo = raw_record["has_stereo"]
                if not isinstance(has_stereo, bool):
                    raise ValueError(f"{location}: has_stereo must be boolean")
                raw_vendors = raw_record["vendors"]
                if not isinstance(raw_vendors, dict):
                    raise ValueError(f"{location}: vendors must be a JSON object")
                vendors: dict[str, float] = {}
                for vendor, raw_price in raw_vendors.items():
                    if not isinstance(vendor, str) or not vendor:
                        raise ValueError(
                            f"{location}: vendor names must be non-empty strings"
                        )
                    if isinstance(raw_price, bool) or not isinstance(
                        raw_price, (int, float, Decimal)
                    ):
                        raise ValueError(f"{location}: {vendor} price must be numeric")
                    price = float(raw_price)
                    if not math.isfinite(price) or price <= 0.0:
                        raise ValueError(
                            f"{location}: {vendor} price must be finite and positive"
                        )
                    vendors[vendor] = price
                block = BuildingBlock(
                    smiles=canonical_smiles,
                    inchikey=key,
                    vendors=frozendict(vendors),
                    has_stereo=has_stereo,
                )
                building_blocks_by_inchikey[key] = block
                candidate_groups[key[:14]].append(block)
    except (OSError, ijson.JSONError) as error:
        raise ValueError(
            f"Could not read building-block catalogue {path}: {error}"
        ) from error

    if not building_blocks_by_inchikey:
        raise ValueError(f"{path}: building-block catalogue is empty")
    return frozendict(building_blocks_by_inchikey), frozendict(
        (prefix, tuple(blocks)) for prefix, blocks in candidate_groups.items()
    )


__all__ = [
    "load_building_block_indexes",
    "standardize_building_block_catalogue",
]
