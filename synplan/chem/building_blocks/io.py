"""Strict TSV preparation and streaming JSON loading for building blocks."""

from __future__ import annotations

import csv
import functools
import gzip
import json
import logging
import math
import os
import tempfile
from contextlib import contextmanager
from decimal import Decimal
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import ijson
from chython.containers import MoleculeContainer
from frozendict import frozendict

from synplan.chem.stereo import parse_smiles_preserving_stereo
from synplan.chem.utils import safe_canonicalization
from synplan.utils.parallel import chunked, process_pool_map_stream

from .core import BuildingBlock
from .database import load_building_block_catalogue
from .identity import (
    molecule_has_stereo,
    molecule_to_inchikey,
    validate_standard_inchikey,
)

logger = logging.getLogger(__name__)


@contextmanager
def _atomic_output(path: Path):
    """Publish a closed file on success; discard temporary files on every exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=path.parent, prefix=f".{path.name}."
    ) as folder:
        temporary = Path(folder) / path.name
        temporary.touch(mode=0o600)
        yield temporary
        os.replace(temporary, path)


def _write_tsv_atomic(path: Path, rows: list[tuple[int, str]]) -> None:
    with (
        _atomic_output(path) as temporary,
        temporary.open("w", encoding="utf-8", newline="") as handle,
    ):
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("line_number", "error"))
        writer.writerows(rows)


def _write_json_atomic(path: Path, records: dict[str, dict[str, Any]]) -> None:
    with _atomic_output(path) as temporary, temporary.open("wb") as handle:
        stream = (
            gzip.GzipFile(
                fileobj=handle, mode="wb", filename="", mtime=0, compresslevel=6
            )
            if path.suffix.lower() == ".gz"
            else handle
        )
        with TextIOWrapper(stream, encoding="utf-8", newline="\n") as text:
            json.dump(records, text, ensure_ascii=False, separators=(",", ":"))
            text.write("\n")


def _merge_vendor_prices(existing: dict, incoming: dict) -> dict:
    for vendor, price in incoming.items():
        existing[vendor] = min(price, existing.get(vendor, price))
    return existing


def standardize_building_block_catalogue(
    input_file: str | Path, output_file: str | Path, *, num_workers: int = 1
) -> str:
    """Convert a vendor-price TSV into a stereo-preserving Chython JSON catalogue.

    Valid rows are published atomically even when other rows fail. All row-level
    failures are collected in ``<output>.errors.tsv``. If no row is valid, the
    report is written but an existing output is left untouched.
    ``num_workers`` parallelizes chemistry in ordered batches; loading the
    prepared JSON/JSON.GZ never repeats this work.
    """

    source = Path(input_file)
    output = Path(output_file)
    if source.resolve() == output.resolve():
        raise ValueError("input_file name and output_file name cannot be the same.")
    if not output.name.lower().endswith((".json", ".json.gz")):
        raise ValueError("building-block catalogue output must use .json or .json.gz")
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    # ponytail: JSON export holds records in RAM; stream from SQLite if large
    # JSON releases are needed. SQLite output already bounds preparation memory.
    records: dict[str, dict[str, Any]] = {}
    errors: list[tuple[int, str]] = []
    for batch_records, batch_errors in _prepare_catalogue_batches(source, num_workers):
        errors.extend(batch_errors)
        for key, record in batch_records:
            existing = records.get(key)
            if existing is None:
                records[key] = record
            else:
                # Standard InChI deliberately merges some tautomeric spellings.
                # Keep the first canonical SMILES and merge only vendor offers.
                existing["has_stereo"] = bool(
                    existing["has_stereo"] or record["has_stereo"]
                )
                _merge_vendor_prices(existing["vendors"], record["vendors"])

    error_path = Path(f"{output}.errors.tsv")
    if not records and not errors:
        errors.append((1, "catalogue contains no data rows"))
    if errors:
        _write_tsv_atomic(error_path, errors)
    if not records:
        raise ValueError(
            f"{source}: no valid rows; {len(errors)} invalid row(s) reported in "
            f"{error_path}"
        )

    _write_json_atomic(output, records)
    if errors:
        logger.warning(
            "Published %d building blocks to %s after dropping %d invalid row(s); "
            "details written to %s",
            len(records),
            output,
            len(errors),
            error_path,
        )
    else:
        error_path.unlink(missing_ok=True)
    return str(output)


def _prepare_catalogue_batches(source: Path, num_workers: int):
    try:
        opener = gzip.open if source.suffix.lower() == ".gz" else open
        stream = opener(source, "rt", encoding="utf-8", newline="")
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
        worker = functools.partial(
            _prepare_catalogue_batch,
            smiles_column=smiles_columns[0],
            price_columns=price_columns,
        )
        batches = chunked(enumerate(reader, start=2), 1000)
        results = (
            map(worker, batches)
            if num_workers == 1
            else process_pool_map_stream(
                batches, worker, max_workers=num_workers, ordered=True, timeout=0
            )
        )
        yield from results


def _prepare_catalogue_batch(rows, *, smiles_column, price_columns):
    records = []
    errors = []
    for line_number, row in rows:
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

            molecule = parse_smiles_preserving_stereo(raw_smiles)
            if not isinstance(molecule, MoleculeContainer):
                raise ValueError("SMILES does not describe one molecule")
            molecule = safe_canonicalization(molecule, clean_stereo=False)
            records.append(
                (
                    molecule_to_inchikey(molecule),
                    {
                        "smiles": str(molecule),
                        "vendors": vendors,
                        "has_stereo": molecule_has_stereo(molecule),
                    },
                )
            )
        except Exception as error:
            errors.append((line_number, str(error) or type(error).__name__))
    return records, errors


def _validate_record(key, raw_record, *, context):
    """Validate stored fields without repeating molecular parsing or identity work."""
    key = validate_standard_inchikey(key, context=context)
    location = f"{context}:{key}"
    if not isinstance(raw_record, dict):
        raise ValueError(f"{location}: record must be a JSON object")
    if set(raw_record) != {"smiles", "vendors", "has_stereo"}:
        raise ValueError(f"{location}: expected smiles, vendors, and has_stereo fields")
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
            raise ValueError(f"{location}: vendor names must be non-empty strings")
        if isinstance(raw_price, bool) or not isinstance(
            raw_price, (int, float, Decimal)
        ):
            raise ValueError(f"{location}: {vendor} price must be numeric")
        price = float(raw_price)
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError(f"{location}: {vendor} price must be finite and positive")
        vendors[vendor] = price
    return BuildingBlock(
        smiles=canonical_smiles,
        inchikey=key,
        vendors=frozendict(vendors),
        has_stereo=has_stereo,
    )


def _iter_prepared_catalogue(path: Path):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rb") as stream:
            yield from ijson.kvitems(stream, "")
    except (OSError, EOFError, ijson.JSONError) as error:
        raise ValueError(
            f"Could not read building-block catalogue {path}: {error}"
        ) from error


__all__ = ["load_building_block_catalogue", "standardize_building_block_catalogue"]
