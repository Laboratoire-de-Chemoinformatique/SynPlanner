"""Unified identity, provenance, and optional pricing for building blocks."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from chython import smiles as smiles_parser
from chython.containers import MoleculeContainer

from synplan.chem.utils import safe_canonicalization

from .identity import molecule_to_inchi_key
from .provenance import (
    EXACT_DEPROTECTION_FIELDS,
    validate_deprotection_provenance,
)
from .reports import IdentityReportRow
from .stock import BuildingBlockStock, StockIdentityFormat

_IDENTITY_COLUMNS = {
    "source_index",
    "input_smiles",
    "canonical_smiles",
    "standard_inchikey",
    "output_origin",
    "status",
}

_STEREO_COLUMNS = {
    "source_index",
    "canonical_smiles",
    "stereo_free_smiles",
    "output_origin",
    "status",
}


class BuildingBlockCatalog:
    """Reusable in-memory view over prepared BB identity and optional prices.

    Identity and price files remain separate physical artifacts because the
    planner stock is deduplicated while provenance and vendor offers are
    one-to-many. Prepared price rows explicitly carry ``source_index`` and
    ``input_smiles``; both are validated during the logical join, and row order
    has no meaning.
    """

    def __init__(
        self,
        records: tuple[IdentityReportRow, ...],
        identity_format: StockIdentityFormat = "smiles",
        prices_by_source: Mapping[int, Mapping[str, float | None]] | None = None,
        stereo_file: str | Path | None = None,
    ) -> None:
        self.records = records
        self.identity_format = identity_format
        self.prices_by_source = {
            source: dict(prices) for source, prices in (prices_by_source or {}).items()
        }
        self.stereo_file = Path(stereo_file) if stereo_file is not None else None
        self._stereo_candidate_cache: dict[str, tuple[str, ...]] = {}
        self._by_stereo_free_smiles: dict[str, tuple[IdentityReportRow, ...]] | None = (
            None
        )
        by_smiles: dict[str, list[IdentityReportRow]] = defaultdict(list)
        by_inchikey: dict[str, list[IdentityReportRow]] = defaultdict(list)
        for record in records:
            if record.status not in {"written", "duplicate_skipped"}:
                continue
            by_smiles[record.canonical_smiles].append(record)
            if record.standard_inchikey:
                by_inchikey[record.standard_inchikey].append(record)
        self.by_canonical_smiles = {
            key: tuple(items) for key, items in by_smiles.items()
        }
        self.by_inchikey = {key: tuple(items) for key, items in by_inchikey.items()}
        self.keys = frozenset(
            self.by_canonical_smiles
            if identity_format == "smiles"
            else self.by_inchikey
        )

    @classmethod
    def from_files(
        cls,
        identity_file: str | Path,
        price_file: str | Path | None = None,
        *,
        identity_format: StockIdentityFormat = "smiles",
        stereo_file: str | Path | None = None,
    ) -> BuildingBlockCatalog:
        """Load prepared identity, price, and optional stereo artifacts.

        A sibling ending in ``_stereo.tsv`` is discovered automatically for
        the standard ``_identity.tsv`` filename. It is consulted lazily only
        when an exact stereo identity does not match.
        """
        path = Path(identity_file)
        resolved_stereo_file = cls._resolve_stereo_file(path, stereo_file)
        records: list[IdentityReportRow] = []
        try:
            with path.open(newline="") as stream:
                reader = csv.reader(stream, delimiter="\t")
                header = next(reader, ())
                missing = _IDENTITY_COLUMNS.difference(header)
                if missing:
                    raise ValueError(
                        f"{path}: missing identity columns: {sorted(missing)}"
                    )
                if len(header) != len(set(header)):
                    raise ValueError(f"{path}: duplicate identity columns")
                columns = {name: index for index, name in enumerate(header)}
                validates_exact_provenance = any(
                    field in columns for field in EXACT_DEPROTECTION_FIELDS
                )

                def value(row: list[str], name: str) -> str:
                    index = columns.get(name)
                    return row[index].strip() if index is not None else ""

                for line_number, row in enumerate(reader, start=2):
                    if len(row) != len(header):
                        raise ValueError(
                            f"{path}:{line_number}: expected {len(header)} "
                            f"columns, got {len(row)}"
                        )
                    try:
                        source_index = int(value(row, "source_index"))
                    except ValueError as error:
                        raise ValueError(
                            f"{path}:{line_number}: invalid source_index"
                        ) from error
                    record = IdentityReportRow(
                        source_index=source_index,
                        input_smiles=value(row, "input_smiles"),
                        canonical_smiles=value(row, "canonical_smiles"),
                        standard_inchi=value(row, "standard_inchi"),
                        standard_inchikey=value(row, "standard_inchikey"),
                        inchi_return_code=value(row, "inchi_return_code"),
                        inchi_warnings=value(row, "inchi_warnings"),
                        output_origin=value(row, "output_origin"),
                        status=value(row, "status"),
                        note=value(row, "note"),
                        standardized_input_smiles=value(
                            row, "standardized_input_smiles"
                        ),
                        deprotection_policy=value(row, "deprotection_policy"),
                        protective_rules_sha256=value(row, "protective_rules_sha256"),
                        deprotection_events=value(row, "deprotection_events"),
                        mapped_deprotection=value(row, "mapped_deprotection"),
                    )
                    if (
                        validates_exact_provenance
                        and record.output_origin == "deprotected"
                    ):
                        validate_deprotection_provenance(
                            record, context=f"{path}:{line_number}"
                        )
                    records.append(record)
        except OSError as error:
            raise ValueError(f"Could not read BB identity file {path}") from error
        prices: dict[int, dict[str, float | None]] = {}
        if price_file is not None:
            prices, price_inputs = cls._load_prices(price_file)
            cls._validate_price_join(
                path, records, Path(price_file), prices, price_inputs
            )
        return cls(
            tuple(records),
            identity_format=identity_format,
            prices_by_source=prices,
            stereo_file=resolved_stereo_file,
        )

    @staticmethod
    def _resolve_stereo_file(
        identity_path: Path,
        stereo_file: str | Path | None,
    ) -> Path | None:
        if stereo_file is not None:
            path = Path(stereo_file)
            if not path.is_file():
                raise ValueError(f"Could not read BB stereo file {path}")
            return path
        suffix = "_identity.tsv"
        if not identity_path.name.endswith(suffix):
            return None
        candidate = identity_path.with_name(
            f"{identity_path.name[: -len(suffix)]}_stereo.tsv"
        )
        return candidate if candidate.is_file() else None

    @staticmethod
    def _load_prices(
        path_value: str | Path,
    ) -> tuple[dict[int, dict[str, float | None]], dict[int, str]]:
        path = Path(path_value)
        result: dict[int, dict[str, float | None]] = {}
        inputs: dict[int, str] = {}
        try:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                fieldnames = tuple(reader.fieldnames or ())
                missing = {"source_index", "input_smiles"}.difference(fieldnames)
                if missing:
                    raise ValueError(
                        f"{path}: missing price columns: {sorted(missing)}"
                    )
                columns = tuple(
                    name
                    for name in (reader.fieldnames or ())
                    if name.casefold().endswith("_ppg")
                )
                if not columns:
                    raise ValueError(f"{path}: expected at least one *_ppg column")
                for line_number, row in enumerate(reader, start=2):
                    try:
                        source_index = int(row["source_index"])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"{path}:{line_number}: invalid source_index"
                        ) from error
                    if source_index in result:
                        raise ValueError(
                            f"{path}:{line_number}: duplicate source_index {source_index}"
                        )
                    input_smiles = row["input_smiles"].strip()
                    if not input_smiles:
                        raise ValueError(
                            f"{path}:{line_number}: empty input_smiles for "
                            f"source_index {source_index}"
                        )
                    parsed: dict[str, float | None] = {}
                    for column in columns:
                        raw = row[column].strip()
                        value = float(raw) if raw else None
                        if value is not None and (
                            not math.isfinite(value) or value < 0
                        ):
                            raise ValueError(
                                f"{path}:{line_number}: source_index {source_index}, column {column}: "
                                f"invalid price {raw!r}"
                            )
                        parsed[column] = value
                    result[source_index] = parsed
                    inputs[source_index] = input_smiles
        except OSError as error:
            raise ValueError(f"Could not read BB price file {path}") from error
        return result, inputs

    @staticmethod
    def _validate_price_join(
        identity_path: Path,
        records: list[IdentityReportRow],
        price_path: Path,
        prices: Mapping[int, Mapping[str, float | None]],
        price_inputs: Mapping[int, str],
    ) -> None:
        identity_inputs: dict[int, str] = {}
        for record in records:
            previous = identity_inputs.setdefault(
                record.source_index, record.input_smiles
            )
            if previous != record.input_smiles:
                raise ValueError(
                    f"{identity_path}: source_index {record.source_index} has conflicting "
                    f"input_smiles values {previous!r} and {record.input_smiles!r}"
                )
        identity_keys = set(identity_inputs)
        price_keys = set(prices)
        missing = sorted(identity_keys - price_keys)
        extra = sorted(price_keys - identity_keys)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing source_index values {missing}")
            if extra:
                details.append(f"unknown source_index values {extra}")
            raise ValueError(
                f"{price_path}: price/identity source_index mismatch with "
                f"{identity_path}: {'; '.join(details)}"
            )
        for source_index, input_smiles in identity_inputs.items():
            price_smiles = price_inputs[source_index]
            if price_smiles != input_smiles:
                raise ValueError(
                    f"{price_path}: source_index {source_index} input_smiles "
                    f"{price_smiles!r} does not match identity input_smiles "
                    f"{input_smiles!r}"
                )

    def stock(
        self, identity_format: StockIdentityFormat = "smiles"
    ) -> BuildingBlockStock:
        """Build a legacy stock view; planners can use the catalog directly."""
        source = (
            self.by_canonical_smiles
            if identity_format == "smiles"
            else self.by_inchikey
        )
        return BuildingBlockStock._from_validated_keys(
            frozenset(source), identity_format
        )

    def key_for_molecule(self, molecule: MoleculeContainer) -> str:
        """Generate the configured planner membership key for a molecule."""
        if self.identity_format == "inchikey":
            return molecule_to_inchi_key(molecule)
        return str(safe_canonicalization(molecule, clean_stereo=False))

    def contains_key(self, key: str) -> bool:
        """Return whether an already generated identity key is in the catalog."""
        return key in self.keys

    def contains_molecule(self, molecule: MoleculeContainer) -> bool:
        """Return whether a molecule belongs to the prepared planner stock."""
        return self.contains_key(self.key_for_molecule(molecule))

    def provenance_for_molecule(
        self, molecule: MoleculeContainer
    ) -> tuple[Mapping[str, object], ...]:
        """Return JSON-compatible preparation provenance for a stock molecule."""
        key = self.key_for_molecule(molecule)
        source = (
            self.by_canonical_smiles
            if self.identity_format == "smiles"
            else self.by_inchikey
        )
        return tuple(asdict(record) for record in source.get(key, ()))

    @property
    def by_stereo_free_smiles(
        self,
    ) -> dict[str, tuple[IdentityReportRow, ...]]:
        """Build the legacy complete stereo-free index only on explicit access."""
        if self._by_stereo_free_smiles is None:
            grouped: dict[str, list[IdentityReportRow]] = defaultdict(list)
            for record in self.records:
                if record.status not in {"written", "duplicate_skipped"}:
                    continue
                molecule = smiles_parser(record.canonical_smiles)
                if not isinstance(molecule, MoleculeContainer):
                    continue
                molecule.clean_stereo()
                stereo_free = str(safe_canonicalization(molecule, clean_stereo=False))
                grouped[stereo_free].append(record)
            self._by_stereo_free_smiles = {
                key: tuple(items) for key, items in grouped.items()
            }
        return self._by_stereo_free_smiles

    def _stereo_candidates_from_file(self, stereo_free: str) -> tuple[str, ...]:
        cached = self._stereo_candidate_cache.get(stereo_free)
        if cached is not None:
            return cached
        path = self.stereo_file
        if path is None:
            return ()
        candidates: list[str] = []
        try:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                missing = _STEREO_COLUMNS.difference(reader.fieldnames or ())
                if missing:
                    raise ValueError(
                        f"{path}: missing stereo columns: {sorted(missing)}"
                    )
                for line_number, row in enumerate(reader, start=2):
                    if (
                        row["status"] not in {"written", "duplicate_skipped"}
                        or row["stereo_free_smiles"] != stereo_free
                    ):
                        continue
                    canonical = row["canonical_smiles"].strip()
                    try:
                        source_index = int(row["source_index"])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"{path}:{line_number}: invalid source_index"
                        ) from error
                    matching_records = self.by_canonical_smiles.get(canonical, ())
                    if not any(
                        record.source_index == source_index
                        and record.output_origin == row["output_origin"]
                        and record.status == row["status"]
                        for record in matching_records
                    ):
                        raise ValueError(
                            f"{path}:{line_number}: stereo row does not match "
                            "the identity artifact"
                        )
                    candidates.append(canonical)
        except OSError as error:
            raise ValueError(f"Could not read BB stereo file {path}") from error
        result = tuple(dict.fromkeys(candidates))
        self._stereo_candidate_cache[stereo_free] = result
        return result

    def validate_stereo_for_molecule(
        self, molecule: MoleculeContainer
    ) -> tuple[bool, tuple[str, ...]]:
        """Compare a propagated leaf with stereo-bearing catalog identities.

        The boolean is true only for an exact canonical identity. Candidates
        share the same stereo-free graph, exposing opposite or unspecified
        catalog assignments to route postprocessors.
        """
        canonical = str(safe_canonicalization(molecule, clean_stereo=False))
        exact = canonical in self.by_canonical_smiles
        if exact:
            return True, (canonical,)
        stereo_free_molecule = molecule.copy()
        stereo_free_molecule.clean_stereo()
        stereo_free = str(
            safe_canonicalization(stereo_free_molecule, clean_stereo=False)
        )
        if self.stereo_file is not None:
            candidates = self._stereo_candidates_from_file(stereo_free)
        else:
            candidates = tuple(
                dict.fromkeys(
                    record.canonical_smiles
                    for record in self.by_stereo_free_smiles.get(stereo_free, ())
                )
            )
        return False, candidates

    def __contains__(self, key: object) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)

    def protected_alternative_records(
        self,
        canonical_smiles: str,
        provenance_records: list[Mapping[str, object]] | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        """Return complete provenance records for protected alternatives."""
        if provenance_records:
            candidates = (
                dict(record)
                for record in provenance_records
                if record.get("output_origin") == "deprotected"
                and record.get("input_smiles")
            )
        else:
            candidates = (
                asdict(record)
                for record in self.by_canonical_smiles.get(canonical_smiles, ())
                if record.output_origin == "deprotected"
            )
        unique: dict[tuple[object, ...], Mapping[str, object]] = {}
        for record in candidates:
            key = (
                record.get("source_index"),
                record.get("input_smiles"),
                record.get("mapped_deprotection"),
            )
            unique.setdefault(key, record)
        return tuple(unique.values())

    def protected_alternatives(
        self,
        canonical_smiles: str,
        provenance_records: list[Mapping[str, object]] | None = None,
    ) -> tuple[str, ...]:
        """Return unique protected inputs for one deprotected planner identity."""
        values = (
            str(record["input_smiles"])
            for record in self.protected_alternative_records(
                canonical_smiles, provenance_records
            )
        )
        return tuple(dict.fromkeys(values))

    def best_prices(
        self, aliases: Mapping[str, str]
    ) -> tuple[
        tuple[str, ...],
        set[str],
        dict[str, tuple[float, str, int, int]],
    ]:
        """Match route aliases to the lowest positive loaded vendor prices."""
        columns = tuple(
            dict.fromkeys(
                column for prices in self.prices_by_source.values() for column in prices
            )
        )
        matched: set[str] = set()
        best: dict[str, tuple[float, str, int, int]] = {}
        for record in self.records:
            canonical = aliases.get(record.input_smiles) or aliases.get(
                record.canonical_smiles
            )
            if canonical is None:
                continue
            matched.add(canonical)
            for column_index, column in enumerate(columns):
                price = self.prices_by_source.get(record.source_index, {}).get(column)
                if price is None or price == 0:
                    continue
                candidate = (price, column, column_index, record.source_index)
                previous = best.get(canonical)
                if previous is None or (price, column_index, record.source_index) < (
                    previous[0],
                    previous[2],
                    previous[3],
                ):
                    best[canonical] = candidate
        return columns, matched, best


__all__ = ["BuildingBlockCatalog"]
