"""Record framing and provenance helpers shared by audited workflows.

The chemistry feature modules own their outcome statuses and report schemas. This
module owns only the format-level contracts needed to read a source record without
losing its identity or provenance and to fingerprint immutable run inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MOLECULE_SUFFIXES = frozenset({".smi", ".smiles", ".cxsmiles"})
_CHEMISTRY_COLUMNS = frozenset({"smiles", "cxsmiles"})


def compact_json(value: Any) -> str:
    """Serialize a value as compact, UTF-8-friendly JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tsv_field(value: object) -> str:
    """Return a single-line TSV field without changing ordinary backslashes."""
    return str(value).replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def _is_complete_cxsmiles(value: str) -> bool:
    """Recognise the whitespace-bearing shape of one complete CXSMILES field."""
    head, separator, extension = value.partition(" ")
    return bool(
        head and separator and extension.startswith("|") and extension.endswith("|")
    )


@dataclass(frozen=True, slots=True)
class InputRecord:
    """One input row with stable ordering and lossless source provenance."""

    sequence: int
    line_number: int
    chemistry: str
    raw: str
    metadata: tuple[str, ...] = ()
    metadata_names: tuple[str, ...] = ()
    headered: bool = False
    kind: str = "molecule"
    fields: tuple[str, ...] = ()
    format_error: str | None = None

    @property
    def metadata_value(self) -> list[str] | dict[str, str]:
        if self.headered:
            return dict(zip(self.metadata_names, self.metadata))
        return list(self.metadata)

    @property
    def source_info(self) -> str:
        return compact_json({"line": self.line_number, "metadata": self.metadata_value})

    def source_info_with(self, context: Mapping[str, object]) -> str:
        payload: dict[str, object] = {
            "line": self.line_number,
            "metadata": self.metadata_value,
        }
        if context:
            payload["context"] = dict(context)
        return compact_json(payload)

    @property
    def input_record(self) -> str:
        return self.chemistry if self.kind == "molecule" else self.raw

    @property
    def fallback_record(self) -> str:
        if self.kind == "pathway" or not self.headered:
            return self.raw
        return f"{self.chemistry}\t{compact_json(self.metadata_value)}"


def _molecule_record(
    *,
    sequence: int,
    line_number: int,
    raw: str,
    chemistry: str,
    metadata: tuple[str, ...],
    metadata_names: tuple[str, ...] = (),
    headered: bool = False,
) -> InputRecord:
    error = None
    if not chemistry:
        error = "the chemistry field is empty"
    elif any(
        character.isspace() for character in chemistry
    ) and not _is_complete_cxsmiles(chemistry):
        error = "metadata must be TAB-separated; arbitrary whitespace is not supported"
    return InputRecord(
        sequence=sequence,
        line_number=line_number,
        chemistry=chemistry,
        raw=raw,
        metadata=metadata,
        metadata_names=metadata_names,
        headered=headered,
        format_error=error,
    )


def iter_molecule_records(path: str | Path) -> Iterator[InputRecord]:
    """Yield strict SMI/CXSMILES records or formally headered TSV rows.

    Headerless inputs use TAB as their only metadata delimiter, preserving spaces in a
    complete CXSMILES field.  TSV inputs require exactly one case-insensitive
    ``SMILES`` or ``CXSMILES`` column and preserve all other columns as provenance.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in _MOLECULE_SUFFIXES and suffix != ".tsv":
        raise ValueError(
            f"unsupported molecule input format {source.suffix!r}; "
            "expected .smi, .smiles, .cxsmiles or .tsv"
        )

    with source.open(encoding="utf-8", newline="") as handle:
        if suffix != ".tsv":
            sequence = 0
            for line_number, line in enumerate(handle, 1):
                raw = line.rstrip("\r\n")
                if not raw.strip():
                    continue
                sequence += 1
                fields = raw.split("\t")
                yield _molecule_record(
                    sequence=sequence,
                    line_number=line_number,
                    raw=raw,
                    chemistry=fields[0].strip(),
                    metadata=tuple(fields[1:]),
                )
            return

        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"{source}: expected a TSV header") from error
        normalized = [column.strip().casefold() for column in header]
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"{source}: TSV header names must be unique (case-insensitive)"
            )
        chemistry_indexes = [
            index
            for index, column in enumerate(normalized)
            if column in _CHEMISTRY_COLUMNS
        ]
        if len(chemistry_indexes) != 1:
            raise ValueError(
                f"{source}: expected exactly one SMILES or CXSMILES column, "
                f"found {len(chemistry_indexes)}"
            )
        chemistry_index = chemistry_indexes[0]
        metadata_indexes = tuple(
            index for index in range(len(header)) if index != chemistry_index
        )
        metadata_names = tuple(header[index].strip() for index in metadata_indexes)
        sequence = 0
        for line_number, fields in enumerate(reader, 2):
            if not fields or not any(field.strip() for field in fields):
                continue
            sequence += 1
            raw = "\t".join(fields)
            field_count_error = None
            if len(fields) != len(header):
                field_count_error = (
                    f"expected {len(header)} TSV fields, found {len(fields)}"
                )
            padded = fields + [""] * max(0, len(header) - len(fields))
            record = _molecule_record(
                sequence=sequence,
                line_number=line_number,
                raw=raw,
                chemistry=padded[chemistry_index].strip(),
                metadata=tuple(padded[index] for index in metadata_indexes),
                metadata_names=metadata_names,
                headered=True,
            )
            if field_count_error is not None:
                record = InputRecord(
                    sequence=record.sequence,
                    line_number=record.line_number,
                    chemistry=record.chemistry,
                    raw=record.raw,
                    metadata=record.metadata,
                    metadata_names=record.metadata_names,
                    headered=True,
                    format_error=field_count_error,
                )
            yield record


def iter_pathway_records(path: str | Path) -> Iterator[InputRecord]:
    """Yield the fixed, headerless five-column fragmentation TSV records."""
    source = Path(path)
    with source.open(encoding="utf-8", newline="") as handle:
        sequence = 0
        for line_number, line in enumerate(handle, 1):
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            sequence += 1
            fields = tuple(raw.split("\t"))
            error = None
            if len(fields) != 5:
                error = f"expected 5 fragmentation TSV fields, found {len(fields)}"
            elif not all(fields[:3]):
                error = "target, pathway id and synthons fields must be non-empty"
            yield InputRecord(
                sequence=sequence,
                line_number=line_number,
                chemistry=fields[0] if fields else "",
                raw=raw,
                metadata=fields[1:],
                metadata_names=("pathway_id", "synthons", "depth", "availability"),
                kind="pathway",
                fields=fields,
                format_error=error,
            )


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_metadata(
    path: str | Path, *, reported_path: str | Path | None = None
) -> dict[str, object]:
    """Return stable path, byte size and SHA-256 provenance for a file."""
    source = Path(path)
    shown = Path(reported_path) if reported_path is not None else source
    return {
        "path": str(shown.resolve()),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def utc_now() -> str:
    """Return a timezone-aware ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "InputRecord",
    "compact_json",
    "file_metadata",
    "iter_molecule_records",
    "iter_pathway_records",
    "sha256_file",
    "tsv_field",
    "utc_now",
]
