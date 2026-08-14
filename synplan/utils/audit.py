"""Record, provenance, and output-transaction helpers for audited workflows.

Chemistry feature modules own their outcome statuses, counters, and report schemas.
This module owns shared record framing, immutable-source checks, staged text artifacts,
and summary-last publication.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from synplan.utils.files import ChemicalRecord, iter_chemical_records

_MOLECULE_SUFFIXES = frozenset({".smi", ".smiles", ".cxsmiles"})

InputRecord = ChemicalRecord


def compact_json(value: Any) -> str:
    """Serialize a value as compact, UTF-8-friendly JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tsv_field(value: object) -> str:
    """Return a single-line TSV field without changing ordinary backslashes."""
    return str(value).replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def iter_molecule_records(path: str | Path) -> Iterator[InputRecord]:
    """Yield strict SMI/CXSMILES records or formally headered TSV rows."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in _MOLECULE_SUFFIXES and suffix != ".tsv":
        raise ValueError(
            f"unsupported molecule input format {source.suffix!r}; "
            "expected .smi, .smiles, .cxsmiles or .tsv"
        )
    input_format = "tsv" if suffix == ".tsv" else "smi"
    yield from iter_chemical_records(
        source,
        input_format=input_format,
        chemistry_columns={"smiles": "smiles", "cxsmiles": "smiles"},
    )


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


def partial_output_paths(
    final_paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Return deterministic partial paths for an output bundle."""
    return {
        key: Path(path).with_name(Path(path).name + ".partial")
        for key, path in final_paths.items()
    }


def guard_output_bundle(
    final_paths: Mapping[str, str | Path],
    partial_paths: Mapping[str, str | Path],
    source_paths: Mapping[str, str | Path],
    overwrite: str,
    *,
    create_parents: bool = False,
) -> None:
    """Validate a transactional output bundle before any file is opened."""
    if overwrite not in {"error", "replace"}:
        raise ValueError(f"unknown audit overwrite policy: {overwrite!r}")
    finals = {key: Path(path) for key, path in final_paths.items()}
    partials = {key: Path(path) for key, path in partial_paths.items()}
    artifacts = (*finals.values(), *partials.values())
    resolved_artifacts = [path.resolve() for path in artifacts]
    if len(resolved_artifacts) != len(set(resolved_artifacts)):
        raise ValueError("audit output and sidecar paths must be distinct")
    artifact_set = set(resolved_artifacts)
    for label, source in source_paths.items():
        if Path(source).resolve() in artifact_set:
            raise ValueError(f"{label} path collides with an audit output: {source}")
    directories = [path for path in artifacts if path.is_dir()]
    if directories:
        names = ", ".join(str(path) for path in directories)
        raise IsADirectoryError(f"audit artifact paths are directories: {names}")
    if overwrite == "error":
        existing = [path for path in artifacts if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"audited outputs already exist: {names}")
    else:
        for path in partials.values():
            if path.exists():
                path.unlink()
    if create_parents:
        for path in partials.values():
            path.parent.mkdir(parents=True, exist_ok=True)


def close_output_handles(handles: Mapping[str, TextIO]) -> None:
    """Flush, fsync, and close all open handles in an output bundle."""
    for handle in handles.values():
        if handle.closed:
            continue
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()


def line_count(path: str | Path) -> int:
    """Count physical lines in a staged text artifact."""
    with Path(path).open("rb") as handle:
        return sum(1 for _ in handle)


def promote_output_bundle(
    final_paths: Mapping[str, str | Path],
    partial_paths: Mapping[str, str | Path],
    *,
    summary_key: str = "summary.json",
) -> None:
    """Promote data files and publish the summary commit marker last."""
    finals = {key: Path(path) for key, path in final_paths.items()}
    partials = {key: Path(path) for key, path in partial_paths.items()}
    if set(finals) != set(partials):
        raise ValueError("final and partial output keys must match")
    summary_final = finals.get(summary_key)
    if summary_final is not None and summary_final.exists():
        summary_final.unlink()
    for key, final_path in finals.items():
        if key == summary_key:
            continue
        os.replace(partials[key], final_path)
    if summary_final is not None:
        os.replace(partials[summary_key], summary_final)


class OutputBundleTransaction:
    """Stage, validate, and publish one bundle of text artifacts.

    Workflow modules retain ownership of their record statuses, counters, report
    schemas, and summary contents. This class owns only the shared filesystem
    transaction and immutable-input provenance contract.
    """

    def __init__(
        self,
        final_paths: Mapping[str, str | Path],
        source_paths: Mapping[str, str | Path],
        overwrite: str,
        *,
        create_parents: bool = False,
        summary_key: str | None = "summary.json",
    ) -> None:
        self.final_paths = {
            key: Path(path).expanduser().resolve(strict=False)
            for key, path in final_paths.items()
        }
        self.partial_paths = partial_output_paths(self.final_paths)
        self.source_paths = {
            key: Path(path).expanduser().resolve(strict=False)
            for key, path in source_paths.items()
        }
        if summary_key is not None and summary_key not in self.final_paths:
            raise ValueError(
                f"summary artifact {summary_key!r} is not present in final_paths"
            )
        self.summary_key = summary_key
        self.handles: dict[str, TextIO] = {}
        self._source_metadata: dict[str, dict[str, object]] = {}
        self._opened = False
        guard_output_bundle(
            self.final_paths,
            self.partial_paths,
            self.source_paths,
            overwrite,
            create_parents=create_parents,
        )

    @property
    def source_metadata(self) -> dict[str, dict[str, object]]:
        """Return immutable source snapshots captured when staging began."""
        if not self._opened:
            raise RuntimeError("output transaction has not been opened")
        return dict(self._source_metadata)

    def open(self, keys: Iterable[str] | None = None) -> dict[str, TextIO]:
        """Capture source provenance and open selected staged text artifacts."""
        if self._opened:
            raise RuntimeError("output transaction can only be opened once")
        selected = tuple(
            key
            for key in (keys if keys is not None else self.partial_paths)
            if key != self.summary_key
        )
        unknown = set(selected).difference(self.partial_paths)
        if unknown:
            raise KeyError(f"unknown output artifact keys: {sorted(unknown)}")
        self._source_metadata = {
            name: file_metadata(path) for name, path in self.source_paths.items()
        }
        self._opened = True
        try:
            for key in selected:
                self.handles[key] = self.partial_paths[key].open(
                    "x", encoding="utf-8", newline=""
                )
        except Exception:
            close_output_handles(self.handles)
            raise
        return self.handles

    def close(self) -> None:
        """Flush, fsync, and close all staged handles."""
        close_output_handles(self.handles)

    def validate_sources_unchanged(self, *, activity: str = "processing") -> None:
        """Require every input/provenance file to match its opening snapshot."""
        if not self._opened:
            raise RuntimeError("output transaction has not been opened")
        for name, path in self.source_paths.items():
            if file_metadata(path) != self._source_metadata[name]:
                raise RuntimeError(f"{name} changed during {activity}: {path}")

    def validate_line_counts(self, expected: Mapping[str, int]) -> None:
        """Validate physical row counts for selected staged artifacts."""
        unknown = set(expected).difference(self.partial_paths)
        if unknown:
            raise KeyError(f"unknown output artifact keys: {sorted(unknown)}")
        for key, expected_lines in expected.items():
            observed = line_count(self.partial_paths[key])
            if observed != expected_lines:
                raise RuntimeError(
                    f"staged line-count mismatch for {key}: "
                    f"expected {expected_lines}, observed {observed}"
                )

    def artifact_metadata(
        self, keys: Iterable[str] | None = None
    ) -> dict[str, dict[str, object]]:
        """Return final-path metadata calculated from staged artifacts."""
        selected = tuple(
            key
            for key in (keys if keys is not None else self.partial_paths)
            if key != self.summary_key
        )
        unknown = set(selected).difference(self.partial_paths)
        if unknown:
            raise KeyError(f"unknown output artifact keys: {sorted(unknown)}")
        output: dict[str, dict[str, object]] = {}
        for key in selected:
            metadata = file_metadata(
                self.partial_paths[key], reported_path=self.final_paths[key]
            )
            metadata["rows"] = line_count(self.partial_paths[key])
            output[key] = metadata
        return output

    def write_summary(self, summary: Mapping[str, object]) -> None:
        """Write and fsync the staged JSON summary commit marker."""
        if self.summary_key is None:
            raise RuntimeError("this output transaction has no summary artifact")
        path = self.partial_paths[self.summary_key]
        with path.open("x", encoding="utf-8", newline="") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def promote(self) -> None:
        """Publish staged artifacts, with the summary commit marker last."""
        promote_output_bundle(
            self.final_paths,
            self.partial_paths,
            summary_key=self.summary_key or "summary.json",
        )

    def __enter__(self) -> OutputBundleTransaction:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


__all__ = [
    "InputRecord",
    "OutputBundleTransaction",
    "close_output_handles",
    "compact_json",
    "file_metadata",
    "guard_output_bundle",
    "iter_molecule_records",
    "iter_pathway_records",
    "line_count",
    "partial_output_paths",
    "promote_output_bundle",
    "sha256_file",
    "tsv_field",
    "utc_now",
]
