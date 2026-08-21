"""Record-aware, transactional audit output for the synthon command line tools.

The chemistry commands decide how one input record is processed.  This module owns the
shared text contracts around that processing: strict SMI/CXSMILES and TSV parsing,
failure sidecars, reproducible counters and atomic publication of a complete run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from synplan import __version__
from synplan.synthon.config import SynthonConfig

FALLBACK_HEADER = "# input_record\tsource_info\tstatus\tdetail\n"
ERROR_HEADER = "# input_record\tsource_info\tstage\terror_type\terror_message\n"
SIDECAR_NAMES = (
    "fallback.smi",
    "fallback.tsv",
    "errors.tsv",
    "summary.json",
    "run.log",
)
_RESERVED_ARTIFACT_NAMES = frozenset(
    (*SIDECAR_NAMES, *(f"{name}.partial" for name in SIDECAR_NAMES))
)

_MOLECULE_SUFFIXES = frozenset({".smi", ".smiles", ".cxsmiles"})
_CHEMISTRY_COLUMNS = frozenset({"smiles", "cxsmiles"})
_SUCCESS_STATUS = {
    "bb_classifying": "classified",
    "bb_synthonizing": "synthonised",
    "synthon_fragment": "fragmented",
    "synthon_enumerate": "enumerated",
    "bb_scaffolds": "scaffolded",
}
_RETRYABLE_STATUS = {
    "bb_classifying": frozenset({"unclassified"}),
    "bb_synthonizing": frozenset({"unclassified", "no_synthon", "max_components"}),
    "synthon_fragment": frozenset({"no_pathways"}),
    "synthon_enumerate": frozenset({"missing_stock_slots", "no_products"}),
    "bb_scaffolds": frozenset(),
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _tsv_field(value: object) -> str:
    """Return one reversible-enough, single-line TSV field."""
    return str(value).replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def _is_complete_cxsmiles(value: str) -> bool:
    """Recognise the whitespace-bearing shape accepted for one CXSMILES field.

    The chemistry parser remains authoritative later.  Here we only distinguish a
    complete ``SMILES |CX extension|`` field from the legacy ``SMILES name`` format,
    whose metadata separator is intentionally no longer ambiguous.
    """
    head, separator, extension = value.partition(" ")
    return bool(
        head and separator and extension.startswith("|") and extension.endswith("|")
    )


@dataclass(frozen=True, slots=True)
class InputRecord:
    """One source row and enough provenance to reproduce or diagnose it."""

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
        return _json({"line": self.line_number, "metadata": self.metadata_value})

    def source_info_with(self, context: Mapping[str, object]) -> str:
        payload: dict[str, object] = {
            "line": self.line_number,
            "metadata": self.metadata_value,
        }
        if context:
            payload["context"] = dict(context)
        return _json(payload)

    @property
    def input_record(self) -> str:
        return self.chemistry if self.kind == "molecule" else self.raw

    @property
    def fallback_record(self) -> str:
        if self.kind == "pathway" or not self.headered:
            return self.raw
        return f"{self.chemistry}\t{_json(self.metadata_value)}"


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
    """Yield SMI/CXSMILES records or rows from a formally headered TSV.

    Headerless molecule files use TAB as their only metadata delimiter, preserving
    whitespace inside a complete CXSMILES field.  A TSV must have exactly one
    case-insensitive ``SMILES`` or ``CXSMILES`` column; its remaining columns are
    retained as named provenance.
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
            format_error = None
            if len(fields) != len(header):
                format_error = f"expected {len(header)} TSV fields, found {len(fields)}"
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
            if format_error is not None:
                record = InputRecord(
                    sequence=record.sequence,
                    line_number=record.line_number,
                    chemistry=record.chemistry,
                    raw=record.raw,
                    metadata=record.metadata,
                    metadata_names=record.metadata_names,
                    headered=True,
                    format_error=format_error,
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
                metadata_names=(
                    "pathway_id",
                    "synthons",
                    "depth",
                    "availability",
                ),
                kind="pathway",
                fields=fields,
                format_error=error,
            )


@dataclass(frozen=True, slots=True)
class AuditError:
    stage: str
    error_type: str
    message: str
    context: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AuditOutcome:
    record: InputRecord
    status: str
    output_rows: tuple[str, ...] = ()
    detail: str = ""
    errors: tuple[AuditError, ...] = ()
    retryable: bool = False
    metrics: Mapping[str, int | bool] = field(default_factory=dict)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_metadata(
    path: Path, *, reported_path: Path | None = None
) -> dict[str, object]:
    return {
        "path": str((reported_path or path).resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditRun:
    """Write and atomically publish one command's primary output and audit bundle."""

    def __init__(
        self,
        command: str,
        input_file: str | Path,
        output_file: str | Path,
        config: SynthonConfig,
        *,
        provenance_files: Mapping[str, str | Path] | None = None,
        progress_every: int = 10_000,
    ) -> None:
        if command not in _SUCCESS_STATUS:
            raise ValueError(f"unknown audited synthon command: {command}")
        self.command = command
        self.input_path = Path(input_file)
        self.output_path = Path(output_file)
        self.config = config
        self.provenance_paths = {
            name: Path(path) for name, path in (provenance_files or {}).items()
        }
        self.progress_every = max(1, progress_every)
        statuses = {
            _SUCCESS_STATUS[command],
            "processing_error",
            *_RETRYABLE_STATUS[command],
        }
        initial_counts = {
            "input_records": 0,
            "successful_input_records": 0,
            "output_rows": 0,
            "fallback_records": 0,
            "fallback_smi_records": 0,
            "records_with_errors": 0,
            "error_rows": 0,
            **{f"status_{status}": 0 for status in statuses},
        }
        if command == "bb_synthonizing":
            initial_counts["forced_keep_pg"] = 0
        self.counters: Counter[str] = Counter(initial_counts)
        self.summary: dict[str, object] | None = None
        self._handles: dict[str, TextIO] = {}
        self._expected_sequence = 1
        self._next_progress = self.progress_every
        self._started_at = ""
        self._started = 0.0
        self._input_metadata: dict[str, object] | None = None
        self._provenance_metadata: dict[str, dict[str, object]] = {}

        directory = self.output_path.parent
        if self.output_path.name in _RESERVED_ARTIFACT_NAMES:
            raise ValueError(
                f"output filename {self.output_path.name!r} is reserved for audit artifacts"
            )
        self.final_paths = {
            "primary": self.output_path,
            **{name: directory / name for name in SIDECAR_NAMES},
        }
        self.partial_paths = {
            key: path.with_name(path.name + ".partial")
            for key, path in self.final_paths.items()
        }
        all_artifacts = (*self.final_paths.values(), *self.partial_paths.values())
        resolved = [path.resolve() for path in all_artifacts]
        if len(set(resolved)) != len(resolved):
            raise ValueError("audit output and sidecar paths must be distinct")
        self._guard_paths()

    def _guard_paths(self) -> None:
        resolved_artifacts = {
            path.resolve()
            for path in (*self.final_paths.values(), *self.partial_paths.values())
        }
        sources = {"input": self.input_path, **self.provenance_paths}
        for label, source in sources.items():
            if source.resolve() in resolved_artifacts:
                raise ValueError(
                    f"{label} path collides with an audit output: {source}"
                )

        guarded = (*self.final_paths.values(), *self.partial_paths.values())
        directories = [path for path in guarded if path.is_dir()]
        if directories:
            names = ", ".join(str(path) for path in directories)
            raise IsADirectoryError(f"audit artifact paths are directories: {names}")
        if self.config.audit_overwrite == "error":
            existing = [path for path in guarded if path.exists()]
            if existing:
                names = ", ".join(str(path) for path in existing)
                raise FileExistsError(f"audited outputs already exist: {names}")
            return

        for path in self.partial_paths.values():
            if path.is_dir():
                raise IsADirectoryError(path)
            if path.exists():
                path.unlink()

    def __enter__(self) -> AuditRun:
        self._started_at = _utc_now()
        self._started = time.perf_counter()
        self._input_metadata = _file_metadata(self.input_path)
        self._provenance_metadata = {
            name: _file_metadata(path) for name, path in self.provenance_paths.items()
        }
        try:
            for key in (
                "primary",
                "fallback.smi",
                "fallback.tsv",
                "errors.tsv",
                "run.log",
            ):
                self._handles[key] = self.partial_paths[key].open(
                    "x", encoding="utf-8", newline=""
                )
        except Exception:
            for handle in self._handles.values():
                handle.close()
            raise
        self._handles["fallback.tsv"].write(FALLBACK_HEADER)
        self._handles["errors.tsv"].write(ERROR_HEADER)
        self._log(
            "INFO",
            f"starting command={self.command} input={self.input_path} "
            f"output={self.output_path}",
        )
        return self

    def _log(self, level: str, message: str) -> None:
        handle = self._handles.get("run.log")
        if handle is not None and not handle.closed:
            handle.write(f"{_utc_now()} {level} {message}\n")
            handle.flush()

    def write(self, outcome: AuditOutcome) -> None:
        if not self._handles:
            raise RuntimeError("AuditRun must be entered before writing outcomes")
        if outcome.record.sequence != self._expected_sequence:
            raise ValueError(
                "audited outcomes must be ordered by input sequence: "
                f"expected {self._expected_sequence}, got {outcome.record.sequence}"
            )
        self._expected_sequence += 1

        success = _SUCCESS_STATUS[self.command]
        retryable_statuses = _RETRYABLE_STATUS[self.command]
        allowed = {success, "processing_error", *retryable_statuses}
        if outcome.status not in allowed:
            raise ValueError(
                f"status {outcome.status!r} is not valid for {self.command}"
            )
        if outcome.output_rows and outcome.status != success:
            raise ValueError("only a successful outcome may have primary output rows")
        if not outcome.output_rows and outcome.status == success:
            raise ValueError("a successful outcome must have a primary output row")

        self.counters["input_records"] += 1
        self.counters[f"status_{outcome.status}"] += 1
        if outcome.output_rows:
            self.counters["successful_input_records"] += 1
            for row in outcome.output_rows:
                self._handles["primary"].write(row.rstrip("\r\n") + "\n")
                self.counters["output_rows"] += 1
        else:
            self.counters["fallback_records"] += 1
            self._handles["fallback.tsv"].write(
                f"{_tsv_field(outcome.record.input_record)}\t"
                f"{_tsv_field(outcome.record.source_info)}\t"
                f"{_tsv_field(outcome.status)}\t{_tsv_field(outcome.detail)}\n"
            )
            retryable = outcome.status != "processing_error" and (
                outcome.retryable or outcome.status in retryable_statuses
            )
            if retryable:
                self._handles["fallback.smi"].write(
                    outcome.record.fallback_record.rstrip("\r\n") + "\n"
                )
                self.counters["fallback_smi_records"] += 1

        if outcome.errors:
            self.counters["records_with_errors"] += 1
        for error in outcome.errors:
            self._handles["errors.tsv"].write(
                f"{_tsv_field(outcome.record.input_record)}\t"
                f"{_tsv_field(outcome.record.source_info_with(error.context))}\t"
                f"{_tsv_field(error.stage)}\t{_tsv_field(error.error_type)}\t"
                f"{_tsv_field(error.message)}\n"
            )
            self.counters["error_rows"] += 1

        for name, value in outcome.metrics.items():
            if not isinstance(value, (bool, int)):
                raise TypeError(f"audit metric {name!r} must be an int or bool")
            self.counters[name] += int(value)

        if self.counters["input_records"] >= self._next_progress:
            elapsed = max(time.perf_counter() - self._started, 1e-9)
            self._log(
                "INFO",
                f"processed={self.counters['input_records']} "
                f"output_rows={self.counters['output_rows']} "
                f"fallback={self.counters['fallback_records']} "
                f"errors={self.counters['error_rows']} "
                f"rate={self.counters['input_records'] / elapsed:.1f} rows/s",
            )
            self._next_progress += self.progress_every

    def _flush_and_close(self) -> None:
        for handle in self._handles.values():
            if handle.closed:
                continue
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    @staticmethod
    def _line_count(path: Path) -> int:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)

    def _validate_partials(self) -> None:
        expected_lines = {
            "primary": self.counters["output_rows"],
            "fallback.smi": self.counters["fallback_smi_records"],
            "fallback.tsv": self.counters["fallback_records"] + 1,
            "errors.tsv": self.counters["error_rows"] + 1,
        }
        for key, expected in expected_lines.items():
            observed = self._line_count(self.partial_paths[key])
            if observed != expected:
                raise RuntimeError(
                    f"audit line-count mismatch for {key}: expected {expected}, "
                    f"observed {observed}"
                )
        if (
            self.counters["successful_input_records"]
            + self.counters["fallback_records"]
            != self.counters["input_records"]
        ):
            raise RuntimeError(
                "successful and fallback inputs do not partition the run"
            )
        if self._input_metadata is None:
            raise RuntimeError("audit input provenance was not captured")
        if _file_metadata(self.input_path) != self._input_metadata:
            raise RuntimeError(
                f"audit input changed during processing: {self.input_path}"
            )
        for name, path in self.provenance_paths.items():
            if _file_metadata(path) != self._provenance_metadata[name]:
                raise RuntimeError(
                    f"audit provenance file changed during processing: {path}"
                )

    def _build_summary(self) -> dict[str, object]:
        output_files: dict[str, dict[str, object]] = {}
        for key in (
            "primary",
            "fallback.smi",
            "fallback.tsv",
            "errors.tsv",
            "run.log",
        ):
            metadata = _file_metadata(
                self.partial_paths[key], reported_path=self.final_paths[key]
            )
            metadata["rows"] = self._line_count(self.partial_paths[key])
            output_files[self.final_paths[key].name] = metadata
        counters = dict(sorted(self.counters.items()))
        status_counts = {
            key.removeprefix("status_"): value
            for key, value in counters.items()
            if key.startswith("status_")
        }
        return {
            "synplan_version": __version__,
            "schema_version": 1,
            "command": self.command,
            "started_at": self._started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - self._started,
            "input": self._input_metadata,
            "provenance_files": dict(sorted(self._provenance_metadata.items())),
            "config": self.config.to_dict(),
            "ordered_execution": True,
            "status_counts": status_counts,
            "counts": counters,
            **counters,
            "output_files": output_files,
        }

    def _write_summary_partial(self) -> None:
        path = self.partial_paths["summary.json"]
        with path.open("x", encoding="utf-8", newline="") as handle:
            json.dump(self.summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _promote(self) -> None:
        summary_final = self.final_paths["summary.json"]
        if summary_final.exists():
            summary_final.unlink()
        for key in ("primary", "fallback.smi", "fallback.tsv", "errors.tsv", "run.log"):
            os.replace(self.partial_paths[key], self.final_paths[key])
        os.replace(self.partial_paths["summary.json"], summary_final)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            self._log(
                "ERROR",
                f"command failed; partial outputs retained: {exc_type.__name__}: "
                f"{exc_value}",
            )
            self._flush_and_close()
            return False

        self._log(
            "INFO",
            f"finished command={self.command} inputs={self.counters['input_records']} "
            f"output_rows={self.counters['output_rows']} "
            f"fallback={self.counters['fallback_records']} "
            f"errors={self.counters['error_rows']}",
        )
        self._flush_and_close()
        try:
            self._validate_partials()
            self.summary = self._build_summary()
            self._write_summary_partial()
            self._promote()
        except Exception:
            # Data partials are intentionally retained unless promotion itself had begun.
            raise
        return False


__all__ = [
    "ERROR_HEADER",
    "FALLBACK_HEADER",
    "SIDECAR_NAMES",
    "AuditError",
    "AuditOutcome",
    "AuditRun",
    "InputRecord",
    "iter_molecule_records",
    "iter_pathway_records",
    "sha256_file",
]
