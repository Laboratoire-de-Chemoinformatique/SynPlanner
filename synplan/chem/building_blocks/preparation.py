"""Prepare ordinary planner building-block stocks with full provenance."""

from __future__ import annotations

import csv
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, TextIO

from chython import smiles as smiles_parser
from chython.containers import MoleculeContainer, ReactionContainer
from rdkit import rdBase
from rdkit.Chem import rdinchi
from tqdm.auto import tqdm

from synplan import __version__
from synplan.chem.utils import safe_canonicalization
from synplan.utils.audit import (
    OutputBundleTransaction,
    compact_json,
    tsv_field,
    utc_now,
)
from synplan.utils.files import (
    ChemicalRecord,
    count_chemical_records,
    iter_chemical_records,
    resolve_chemical_input_format,
)
from synplan.utils.parallel import default_num_workers, process_pool_map_stream

from .config import BuildingBlockPreparationConfig, DeprotectionPolicy
from .deprotection import DeprotectionEvent, remove_protective_groups
from .identity import molecule_identity
from .reports import (
    COLLISION_FIELDS,
    DUPLICATE_FIELDS,
    IDENTITY_FIELDS,
    STEREO_FIELDS,
    CollisionReportRow,
    DuplicateReportRow,
    IdentityReportRow,
    StereoReportRow,
)
from .rules import protective_rules_path

_FALLBACK_HEADER = "# input_record\tsource_info\tstatus\tdetail\n"
_ERROR_HEADER = "# input_record\tsource_info\tstage\terror_type\terror_message\n"
_AUDIT_NAMES = ("fallback.smi", "fallback.tsv", "errors.tsv", "summary.json", "run.log")


@dataclass(frozen=True, slots=True)
class BuildingBlockPreparationResult:
    """Paths and counters committed by a completed preparation run."""

    output_file: str
    synthon_input: str
    protected_output_file: str | None = None
    inchikey_file: str | None = None
    identity_reference_file: str | None = None
    price_reference_file: str | None = None
    duplicates_file: str | None = None
    collisions_file: str | None = None
    stereo_file: str | None = None
    audit_files: Mapping[str, str] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)


class PreparationReader:
    """Read preparation inputs through the shared chemical-record contract."""

    def __init__(
        self, input_file: str | Path, config: BuildingBlockPreparationConfig
    ) -> None:
        self.path = Path(input_file)
        self.config = config
        self.input_format = resolve_chemical_input_format(
            self.path, config.input_format
        )

    def __iter__(self) -> Iterator[ChemicalRecord]:
        yield from iter_chemical_records(
            self.path,
            input_format=self.input_format,
            chemistry_columns={"smiles": "smiles", "cxsmiles": "smiles"},
            chemistry_column=self.config.smiles_column,
        )

    def count_records(self) -> int:
        """Return the framed input count without parsing molecule graphs."""
        return count_chemical_records(self.path, input_format=self.input_format)


@dataclass(frozen=True, slots=True)
class _ProcessedRecord:
    source: ChemicalRecord
    protected_molecule: MoleculeContainer | None = field(
        default=None, repr=False, compare=False
    )
    protected_smiles: str = ""
    protected_stereo_free: str = ""
    deprotected_molecule: MoleculeContainer | None = field(
        default=None, repr=False, compare=False
    )
    deprotected_smiles: str | None = None
    deprotected_stereo_free: str | None = None
    deprotection_changed: bool = False
    deprotection_events: tuple[DeprotectionEvent, ...] = ()
    mapped_deprotection: str = ""
    error_stage: str = ""
    error_type: str = ""
    error_message: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.error_type)


def _stereo_free_smiles(molecule: MoleculeContainer) -> str:
    stereo_free = molecule.copy()
    stereo_free.clean_stereo()
    return str(stereo_free)


def _process_record(
    args: tuple[ChemicalRecord, bool, DeprotectionPolicy],
) -> _ProcessedRecord:
    source, deprotect, policy = args
    if source.format_error:
        return _ProcessedRecord(
            source=source,
            error_stage="input",
            error_type="InputFormatError",
            error_message=source.format_error,
        )
    try:
        if source.molecule is not None:
            molecule = source.molecule
        else:
            molecule = smiles_parser(source.chemistry, ignore=True)
            if not isinstance(molecule, MoleculeContainer):
                raise ValueError("input was not parsed as one molecule")
        protected = safe_canonicalization(molecule, clean_stereo=False)
        protected_smiles = str(protected)
        protected_stereo_free = _stereo_free_smiles(protected)
    except Exception as error:
        return _ProcessedRecord(
            source=source,
            error_stage="standardization",
            error_type=type(error).__name__,
            error_message=str(error),
        )

    if not deprotect:
        return _ProcessedRecord(
            source=source,
            protected_molecule=protected,
            protected_smiles=protected_smiles,
            protected_stereo_free=protected_stereo_free,
        )
    try:
        deprotected = protected.copy()
        events: list[DeprotectionEvent] = []
        changed = remove_protective_groups(
            deprotected, policy=policy, event_collector=events
        )
        deprotected = safe_canonicalization(deprotected, clean_stereo=False)
        deprotected_smiles = str(deprotected)
        changed = changed and deprotected_smiles != protected_smiles
        mapped_deprotection = (
            format(
                ReactionContainer(
                    reactants=[protected.copy()], products=[deprotected.copy()]
                ),
                "m",
            )
            if changed
            else ""
        )
        return _ProcessedRecord(
            source=source,
            protected_molecule=protected,
            protected_smiles=protected_smiles,
            protected_stereo_free=protected_stereo_free,
            deprotected_molecule=deprotected,
            deprotected_smiles=deprotected_smiles,
            deprotected_stereo_free=_stereo_free_smiles(deprotected),
            deprotection_changed=changed,
            deprotection_events=tuple(events) if changed else (),
            mapped_deprotection=mapped_deprotection,
        )
    except Exception as error:
        return _ProcessedRecord(
            source=source,
            protected_molecule=protected,
            protected_smiles=protected_smiles,
            protected_stereo_free=protected_stereo_free,
            error_stage="deprotection",
            error_type=type(error).__name__,
            error_message=str(error),
        )


def _record_batches(
    records: Iterable[ChemicalRecord], batch_size: int
) -> Iterator[tuple[ChemicalRecord, ...]]:
    iterator = iter(records)
    while batch := tuple(islice(iterator, batch_size)):
        yield batch


def _process_batch(
    args: tuple[tuple[ChemicalRecord, ...], bool, DeprotectionPolicy],
) -> tuple[_ProcessedRecord, ...]:
    records, deprotect, policy = args
    return tuple(_process_record((record, deprotect, policy)) for record in records)


def _processed_records(
    records: Iterable[ChemicalRecord], config: BuildingBlockPreparationConfig
) -> Iterator[_ProcessedRecord]:
    workers = config.num_workers or default_num_workers(cap=8)
    if workers == 1:
        args = (
            (record, config.deprotect, config.deprotect_policy) for record in records
        )
        yield from map(_process_record, args)
        return
    batches = (
        (batch, config.deprotect, config.deprotect_policy)
        for batch in _record_batches(records, config.batch_size)
    )
    for processed_batch in process_pool_map_stream(
        batches,
        _process_batch,
        max_workers=workers,
        max_pending=workers,
        timeout=0,
        ordered=True,
        max_tasks_per_child=max(1, 50_000 // config.batch_size),
    ):
        yield from processed_batch


def _derived_output_path(output_path: Path, suffix: str) -> Path:
    """Build a sibling artifact name from the primary output stem."""
    stem = output_path.with_suffix("")
    return stem.parent / f"{stem.name}{suffix}"


class PreparationRun:
    """Own the state and output lifecycle of one building-block preparation run."""

    def __init__(
        self,
        input_file: str | Path,
        output_file: str | Path,
        config: BuildingBlockPreparationConfig | None = None,
    ) -> None:
        self.config = config or BuildingBlockPreparationConfig()
        self.input_path = Path(input_file).expanduser().resolve(strict=True)
        self.output_path = Path(output_file).expanduser().resolve(strict=False)
        self.reader = PreparationReader(self.input_path, self.config)
        first_record = next(iter(self.reader), None)
        self.price_columns = tuple(
            name
            for name in (first_record.metadata_names if first_record else ())
            if name.casefold().endswith("_ppg")
        )
        if self.config.price_reference_file is not None and not self.price_columns:
            raise ValueError(
                "price_reference_file requires at least one input *_ppg column"
            )
        self.rules_path = protective_rules_path() if self.config.deprotect else None
        if self.output_path.name in _AUDIT_NAMES or self.output_path.name.endswith(
            ".partial"
        ):
            raise ValueError(
                f"reserved preparation output name: {self.output_path.name}"
            )

        final_paths: dict[str, Path] = {"primary": self.output_path}
        if self.config.deprotect:
            final_paths["protected"] = Path(
                self.config.protected_output_file
                or _derived_output_path(self.output_path, "_protected.smi")
            )
        if (
            self.config.deprotect
            or self.config.duplicates_file is not None
            or self.config.write_audit_files
        ):
            final_paths["duplicates"] = Path(
                self.config.duplicates_file
                or _derived_output_path(self.output_path, "_duplicates.tsv")
            )
        if self.config.write_inchikey_stock:
            final_paths["inchikey"] = Path(
                self.config.inchikey_file
                or _derived_output_path(self.output_path, ".inchikey")
            )
            final_paths["identity"] = Path(
                self.config.identity_reference_file
                or _derived_output_path(self.output_path, "_identity.tsv")
            )
            if self.price_columns:
                final_paths["prices"] = Path(
                    self.config.price_reference_file
                    or _derived_output_path(self.output_path, "_prices.tsv")
                )
            final_paths["collisions"] = Path(
                self.config.collisions_file
                or _derived_output_path(self.output_path, "_collisions.tsv")
            )
        if self.config.stereo_file is not None or self.config.write_audit_files:
            final_paths["stereo"] = Path(
                self.config.stereo_file
                or _derived_output_path(self.output_path, "_stereo.tsv")
            )
        if self.config.write_audit_files:
            final_paths.update(
                {name: self.output_path.parent / name for name in _AUDIT_NAMES}
            )

        source_paths = {"input": self.input_path}
        if self.rules_path is not None:
            source_paths["protective-rule taxonomy"] = self.rules_path
        self.transaction = OutputBundleTransaction(
            final_paths,
            source_paths,
            (
                self.config.audit_overwrite
                if self.config.write_audit_files
                else "replace"
            ),
            create_parents=True,
            summary_key="summary.json" if self.config.write_audit_files else None,
        )
        self.started = time.perf_counter()
        self.started_at = utc_now()
        self.progress_every = 10_000
        self.next_progress = self.progress_every
        self.total_records = 0
        self.handles: dict[str, TextIO] = {}
        self.input_metadata: dict[str, object] | None = None
        self.rules_metadata: dict[str, object] | None = None
        self.counts: Counter[str] = Counter(
            input_records=0,
            successful_input_records=0,
            processing_errors=0,
            primary_rows=0,
            protected_rows=0,
            duplicate_rows=0,
            inchikey_rows=0,
            identity_rows=0,
            collision_rows=0,
            stereo_rows=0,
            partial_identity_errors=0,
            price_rows=0,
        )
        self.seen_primary: dict[str, tuple[int, str]] = {}
        self.seen_protected: set[str] = set()
        self.seen_inchikey: set[str] = set()
        self.identities: defaultdict[str, defaultdict[str, list[tuple[int, str]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self.duplicate_writer: csv.DictWriter | None = None
        self.identity_writer: csv.DictWriter | None = None
        self.price_writer: csv.DictWriter | None = None
        self.collision_writer: csv.DictWriter | None = None
        self.stereo_writer: csv.DictWriter | None = None

    def run(self) -> BuildingBlockPreparationResult:
        """Process every input record and publish the validated output bundle."""
        try:
            self._open_outputs()
            self.total_records = self.reader.count_records()
            self._log(
                "INFO",
                f"starting building-block preparation input={self.input_path} "
                f"output={self.output_path} total={self.total_records} "
                f"workers={self.config.num_workers or default_num_workers(cap=8)}",
            )
            with tqdm(
                total=self.total_records,
                desc="Preparing building blocks",
                unit="record",
                unit_scale=True,
                dynamic_ncols=True,
                smoothing=0.1,
                disable=None,
            ) as progress:
                for processed in _processed_records(self.reader, self.config):
                    self.consume(processed)
                    progress.update()
                    self._write_progress_log(progress)
            self._write_collisions()
            self._write_completed_log()
            self.transaction.close()
            self._validate()
            if self.config.write_audit_files:
                self.transaction.write_summary(self._build_summary())
            self.transaction.promote()
        except Exception:
            self.transaction.close()
            raise
        return self.result()

    def _open_outputs(self) -> None:
        self.handles = self.transaction.open()
        source_metadata = self.transaction.source_metadata
        self.input_metadata = source_metadata["input"]
        self.rules_metadata = source_metadata.get("protective-rule taxonomy")
        self.duplicate_writer = self._open_tsv_writer("duplicates", DUPLICATE_FIELDS)
        self.identity_writer = self._open_tsv_writer("identity", IDENTITY_FIELDS)
        self.price_writer = self._open_tsv_writer(
            "prices", ("source_index", "input_smiles", *self.price_columns)
        )
        self.collision_writer = self._open_tsv_writer("collisions", COLLISION_FIELDS)
        self.stereo_writer = self._open_tsv_writer("stereo", STEREO_FIELDS)
        if self.config.write_audit_files:
            self.handles["fallback.tsv"].write(_FALLBACK_HEADER)
            self.handles["errors.tsv"].write(_ERROR_HEADER)

    def _log(self, level: str, message: str) -> None:
        handle = self.handles.get("run.log")
        if handle is not None and not handle.closed:
            handle.write(f"{utc_now()} {level} {message}\n")
            handle.flush()

    def _write_progress_log(self, progress: Any) -> None:
        processed = self.counts["input_records"]
        if processed < self.next_progress:
            return
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        rate = processed / elapsed
        progress.set_postfix(
            output=self.counts["primary_rows"],
            errors=self.counts["processing_errors"],
            refresh=False,
        )
        self._log(
            "INFO",
            f"processed={processed}/{self.total_records} "
            f"success={self.counts['successful_input_records']} "
            f"errors={self.counts['processing_errors']} "
            f"output={self.counts['primary_rows']} "
            f"rate={rate:.1f} records/s",
        )
        self.next_progress += self.progress_every

    def _open_tsv_writer(
        self, key: str, fieldnames: tuple[str, ...]
    ) -> csv.DictWriter | None:
        handle = self.handles.get(key)
        if handle is None:
            return None
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        return writer

    def consume(self, processed: _ProcessedRecord) -> None:
        """Write one ordered worker result into the run's reports and indexes."""
        self.counts["input_records"] += 1
        self._write_price(processed.source)
        if processed.failed:
            self._write_failure(processed)
            return
        self.counts["successful_input_records"] += 1
        if (
            self.config.deprotect
            and processed.protected_smiles not in self.seen_protected
        ):
            self.handles["protected"].write(processed.protected_smiles + "\n")
            self.seen_protected.add(processed.protected_smiles)
            self.counts["protected_rows"] += 1

        candidate_status: dict[tuple[str, str], str] = {}
        for origin, candidate_smiles, stereo_free in self._candidates(processed):
            status = self._write_candidate(
                processed, origin, candidate_smiles, stereo_free
            )
            candidate_status[(origin, candidate_smiles)] = status

        if self.identity_writer is None:
            return
        identity_candidates = [
            (
                "protected",
                processed.protected_molecule,
                processed.protected_smiles,
            )
        ]
        if self.config.deprotect and processed.deprotection_changed:
            identity_candidates.append(
                (
                    "deprotected",
                    processed.deprotected_molecule,
                    processed.deprotected_smiles or "",
                )
            )
        for origin, candidate_molecule, candidate_smiles in identity_candidates:
            if candidate_molecule is None:
                raise RuntimeError("processed candidate has no canonical molecule")
            self._write_identity(
                processed,
                origin,
                candidate_molecule,
                candidate_smiles,
                candidate_status.get((origin, candidate_smiles), "synthon_only"),
            )

    def _write_price(self, source: ChemicalRecord) -> None:
        if self.price_writer is None:
            return
        metadata = dict(zip(source.metadata_names, source.metadata, strict=True))
        self.price_writer.writerow(
            {
                "source_index": source.sequence,
                "input_smiles": source.chemistry,
                **{column: metadata.get(column, "") for column in self.price_columns},
            }
        )
        self.counts["price_rows"] += 1

    def _write_failure(self, processed: _ProcessedRecord) -> None:
        source = processed.source
        if self.identity_writer is not None:
            detail = ": ".join(
                part
                for part in (
                    processed.error_stage,
                    processed.error_type,
                    processed.error_message,
                )
                if part
            )
            self.identity_writer.writerow(
                asdict(
                    IdentityReportRow(
                        source_index=source.sequence,
                        input_smiles=source.chemistry,
                        canonical_smiles="",
                        standard_inchi="",
                        standard_inchikey="",
                        inchi_return_code="",
                        inchi_warnings="",
                        output_origin="input",
                        status="processing_error",
                        note=detail,
                    )
                )
            )
            self.counts["identity_rows"] += 1
        self.counts["processing_errors"] += 1
        if self.config.write_audit_files:
            self.handles["fallback.tsv"].write(
                f"{tsv_field(source.chemistry)}\t{tsv_field(source.source_info)}\t"
                f"processing_error\t{tsv_field(processed.error_message)}\n"
            )
            self.handles["errors.tsv"].write(
                f"{tsv_field(source.chemistry)}\t{tsv_field(source.source_info)}\t"
                f"{tsv_field(processed.error_stage)}\t"
                f"{tsv_field(processed.error_type)}\t"
                f"{tsv_field(processed.error_message)}\n"
            )

    def _candidates(self, processed: _ProcessedRecord) -> list[tuple[str, str, str]]:
        protected = (
            "protected",
            processed.protected_smiles,
            processed.protected_stereo_free,
        )
        if not self.config.deprotect or not processed.deprotection_changed:
            return [protected]
        deprotected = (
            "deprotected",
            processed.deprotected_smiles or processed.protected_smiles,
            processed.deprotected_stereo_free or processed.protected_stereo_free,
        )
        if self.config.deprotect_output == "replace":
            return [deprotected]
        return [protected, deprotected]

    def _write_candidate(
        self,
        processed: _ProcessedRecord,
        origin: str,
        candidate_smiles: str,
        stereo_free: str,
    ) -> str:
        source = processed.source
        previous = self.seen_primary.get(candidate_smiles)
        if previous is None:
            self.handles["primary"].write(candidate_smiles + "\n")
            self.seen_primary[candidate_smiles] = (source.sequence, origin)
            self.counts["primary_rows"] += 1
            status = "written"
        else:
            status = "duplicate_skipped"
            if self.duplicate_writer is not None:
                self.duplicate_writer.writerow(
                    asdict(
                        DuplicateReportRow(
                            duplicate_kind=(
                                "deprotection_collapse"
                                if origin == "deprotected"
                                else "canonical_duplicate"
                            ),
                            canonical_smiles=candidate_smiles,
                            first_source_index=previous[0],
                            duplicate_source_index=source.sequence,
                            first_origin=previous[1],
                            duplicate_origin=origin,
                        )
                    )
                )
                self.counts["duplicate_rows"] += 1
        if self.stereo_writer is not None:
            self.stereo_writer.writerow(
                asdict(
                    StereoReportRow(
                        source_index=source.sequence,
                        input_smiles=source.chemistry,
                        canonical_smiles=candidate_smiles,
                        stereo_free_smiles=stereo_free,
                        stereo_present=candidate_smiles != stereo_free,
                        output_origin=origin,
                        status=status,
                        note=(
                            "no_protective_group"
                            if self.config.deprotect
                            and not processed.deprotection_changed
                            else ""
                        ),
                    )
                )
            )
            self.counts["stereo_rows"] += 1
        return status

    def _identity_provenance(
        self, processed: _ProcessedRecord, origin: str
    ) -> dict[str, str]:
        fields = {"standardized_input_smiles": processed.protected_smiles}
        if origin != "deprotected":
            return fields
        rules_hash = str((self.rules_metadata or {}).get("sha256", ""))
        return {
            **fields,
            "deprotection_policy": self.config.deprotect_policy,
            "protective_rules_sha256": rules_hash,
            "deprotection_events": compact_json(
                [event.as_dict() for event in processed.deprotection_events]
            ),
            "mapped_deprotection": processed.mapped_deprotection,
        }

    def _write_identity(
        self,
        processed: _ProcessedRecord,
        origin: str,
        candidate_molecule: MoleculeContainer,
        candidate_smiles: str,
        status: str,
    ) -> None:
        if self.identity_writer is None:
            return
        source = processed.source
        note = (
            "no_protective_group"
            if self.config.deprotect and not processed.deprotection_changed
            else ""
        )
        try:
            identity = molecule_identity(candidate_molecule)
        except Exception as error:
            self.counts["partial_identity_errors"] += 1
            self.identity_writer.writerow(
                asdict(
                    IdentityReportRow(
                        source_index=source.sequence,
                        input_smiles=source.chemistry,
                        canonical_smiles=candidate_smiles,
                        standard_inchi="",
                        standard_inchikey="",
                        inchi_return_code="",
                        inchi_warnings="",
                        output_origin=origin,
                        status="identity_error",
                        note=f"{type(error).__name__}: {error}",
                        **self._identity_provenance(processed, origin),
                    )
                )
            )
            self.counts["identity_rows"] += 1
            if self.config.write_audit_files:
                self.handles["errors.tsv"].write(
                    f"{tsv_field(source.chemistry)}\t"
                    f"{tsv_field(source.source_info)}\tidentity\t"
                    f"{type(error).__name__}\t{tsv_field(str(error))}\n"
                )
            return

        self.identity_writer.writerow(
            asdict(
                IdentityReportRow(
                    source_index=source.sequence,
                    input_smiles=source.chemistry,
                    canonical_smiles=candidate_smiles,
                    standard_inchi=identity.standard_inchi,
                    standard_inchikey=identity.inchi_key,
                    inchi_return_code=identity.return_code,
                    inchi_warnings="; ".join(identity.warnings),
                    output_origin=origin,
                    status=status,
                    note=note,
                    **self._identity_provenance(processed, origin),
                )
            )
        )
        self.counts["identity_rows"] += 1
        if status in {"written", "duplicate_skipped"}:
            self.identities[identity.inchi_key][candidate_smiles].append(
                (source.sequence, origin)
            )
        if status == "written" and identity.inchi_key not in self.seen_inchikey:
            self.handles["inchikey"].write(identity.inchi_key + "\n")
            self.seen_inchikey.add(identity.inchi_key)
            self.counts["inchikey_rows"] += 1

    def _write_collisions(self) -> None:
        if self.collision_writer is None:
            return
        collisions = {
            inchikey: structures
            for inchikey, structures in self.identities.items()
            if len(structures) > 1
        }
        source_indexes = {
            sequence
            for structures in collisions.values()
            for provenance in structures.values()
            for sequence, _origin in provenance
        }
        source_info = {
            record.sequence: record.source_info
            for record in self.reader
            if record.sequence in source_indexes
        }
        for inchikey, structures in collisions.items():
            for canonical_smiles, provenance in structures.items():
                self.collision_writer.writerow(
                    asdict(
                        CollisionReportRow(
                            standard_inchikey=inchikey,
                            canonical_smiles=canonical_smiles,
                            source_indexes=compact_json(
                                [item[0] for item in provenance]
                            ),
                            source_info=compact_json(
                                [source_info[item[0]] for item in provenance]
                            ),
                            output_origins=compact_json(
                                [item[1] for item in provenance]
                            ),
                        )
                    )
                )
                self.counts["collision_rows"] += 1

    def _write_completed_log(self) -> None:
        self._log(
            "INFO",
            f"completed input={self.counts['input_records']} "
            f"success={self.counts['successful_input_records']} "
            f"errors={self.counts['processing_errors']} "
            f"output={self.counts['primary_rows']}",
        )

    def _validate(self) -> None:
        self.transaction.validate_sources_unchanged(activity="preparation")
        if self.counts["input_records"] != self.total_records:
            raise RuntimeError(
                "processed input count does not match the framed source count: "
                f"{self.counts['input_records']} != {self.total_records}"
            )
        if (
            self.counts["successful_input_records"] + self.counts["processing_errors"]
            != self.counts["input_records"]
        ):
            raise RuntimeError("successful and failed records do not partition input")
        keys = set(self.transaction.partial_paths)
        expected = {"primary": self.counts["primary_rows"]}
        optional_counts = {
            "protected": self.counts["protected_rows"],
            "duplicates": self.counts["duplicate_rows"] + 1,
            "inchikey": self.counts["inchikey_rows"],
            "identity": self.counts["identity_rows"] + 1,
            "prices": self.counts["price_rows"] + 1,
            "collisions": self.counts["collision_rows"] + 1,
            "stereo": self.counts["stereo_rows"] + 1,
        }
        expected.update(
            {key: value for key, value in optional_counts.items() if key in keys}
        )
        if "fallback.smi" in keys:
            expected.update(
                {
                    "fallback.smi": 0,
                    "fallback.tsv": self.counts["processing_errors"] + 1,
                    "errors.tsv": self.counts["processing_errors"]
                    + self.counts["partial_identity_errors"]
                    + 1,
                    "run.log": 2 + self.counts["input_records"] // self.progress_every,
                }
            )
        self.transaction.validate_line_counts(expected)

    def _build_summary(self) -> dict[str, object]:
        if self.input_metadata is None:
            raise RuntimeError("preparation input provenance was not captured")
        artifacts = {
            self.transaction.final_paths[key].name: metadata
            for key, metadata in self.transaction.artifact_metadata().items()
        }
        return {
            "synplan_version": __version__,
            "schema_version": 2,
            "command": "prepare_building_blocks",
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - self.started,
            "input": self.input_metadata,
            "protective_rules": self.rules_metadata,
            "engines": {
                "rdkit": rdBase.rdkitVersion,
                "inchi": rdinchi.GetInchiVersion(),
            },
            "input_format": self.reader.input_format,
            "config": self.config.to_dict(),
            "ordered_execution": True,
            "resolved_num_workers": self.config.num_workers
            or default_num_workers(cap=8),
            "counts": dict(sorted(self.counts.items())),
            "output_files": artifacts,
        }

    def result(self) -> BuildingBlockPreparationResult:
        """Build the public result from the committed transaction paths."""
        final_paths = self.transaction.final_paths
        audit_files = {
            name: str(final_paths[name]) for name in _AUDIT_NAMES if name in final_paths
        }
        protected_path = final_paths.get("protected")
        return BuildingBlockPreparationResult(
            output_file=str(final_paths["primary"]),
            synthon_input=str(protected_path or final_paths["primary"]),
            protected_output_file=str(protected_path) if protected_path else None,
            inchikey_file=(
                str(final_paths["inchikey"]) if "inchikey" in final_paths else None
            ),
            identity_reference_file=(
                str(final_paths["identity"]) if "identity" in final_paths else None
            ),
            price_reference_file=(
                str(final_paths["prices"]) if "prices" in final_paths else None
            ),
            duplicates_file=(
                str(final_paths["duplicates"]) if "duplicates" in final_paths else None
            ),
            collisions_file=(
                str(final_paths["collisions"]) if "collisions" in final_paths else None
            ),
            stereo_file=(
                str(final_paths["stereo"]) if "stereo" in final_paths else None
            ),
            audit_files=audit_files,
            counts=dict(sorted(self.counts.items())),
        )


def prepare_building_blocks(
    input_file: str | Path,
    output_file: str | Path,
    config: BuildingBlockPreparationConfig | None = None,
) -> BuildingBlockPreparationResult:
    """Standardize, optionally deprotect, deduplicate, and audit building blocks."""
    return PreparationRun(input_file, output_file, config).run()


def standardize_building_blocks(input_file: str | Path, output_file: str | Path) -> str:
    """Compatibility wrapper for defaults-only building-block standardization."""
    return prepare_building_blocks(input_file, output_file).output_file


__all__ = [
    "BuildingBlockPreparationResult",
    "PreparationReader",
    "PreparationRun",
    "prepare_building_blocks",
    "standardize_building_blocks",
]
