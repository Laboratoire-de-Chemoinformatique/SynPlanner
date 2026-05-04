"""Shared serialization and write utilities for reaction processing pipelines.

Workers call build_batch_result to produce a BatchResult with pre-serialized
records so that no ReactionContainer crosses the IPC boundary.  The parent
calls write_batch_results to consume BatchResult objects and write to disk.
"""

from collections.abc import Callable, Iterable
from io import StringIO

from chython.containers import ReactionContainer
from chython.files.RDFrw import RDFWrite

from synplan.chem.data.reaction_result import (
    BatchResult,
    ErrorEntry,
    FilteredEntry,
    PipelineSummary,
)
from synplan.utils.files import to_reaction_smiles_record


def reaction_cgr_key(rxn: ReactionContainer) -> str | None:
    """Stable cross-process deduplication key: canonical CGR SMILES.

    Uses str(~rxn) rather than hash(~rxn) because Python's built-in hash()
    is randomised per process (PYTHONHASHSEED), so the same reaction produces
    different integers in different workers — breaking cross-batch dedup.
    """
    try:
        return str(~rxn)
    except Exception:
        return None


def serialize_reaction(rxn: ReactionContainer, fmt: str) -> str:
    """Serialize a ReactionContainer to a string in the given output format.

    ``SMI`` output returns a tab-separated SMILES record with meta fields.
    ``RDF`` output returns one RDF record block as a string without the file
    header or footer, produced via ``StringIO`` and ``RDFWrite(append=True)``.

    Called inside worker processes so serialization runs in parallel.
    The parent only writes the returned string to disk.
    """
    if fmt == "smi":
        return to_reaction_smiles_record(rxn)
    if fmt == "rdf":
        buf = StringIO()
        with RDFWrite(buf, append=True) as w:
            w.write(rxn)
        return buf.getvalue()
    raise ValueError(f"Unsupported output format: {fmt!r}")


def build_batch_result(
    reactions: list[ReactionContainer],
    errors: list[ErrorEntry],
    filtered: list[FilteredEntry],
    fmt: str,
    compute_dedup: bool = True,
) -> BatchResult:
    """Convert a list of processed reactions into a BatchResult ready for IPC.

    Serializes each reaction in-worker so the parent receives only strings.
    """
    records: list[str] = []
    dedup_keys: list[str | None] = []
    for rxn in reactions:
        records.append(serialize_reaction(rxn, fmt))
        dedup_keys.append(reaction_cgr_key(rxn) if compute_dedup else None)
    return BatchResult(
        records=records,
        dedup_keys=dedup_keys,
        filtered=filtered,
        errors=errors,
    )


def write_batch_results(
    results: Iterable[BatchResult],
    write_fn: Callable[[str], None],
    summary: PipelineSummary,
    error_file=None,
    dedup: bool = False,
    seen: set | None = None,
    on_batch: Callable[[int], None] | None = None,
) -> None:
    """Consume BatchResult objects and write records via write_fn.

    Handles deduplication, error/filter logging, and summary accounting.
    The caller owns the file handles, summary object, and progress bar.

    :param results: Iterable of BatchResult objects from workers.
    :param write_fn: Callable that accepts a single pre-serialized record
        string and writes it to the output (e.g. writer.write_string).
    :param summary: PipelineSummary updated in-place.
    :param error_file: Optional open text file handle for error rows.
    :param dedup: If True, skip records whose dedup key was seen before.
    :param seen: Existing dedup set; a new set is created if None.
    :param on_batch: Optional callback called with the batch total count
        after each batch (used for progress bar updates).
    """
    if seen is None:
        seen = set()
    for batch in results:
        for record, key in zip(batch.records, batch.dedup_keys):
            if dedup:
                if key is None:
                    raise ValueError(
                        "Deduplication requires worker-computed dedup keys. "
                        "Build BatchResult with compute_dedup=True or disable "
                        "deduplication for this pipeline."
                    )
                h = key
                if h in seen:
                    summary.duplicates += 1
                    continue
                seen.add(h)
            write_fn(record)
            summary.succeeded += 1
        for err in batch.errors:
            err_key = f"{err.stage}/{err.error_type}"
            summary.error_breakdown[err_key] = (
                summary.error_breakdown.get(err_key, 0) + 1
            )
            summary.errored += 1
            if error_file is not None:
                error_file.write(
                    f"{err.original}\t{err.stage}\t{err.error_type}\t{err.message}\n"
                )
        for flt in batch.filtered:
            summary.filter_breakdown[flt.reason] = (
                summary.filter_breakdown.get(flt.reason, 0) + 1
            )
            summary.filtered += 1
        batch_total = len(batch.records) + len(batch.errors) + len(batch.filtered)
        summary.total_input += batch_total
        if on_batch is not None:
            on_batch(batch_total)
