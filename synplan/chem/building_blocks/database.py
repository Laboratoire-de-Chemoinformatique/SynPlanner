"""Build immutable SQLite stock once; keep chemistry out of prepared reloads."""

from __future__ import annotations

import csv
import functools
import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from frozendict import frozendict

from .core import BuildingBlock

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 2
# Bump when catalogue preparation/identity semantics change independently of Chython.
PREPARATION_VERSION = 2


class SQLiteBuildingBlockCatalogue(Mapping[str, tuple[BuildingBlock, ...]]):
    """Read-only stock; each thread opens its own connection on first lookup.

    Pass the database path to workers, which open their own catalogue instance.
    Prepare the cache before starting worker processes.
    Explicit iteration visits the entire catalogue; search uses indexed get().
    """

    def __init__(self, path: str | Path, expected_cache_id: str | None = None):
        self._path = Path(path).resolve(strict=True)
        self._local = threading.local()
        with closing(self._connect()) as connection:
            self._metadata = frozendict(
                json.loads(
                    connection.execute("SELECT json FROM metadata").fetchone()[0]
                )
            )
        if self._metadata.get("schema_version") not in (1, SCHEMA_VERSION):
            raise ValueError(f"{self._path}: unsupported building-block SQLite schema")
        if (
            expected_cache_id is not None
            and self._metadata["cache_id"] != expected_cache_id
        ):
            raise ValueError(f"{self._path}: building-block catalogue release changed")
        if self.record_count < 1 or len(self) < 1:
            raise ValueError(f"{self._path}: building-block catalogue is empty")
        self._record_sql = (
            "record"
            if self._metadata["schema_version"] >= 2
            else (
                "json_object('smiles',smiles,'vendors',json(vendors),'has_stereo',"
                "json(CASE has_stereo WHEN 1 THEN 'true' WHEN 0 THEN 'false' ELSE 'null' END))"
            )
        )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata(self) -> frozendict:
        return self._metadata

    @property
    def record_count(self) -> int:
        return self._metadata["records"]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute("PRAGMA mmap_size=0")
        return connection

    def _reader(self):
        if not hasattr(self._local, "connection"):
            connection = self._connect()
            metadata = json.loads(
                connection.execute("SELECT json FROM metadata").fetchone()[0]
            )
            if metadata["cache_id"] != self._metadata["cache_id"]:
                connection.close()
                raise ValueError(
                    f"{self._path}: building-block catalogue release changed"
                )
            self._local.connection = connection
            self._local.pid = os.getpid()

            @functools.lru_cache(maxsize=8192)
            def lookup(prefix):
                from .io import _validate_record

                return tuple(
                    _validate_record(key, json.loads(record), context=str(self._path))
                    for key, record in connection.execute(
                        f"SELECT inchikey, {self._record_sql} FROM blocks "
                        "WHERE inchikey>=? AND inchikey<? ORDER BY rowid",
                        (prefix + "-", prefix + "."),
                    )
                )

            self._local.lookup = lookup
        if self._local.pid != os.getpid():
            raise RuntimeError(
                "Reopen the building-block catalogue in the worker; use spawn instead of sharing a connection across fork"
            )
        return self._local

    def __getitem__(self, prefix: str) -> tuple[BuildingBlock, ...]:
        records = self._reader().lookup(prefix)
        if not records:
            raise KeyError(prefix)
        return records

    def __iter__(self) -> Iterator[str]:
        rows = self._reader().connection.execute(
            "SELECT substr(inchikey,1,14) FROM blocks "
            "GROUP BY substr(inchikey,1,14) ORDER BY min(rowid)"
        )
        return (row[0] for row in rows)

    def __len__(self) -> int:
        return self._metadata["buckets"]

    def records(self):
        """Stream grouped JSON records without loading stock or repeating chemistry."""
        from .io import _validate_record

        for key, records in self._reader().connection.execute(
            f"SELECT inchikey,json_group_array(json({self._record_sql})) FROM blocks "
            "GROUP BY inchikey ORDER BY min(rowid)"
        ):
            for record in json.loads(records):
                yield (
                    key,
                    _validate_record(key, record, context=str(self._path)).to_record(),
                )

    def close(self) -> None:
        """Release this thread's connection and bounded record cache."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            self._local.__dict__.clear()


def _signature(path: Path):
    stat = path.stat()
    return (
        str(path),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _source_metadata(path: Path) -> Path | None:
    metadata = path.with_name("meta.yaml")
    return metadata if metadata.is_file() else None


@functools.cache
def _versions():
    return SCHEMA_VERSION, PREPARATION_VERSION, version("chython-synplan")


def _source_token(path: Path):
    metadata = _source_metadata(path)
    return _signature(path), _signature(metadata) if metadata else None, _versions()


def _rows(records, *, context):
    from .io import _validate_record

    for key, record in records:
        block = _validate_record(key, record, context=context)
        yield (
            block.inchikey,
            json.dumps(block.to_record(), separators=(",", ":")),
            json.dumps(
                [block.stereo_type, block.smiles if "|" in block.smiles else ""]
            ),
        )


def build_catalogue(source: Path, output: Path, *, num_workers: int = 1) -> Path:
    """Stream validated JSON or ordered raw TSV batches to an atomic SQLite file.

    Prepared-file defects reject the artifact. Raw-row errors are reported while
    valid rows retain the first SMILES and minimum positive price per vendor.
    """
    from .io import (
        _atomic_output,
        _iter_prepared_catalogue,
        _merge_records,
        _prepare_catalogue_batches,
    )

    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if source == output:
        raise ValueError("input_file name and output_file name cannot be the same.")
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    prepared = source.name.lower().endswith((".json", ".json.gz"))
    if not prepared and not source.name.lower().endswith((".tsv", ".tsv.gz")):
        raise ValueError(
            "Catalogue input must be prepared JSON/JSON.GZ or raw vendor TSV/TSV.GZ"
        )
    before = _source_token(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    source_sha256 = digest.hexdigest()
    source_metadata = _source_metadata(source)
    metadata_text = (
        source_metadata.read_text(encoding="utf-8") if source_metadata else None
    )
    cache_id = hashlib.sha256(
        json.dumps([source_sha256, metadata_text, _versions()]).encode()
    ).hexdigest()
    logger.info("Preparing SQLite building blocks from %s", source)
    errors = output.with_name(output.name + ".errors.tsv")
    error_count = 0
    with _atomic_output(output) as temporary:
        error_temporary = temporary.with_name(errors.name)
        error_temporary.touch(mode=0o600)
        with (
            error_temporary.open("w", encoding="utf-8", newline="") as error_handle,
            closing(sqlite3.connect(temporary)) as connection,
            connection,
        ):
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute(
                "CREATE TABLE blocks (inchikey TEXT NOT NULL, record TEXT NOT NULL, variant TEXT NOT NULL, UNIQUE(inchikey, variant))"
            )
            insert = "INSERT INTO blocks VALUES (?, ?, ?)"
            writer = csv.writer(error_handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("line_number", "error"))
            if prepared:
                batches = ((_iter_prepared_catalogue(source), ()),)
            else:
                connection.create_function(
                    "merge_records",
                    2,
                    lambda old, new: json.dumps(
                        _merge_records(json.loads(old), json.loads(new)),
                        separators=(",", ":"),
                    ),
                )
                insert += " ON CONFLICT(inchikey,variant) DO UPDATE SET record=merge_records(blocks.record,excluded.record)"
                batches = _prepare_catalogue_batches(source, num_workers)
            try:
                for batch_records, batch_errors in batches:
                    writer.writerows(batch_errors)
                    error_count += len(batch_errors)
                    connection.executemany(
                        insert, _rows(batch_records, context=str(source))
                    )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"{source}: duplicate InChIKey") from error
            records, buckets = connection.execute(
                "SELECT count(*), count(DISTINCT substr(inchikey,1,14)) FROM blocks"
            ).fetchone()
            if not records and not prepared and not error_count:
                writer.writerow((1, "catalogue contains no data rows"))
                error_count = 1
            if records and _source_token(source) != before:
                raise ValueError(
                    f"{source}: source changed during catalogue preparation; retry"
                )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "preparation_version": PREPARATION_VERSION,
                "chython_version": _versions()[2],
                "cache_id": cache_id,
                "source_sha256": source_sha256,
                "source_path": str(source),
                "source_metadata": metadata_text,
                "records": records,
                "buckets": buckets,
                "rejected_rows": error_count,
            }
            connection.execute("CREATE TABLE metadata (json TEXT NOT NULL)")
            connection.execute(
                "INSERT INTO metadata VALUES (?)", (json.dumps(metadata),)
            )
        if error_count:
            os.replace(error_temporary, errors)
        if not records:
            raise ValueError(
                f"{source}: building-block catalogue is empty"
                if prepared
                else f"{source}: no valid rows; {error_count} invalid row(s) reported in {errors}"
            )
        # ponytail: competing first builds may repeat work. Prepare once before
        # launching tree workers; atomic publication exposes only complete files.
    if error_count:
        logger.warning(
            "Published %d building blocks after dropping %d invalid row(s); details written to %s",
            records,
            error_count,
            errors,
        )
    else:
        errors.unlink(missing_ok=True)
    logger.info("Prepared %d building blocks in %s", records, output)
    return output


@functools.lru_cache(maxsize=8)
def _open_catalogue(path: Path, signature: tuple, pid: int):
    return SQLiteBuildingBlockCatalogue(path)


def load_building_block_catalogue(
    path_value: str | Path, *, num_workers: int = 1, cache_dir: str | Path | None = None
) -> SQLiteBuildingBlockCatalogue:
    """Open SQLite stock, importing prepared JSON or raw vendor TSV once.

    Cache files live in XDG_CACHE_HOME/synplanner/building_blocks (by default
    ~/.cache). Source changes and preparation/schema versions create a new
    immutable snapshot. Prepared JSON never repeats chemistry.
    """
    path = Path(path_value).resolve(strict=True)
    if path.suffix.lower() == ".sqlite":
        destination = path
    else:
        root = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "synplanner"
            / "building_blocks"
        )
        token = hashlib.sha256(json.dumps(_source_token(path)).encode()).hexdigest()
        destination = root / f"{token}.sqlite"
        if not destination.exists():
            if path.name.lower().endswith(".sqlite.gz"):
                from .io import _atomic_output

                with _atomic_output(destination) as temporary:
                    with (
                        gzip.open(path, "rb") as source,
                        temporary.open("wb") as output,
                    ):
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    SQLiteBuildingBlockCatalogue(temporary).close()
            else:
                build_catalogue(path, destination, num_workers=num_workers)
    try:
        return _open_catalogue(destination, _signature(destination), os.getpid())
    except (sqlite3.Error, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read building-block catalogue {destination}: {error}"
        ) from error
