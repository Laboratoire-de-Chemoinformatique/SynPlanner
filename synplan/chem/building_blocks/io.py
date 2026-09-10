"""Building-block format dispatch, loading and preparation."""

from __future__ import annotations

import contextlib
import csv
import functools
import gzip
import json
import logging
import math
import os
import re
import shutil
import tempfile
from contextlib import closing, contextmanager
from decimal import Decimal
from io import TextIOWrapper
from itertools import groupby
from pathlib import Path
from typing import Any

import ijson
import yaml
from chython import inchi_key, smiles
from chython.containers import MoleculeContainer
from chython.files.SDFrw import SDFRead
from frozendict import frozendict
from tqdm.auto import tqdm

from synplan.chem.stereo import has_stereo, has_stereo_groups
from synplan.chem.utils import (
    safe_canonicalization,
    standardize_sdf_text,
    standardize_smiles_batch,
)
from synplan.utils.files import (
    MoleculeReader,
    MoleculeWriter,
    count_sdf_records,
    count_smiles_records,
    iter_csv_smiles,
    iter_csv_smiles_blocks,
    iter_sdf_text_blocks,
    iter_smiles,
    iter_smiles_blocks,
)
from synplan.utils.parallel import chunked, process_pool_map_stream

from .core import BuildingBlock, BuildingBlockCatalogue
from .database import build_catalogue, load_building_block_catalogue

logger = logging.getLogger(__name__)


def load_building_blocks(
    building_blocks_path: str | Path,
    standardize: bool = True,
    silent: bool = True,
    num_workers: int | None = None,
    chunksize: int = 1000,
    *,
    header: bool = True,
    delimiter: str = ",",
    smiles_column: str = "SMILES",
) -> frozenset[str] | BuildingBlockCatalogue:
    """Load molecular stock or a cached stereo-aware vendor catalogue.

    Prepared catalogues retain their InChIKeys and vendor offers and skip
    chemistry preparation, regardless of ``standardize``.
    JSON/JSON.GZ is indexed into SQLite once; vendor TSV/TSV.GZ first performs
    chemistry preparation. SQLite files open directly with bounded reader caches.

    :param building_blocks_path: The path to the file containing the building blocks.
    :param standardize: Flag if building blocks have to be standardized before loading. Default=True.
    :param header: For CSV/CSV.GZ files: treat the first row as header. Default=True.
    :param delimiter: For CSV/CSV.GZ files: delimiter character. Default=",".
    :param smiles_column: For CSV/CSV.GZ files: header column name containing SMILES.
        Default="SMILES" (case-insensitive match is supported).
    :return: A read-only catalogue for JSON/SQLite/vendor TSV, otherwise a SMILES set.
    """

    building_blocks_path = Path(building_blocks_path).resolve()
    suffixes = "".join(building_blocks_path.suffixes).lower()
    vendor_tsv = False
    if header and suffixes.endswith((".tsv", ".tsv.gz")):
        opener = gzip.open if suffixes.endswith(".gz") else open
        with opener(building_blocks_path, "rt", encoding="utf-8", newline="") as stream:
            columns = next(csv.reader(stream, delimiter="\t"), ())
        vendor_tsv = "sources" in columns or any(
            column.casefold().endswith("_ppg") for column in columns
        )
    if suffixes.endswith((".json", ".json.gz", ".sqlite", ".sqlite.gz")) or vendor_tsv:
        return load_building_block_catalogue(
            building_blocks_path, num_workers=1 if num_workers is None else num_workers
        )
    return _load_smiles_building_blocks(
        building_blocks_path,
        standardize,
        silent,
        num_workers,
        chunksize,
        header=header,
        delimiter=delimiter,
        smiles_column=smiles_column,
    )


@functools.cache
def _load_smiles_building_blocks(
    building_blocks_path,
    standardize,
    silent,
    num_workers,
    chunksize,
    *,
    header,
    delimiter,
    smiles_column,
):
    suffixes = "".join(building_blocks_path.suffixes).lower()
    is_csv = suffixes.endswith(".csv") or suffixes.endswith(".csv.gz")
    is_tsv = suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz")
    if is_tsv:
        is_csv = True
        delimiter = "\t"
    suffix = building_blocks_path.suffix.lower()
    if not is_csv and suffix not in {".smi", ".smiles", ".sdf"}:
        raise ValueError(
            f"Unsupported building blocks file extension: '{building_blocks_path.name}'. "
            "Supported: .smi, .smiles, .sdf, .csv, .csv.gz, .tsv, .tsv.gz, .json, .json.gz"
        )

    building_blocks_smiles = set()
    if standardize:
        if num_workers is None:
            num_workers = max(1, os.cpu_count() - 1)
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")

        if suffix in {".smi", ".smiles"}:
            total = count_smiles_records(building_blocks_path) if not silent else None
            step = max(1, chunksize or 1000)

            progress_iter = _building_blocks_progress(total, silent=silent)
            for out in _map_blocks(
                iter_smiles_blocks(building_blocks_path, step),
                standardize_smiles_batch,
                num_workers=num_workers,
            ):
                if out:
                    building_blocks_smiles.update(out)
                    if progress_iter is not None:
                        progress_iter.update(len(out))
            if progress_iter is not None:
                progress_iter.close()

        elif is_csv:
            step = max(1, chunksize or 1000)
            progress_iter = _building_blocks_progress(None, silent=silent)
            blocks = iter_csv_smiles_blocks(
                building_blocks_path,
                step,
                header=header,
                delimiter=delimiter,
                smiles_column=smiles_column,
            )
            for out in _map_blocks(
                blocks, standardize_smiles_batch, num_workers=num_workers
            ):
                if out:
                    building_blocks_smiles.update(out)
                    if progress_iter is not None:
                        progress_iter.update(len(out))
            if progress_iter is not None:
                progress_iter.close()

        elif suffix == ".sdf":
            n = count_sdf_records(building_blocks_path) if not silent else None
            step = max(1, chunksize or 5000)
            blocks = iter_sdf_text_blocks(building_blocks_path, step)

            progress = _building_blocks_progress(n, silent=silent)
            for chunk_out in _map_blocks(
                blocks, standardize_sdf_text, num_workers=num_workers
            ):
                if chunk_out:
                    building_blocks_smiles.update(chunk_out)
                    if progress is not None:
                        progress.update(len(chunk_out))
            if progress is not None:
                progress.close()
    else:
        if suffix in {".smi", ".smiles"}:
            for smiles in iter_smiles(building_blocks_path):
                building_blocks_smiles.add(smiles)
        elif is_csv:
            for smiles in iter_csv_smiles(
                building_blocks_path,
                header=header,
                delimiter=delimiter,
                smiles_column=smiles_column,
            ):
                building_blocks_smiles.add(smiles)
        elif suffix == ".sdf":
            with SDFRead(str(building_blocks_path)) as sdf:
                for mol in sdf:
                    with contextlib.suppress(Exception):
                        building_blocks_smiles.add(str(mol))

    return frozenset(building_blocks_smiles)


def standardize_building_blocks(
    input_file: str, output_file: str, *, num_workers: int = 1
) -> str:
    """Standardizes custom building blocks.

    :param input_file: The path to the file that stores the original building blocks.
    :param output_file: The path to the file that will store the standardized building
        blocks.
    :param num_workers: Worker processes for TSV to JSON/JSON.GZ/SQLite preparation.
    :return: The path to the file with standardized building blocks.
    """
    if input_file == output_file:
        raise ValueError("input_file name and output_file name cannot be the same.")

    if Path(output_file).suffix.lower() == ".sqlite":
        return str(
            build_catalogue(
                Path(input_file), Path(output_file), num_workers=num_workers
            )
        )

    if Path(output_file).name.lower().endswith((".json", ".json.gz")):
        return standardize_building_block_catalogue(
            input_file, output_file, num_workers=num_workers
        )

    with (
        MoleculeReader(input_file) as inp_file,
        MoleculeWriter(output_file) as out_file,
    ):
        for mol in tqdm(
            inp_file,
            desc="Number of building blocks processed: ",
            bar_format="{desc}{n} [{elapsed}]",
        ):
            try:
                mol = safe_canonicalization(mol)
            except Exception as e:
                logging.debug(e)
                continue
            out_file.write(mol)

    return output_file


def _building_blocks_progress(total: int | None, *, silent: bool):
    """Create a consistent progress bar for building blocks loading."""
    if silent:
        return None
    return tqdm(
        total=total,
        desc="Building blocks",
        unit="mol",
        unit_scale=True,
        unit_divisor=1000,
        dynamic_ncols=True,
        smoothing=0.1,
        disable=silent,
    )


def _map_blocks(blocks, worker_fn, *, num_workers: int):
    """Map blocks through worker function, optionally using a process pool.

    For `num_workers == 1`, this runs sequentially to avoid process-spawn overhead.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if num_workers == 1:
        for block in blocks:
            yield worker_fn(block)
        return
    yield from process_pool_map_stream(blocks, worker_fn, max_workers=num_workers)


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


def _write_json_atomic(path: Path, records) -> None:
    with _atomic_output(path) as temporary, temporary.open("wb") as handle:
        stream = (
            gzip.GzipFile(
                fileobj=handle, mode="wb", filename="", mtime=0, compresslevel=6
            )
            if path.suffix.lower() == ".gz"
            else handle
        )
        with TextIOWrapper(stream, encoding="utf-8", newline="\n") as text:
            text.write("{")
            for index, (key, group) in enumerate(
                groupby(records, key=lambda item: item[0])
            ):
                values = [
                    _validate_record(key, value, context=str(path)).to_record()
                    for _, value in group
                ]
                if index:
                    text.write(",")
                text.write(json.dumps(key) + ":")
                json.dump(
                    values[0] if len(values) == 1 else values,
                    text,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            text.write("}\n")


def _merge_records(existing, incoming):
    """Keep the first spelling and distinct source offers."""
    existing["has_stereo"] |= incoming["has_stereo"]
    if incoming.get("sources"):
        sources = existing.setdefault("sources", [])
        sources.extend(
            source for source in incoming["sources"] if source not in sources
        )
    return existing


def standardize_building_block_catalogue(
    input_file: str | Path, output_file: str | Path, *, num_workers: int = 1
) -> str:
    """Prepare once through SQLite, then stream JSON with bounded memory."""

    source = Path(input_file)
    output = Path(output_file)
    if source.resolve() == output.resolve():
        raise ValueError("input_file name and output_file name cannot be the same.")
    if not output.name.lower().endswith((".json", ".json.gz")):
        raise ValueError("building-block catalogue output must use .json or .json.gz")
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    error_path = Path(f"{output}.errors.tsv")
    with tempfile.TemporaryDirectory() as folder:
        database = (
            source
            if source.suffix.lower() == ".sqlite"
            else Path(folder) / "stock.sqlite"
        )
        try:
            if database != source:
                build_catalogue(source, database, num_workers=num_workers)
            with closing(load_building_block_catalogue(database)) as stock:
                _write_json_atomic(output, stock.records())
            error_path.unlink(missing_ok=True)
        finally:
            errors = Path(f"{database}.errors.tsv")
            if errors.exists():
                error_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(errors, error_path)
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
        if not price_columns and "sources" not in header:
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


def mcule_to_cxsmiles(smiles_text: str, stereo_type: str) -> tuple[str, str]:
    """Return canonical CXSMILES and the Mcule material declaration; retain both.

    REL/RAC link all depicted tetrahedral centres in one OR/AND group. E/Z stays
    unchanged. UNK is not an OR group, and RAC's 1:1 ratio requires the declaration.
    Plain SMILES cannot recover the distinction between wavy and unmarked centres
    in a source drawing; unassigned centres are never assigned here.
    """
    kind = stereo_type.strip().lower().removesuffix(" (not confirmed)")
    kind = {
        "abs": "absolute",
        "rel": "relative",
        "rac": "racemic",
        "unk": "unknown",
    }.get(kind, kind)
    if kind not in ("", "absolute", "relative", "racemic", "unknown", "unspecified"):
        raise ValueError(f"Unknown Mcule stereo type: {stereo_type!r}")
    if re.search(r"(?:\||,)r(?=[:,|])", smiles_text):
        raise ValueError(
            "Legacy CX r needs separate assessment; it is not a Mcule stereo type"
        )
    molecule = smiles(smiles_text, strict_stereo=True)
    if not isinstance(molecule, MoleculeContainer):
        raise ValueError("Mcule SMILES must describe a molecule")
    if not kind and has_stereo(molecule):
        kind = (
            "unknown"  # Missing export metadata is not confirmation of absolute stock.
        )
    text = str(molecule)
    group = {"relative": "o", "racemic": "&"}.get(kind)
    centres = {n for n, atom in molecule.atoms() if atom.stereo is not None}
    if has_stereo_groups(molecule):
        groups = {molecule.atom(n).extended_stereo for n in centres}
        if kind == "absolute" or (
            group
            and (
                None in groups
                or len(groups) != 1
                or (next(iter(groups)) < 0) != (group == "o")
            )
        ):
            raise ValueError(
                "Existing CX groups conflict with the Mcule material declaration"
            )
        return text, kind
    if group and centres:
        if not centres <= molecule.stereogenic_tetrahedrons.keys():
            raise ValueError(
                "Mcule REL/RAC conversion supports tetrahedral centres only"
            )
        indices = ",".join(
            str(i) for i, n in enumerate(molecule.smiles_atoms_order) if n in centres
        )
        annotation = f"{group}1:{indices}"
        text = (
            f"{text[:-1]},{annotation}|"
            if text.endswith("|")
            else f"{text} |{annotation}|"
        )
        text = str(smiles(text, strict_stereo=True))
    return text, kind


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

            sources = json.loads(row["sources"]) if row.get("sources") else []
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
                vendor = column[: -len("_ppg")]
                if price > 0.0 and not any(
                    s.get("vendor") == vendor and float(s.get("ppg") or 0) == price
                    for s in sources
                ):
                    sources.append({"vendor": vendor, "ppg": str(price)})

            molecule = smiles(raw_smiles, strict_stereo=True)
            if not isinstance(molecule, MoleculeContainer):
                raise ValueError("SMILES does not describe one molecule")
            molecule = safe_canonicalization(molecule, clean_stereo=False)
            smiles_text = str(molecule)
            key = inchi_key(molecule)
            restored = smiles(smiles_text, strict_stereo=True)
            if inchi_key(restored) != key:
                raise ValueError("prepared SMILES changes identity when read back")
            record = {
                "smiles": smiles_text,
                "sources": sources,
                "has_stereo": has_stereo(molecule),
            }
            if row.get("stereo_type"):
                record["stereo_type"] = row["stereo_type"].removesuffix(
                    " (not confirmed)"
                )
            # The pinned reader ignores the legacy molecule-level CX r flag.
            # Keep its supplier ID, but never promote it to absolute stock.
            if re.search(r"(?:\||,)r(?=[:,|])", raw_smiles):
                if not record.get("sources"):
                    raise ValueError(
                        "CX r requires source metadata for unassessed stock"
                    )
                record["stereo_type"] = "unknown"
            record = _validate_record(
                key, record, context=f"row {line_number}"
            ).to_record()
            records.append((key, record))
        except Exception as error:
            errors.append((line_number, str(error) or type(error).__name__))
    return records, errors


def _validate_record(key, raw_record, *, context):
    """Validate stored fields without repeating molecular parsing or identity work."""
    if not isinstance(key, str) or not re.fullmatch(r"[A-Z]{14}-[A-Z]{8}SA-[A-Z]", key):
        raise ValueError(f"{context}: invalid Standard InChIKey {key!r}")
    location = f"{context}:{key}"
    if not isinstance(raw_record, dict):
        raise ValueError(f"{location}: record must be a JSON object")
    required = {"smiles", "has_stereo"}
    if (
        not required <= raw_record.keys()
        or raw_record.keys()
        - required
        - {
            "vendors",
            "sources",
            "stereo_type",
        }
        or not {"vendors", "sources"} & raw_record.keys()
    ):
        raise ValueError(
            f"{location}: expected smiles, has_stereo, and vendors or sources fields"
        )
    canonical_smiles = raw_record["smiles"]
    if not isinstance(canonical_smiles, str) or not canonical_smiles:
        raise ValueError(f"{location}: smiles must be a non-empty string")
    has_stereo = raw_record["has_stereo"]
    if not isinstance(has_stereo, bool):
        raise ValueError(f"{location}: has_stereo must be boolean")
    stereo_type = raw_record.get("stereo_type", "")
    if stereo_type not in (
        "",
        "absolute",
        "relative",
        "racemic",
        "unknown",
        "unspecified",
    ):
        raise ValueError(f"{location}: invalid stereo_type")
    sources = raw_record.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError(f"{location}: sources must be a list")
    offers = []
    for source in sources:
        if (
            not isinstance(source, dict)
            or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in source.items()
            )
            or not source.get("vendor")
            or not (source.get("id") or source.get("ppg"))
        ):
            raise ValueError(
                f"{location}: each source needs vendor and id strings or a price"
            )
        source = {
            k: v
            for k, v in source.items()
            if k not in {"smiles", "url", "lead_time", "availability"}
        }
        if source.get("ppg"):
            price = float(source["ppg"])
            if not math.isfinite(price) or price < 0:
                raise ValueError(
                    f"{location}: source price must be finite and non-negative"
                )
            if price == 0:
                source.pop("ppg")
                if not source.get("id"):
                    continue
        offers.append(source)
    # Read old catalogue prices once into source offers; never retain the summary.
    legacy_prices = raw_record.get("vendors", {})
    if not isinstance(legacy_prices, dict):
        raise ValueError(f"{location}: vendors must be a JSON object")
    for vendor, raw_price in legacy_prices.items():
        if not isinstance(vendor, str) or not vendor:
            raise ValueError(f"{location}: vendor names must be non-empty strings")
        if isinstance(raw_price, bool) or not isinstance(
            raw_price, (int, float, Decimal)
        ):
            raise ValueError(f"{location}: {vendor} price must be numeric")
        price = float(raw_price)
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"{location}: {vendor} price must be finite and positive")
        if not any(
            s["vendor"] == vendor and s.get("ppg") and float(s["ppg"]) == price
            for s in offers
        ):
            offers.append({"vendor": vendor, "ppg": str(price)})
    return BuildingBlock(
        smiles=canonical_smiles,
        inchikey=key,
        has_stereo=has_stereo,
        sources=tuple(frozendict(source) for source in offers),
        stereo_type=stereo_type,
    )


def _iter_prepared_catalogue(path: Path):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rb") as stream:
            for key, record in ijson.kvitems(stream, ""):
                if isinstance(record, list):
                    if not record:
                        raise ValueError(f"{path}: empty material variants for {key}")
                    yield from ((key, item) for item in record)
                else:
                    yield key, record
    except (OSError, EOFError, ijson.JSONError) as error:
        raise ValueError(
            f"Could not read building-block catalogue {path}: {error}"
        ) from error


def vendor_names(catalogue: Any) -> dict[str, str]:
    """``{vendor code: trading name}`` read from a catalogue's release metadata.

    Only a prepared catalogue carries the ``meta.yaml`` its release was built
    with. Anything else answers empty, which leaves the caller showing the codes
    the records themselves hold.
    """

    metadata = getattr(catalogue, "metadata", None) or {}
    try:
        source = yaml.safe_load(metadata.get("source_metadata") or "") or {}
    except yaml.YAMLError:
        return {}
    vendors = source.get("vendors") if isinstance(source, dict) else None
    if not isinstance(vendors, dict):
        return {}
    return {
        str(code): str(entry["name"])
        for code, entry in vendors.items()
        if isinstance(entry, dict) and entry.get("name")
    }


__all__ = [
    "load_building_block_catalogue",
    "load_building_blocks",
    "standardize_building_block_catalogue",
    "standardize_building_blocks",
    "vendor_names",
]
