"""Module containing classes and functions needed for reactions/molecules data
reading/writing."""

import contextlib
import csv
import gzip
from collections.abc import Iterable, Iterator
from io import StringIO
from os.path import splitext
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from chython import smiles
from chython.containers import CGRContainer, MoleculeContainer, ReactionContainer
from chython.exceptions import MappingError
from chython.files.RDFrw import RDFRead, RDFWrite
from chython.files.SDFrw import SDFRead, SDFWrite

if TYPE_CHECKING:
    from synplan.chem.utils import AtomMappingCheck


class FileHandler:
    """General class to handle chemical files."""

    def __init__(self, filename: str | Path, **kwargs):
        """General class to handle chemical files.

        :param filename: The path and name of the file.
        :return: None.
        """
        self._file = None
        _, ext = splitext(filename)
        file_types = {
            ".smi": "SMI",
            ".smiles": "SMI",
            ".rdf": "RDF",
            ".sdf": "SDF",
            ".pb": "PB",
        }
        try:
            self._file_type = file_types[ext]
        except KeyError as e:
            raise ValueError("I don't know the file extension,", ext) from e

    def close(self):
        self._file.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Reader(FileHandler):
    def __init__(self, filename: str | Path, **kwargs):
        """General class to read reactions/molecules data files.

        :param filename: The path and name of the file.
        :return: None.
        """
        super().__init__(filename, **kwargs)

    def __enter__(self):
        return self._file

    def __iter__(self):
        return iter(self._file)

    def __next__(self):
        return next(self._file)

    def __len__(self):
        return len(self._file)


def split_smiles_record(record: str) -> tuple[str, list[str]]:
    """Split a SMILES/SMIRKS text record into chemistry and source fields.

    SynPlanner writes SMI records as ``reaction_smiles<TAB>meta...``.  Mapping
    tools often append source/provenance columns in the same shape.  Parsing
    only the first token is correct chemically, but the trailing fields must be
    carried through curation so users can trace standardized outputs and errors
    back to the source dataset row.
    """
    raw = record.strip()
    if not raw:
        return "", []
    if "\t" in raw:
        smiles_part, *source_fields = raw.split("\t")
    else:
        parts = raw.split(maxsplit=1)
        smiles_part = parts[0]
        source_fields = parts[1:]
    return smiles_part.strip(), [field.strip() for field in source_fields]


def _store_source_fields(
    container: ReactionContainer | CGRContainer | MoleculeContainer,
    source_fields: list[str],
) -> None:
    """Store positional source fields under sortable metadata keys."""
    for index, field in enumerate(source_fields, start=1):
        if field:
            container.meta[f"source_{index:04d}"] = field


def format_source_fields(source_fields: Iterable[str]) -> str:
    """Format source/provenance fields for a single TSV cell."""
    return ";".join(
        str(field).replace("\t", " ").replace("\n", " ")
        for field in source_fields
        if str(field)
    )


def reaction_source_info(reaction: ReactionContainer) -> str:
    """Return source/provenance metadata as a compact TSV-safe string."""
    source_fields = [
        str(value)
        for key, value in sorted(reaction.meta.items(), key=lambda item: item[0])
        if key.startswith("source_")
    ]
    if source_fields:
        return format_source_fields(source_fields)

    meta_fields = [
        f"{key}={value}"
        for key, value in sorted(reaction.meta.items(), key=lambda item: item[0])
        if key != "init_smiles"
    ]
    return format_source_fields(meta_fields)


def tsv_safe(text: str) -> str:
    """Strip embedded tabs/newlines so TSV row alignment is preserved."""
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def extract_origin_fields(raw_item, fmt: str) -> tuple[str, str]:
    """Return ``(original_smiles, source_info)`` for an error entry.

    For SMILES input, splits the raw record into the SMILES part and the
    extra source columns (USPTO patent id, etc.); putting the latter into
    ``source_info`` keeps the TSV's first column clean.
    For RDF input, the raw record is a multi-line block; we collapse it to
    a single first-line identifier to preserve column alignment.
    """
    if not isinstance(raw_item, str):
        return tsv_safe(str(raw_item)), ""
    if fmt == "smi":
        smiles_part, source_fields = split_smiles_record(raw_item)
        return tsv_safe(smiles_part), format_source_fields(source_fields)
    # RDF (or unknown framing): take first non-empty line as the identifier.
    first_line = next(
        (ln.strip() for ln in raw_item.splitlines() if ln.strip()),
        raw_item.strip(),
    )
    return tsv_safe(first_line), ""


ERROR_TSV_HEADER = "# original_smiles\tsource_info\tstage\terror_type\terror_message\n"


def write_error_tsv_header(file: TextIO) -> None:
    """Write the canonical commented header row for an error TSV."""
    file.write(ERROR_TSV_HEADER)


def parse_error_message(
    msg: str, default_stage: str, default_error_type: str = "Error"
) -> tuple[str, str, str]:
    """Split a blended ``"stage: type: detail"`` exception string.

    Fall back to ``(default_stage, default_error_type, msg)`` when the
    string is not in the blended form.
    """
    if ":" not in msg:
        return default_stage, default_error_type, msg
    head, _, rest = msg.partition(":")
    stage = head.strip() or default_stage
    etype, _, detail = rest.partition(":")
    error_type = etype.strip() or default_error_type
    message = (detail.strip() or rest.strip()) or msg
    return stage, error_type, message


def write_error_row(
    file: TextIO,
    original: str,
    source_info: str,
    stage: str,
    error_type: str,
    message: str,
) -> None:
    """Write one row to an error TSV, escaping every field for column safety."""
    file.write(
        f"{tsv_safe(original)}\t{tsv_safe(source_info)}\t"
        f"{tsv_safe(stage)}\t{tsv_safe(error_type)}\t"
        f"{tsv_safe(message)}\n"
    )


class SMILESRead:
    def __init__(self, filename: str | Path, **kwargs):
        """Simplified class to read files containing a SMILES (Molecules or Reaction)
        string per line.

        :param filename: The path and name of the SMILES file to parse.
        :return: None.
        """
        filename = str(Path(filename).resolve(strict=True))
        self._file = open(filename, encoding="utf-8")
        self._data = self.__data()

    def __data(
        self,
    ) -> Iterable[ReactionContainer | CGRContainer | MoleculeContainer]:
        for line in iter(self._file.readline, ""):
            line = line.strip()
            smiles_part, source_fields = split_smiles_record(line)
            x = smiles(smiles_part)
            if isinstance(x, (ReactionContainer, CGRContainer, MoleculeContainer)):
                x.meta["init_smiles"] = smiles_part
                _store_source_fields(x, source_fields)
                yield x

    def __enter__(self):
        return self

    def read(self):
        """Parse the whole SMILES file.

        :return: List of parsed molecules or reactions.
        """
        return list(iter(self))

    def __iter__(self):
        return (x for x in self._data)

    def __next__(self):
        return next(iter(self))

    def close(self):
        self._file.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Writer(FileHandler):
    def __init__(self, filename: str | Path, mapping: bool = True, **kwargs):
        """General class to write chemical files.

        :param filename: The path and name of the file.
        :param mapping: Whenever to save mapping or not.
        :return: None.
        """
        super().__init__(filename, **kwargs)
        self._mapping = mapping

    def __enter__(self):
        return self


class _ORDReadAdapter:
    """Adapts iter_ord_reactions to the Reader protocol used by ReactionReader."""

    def __init__(self, filename: str | Path):
        self._path = Path(filename)

    def __iter__(self):
        from synplan.utils.ord.reader import iter_ord_reactions

        return iter_ord_reactions(self._path)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def close(self):
        pass


class ReactionReader(Reader):
    def __init__(self, filename: str | Path, **kwargs):
        """Class to read reaction files.

        :param filename: The path and name of the file.
        :return: None.
        """
        super().__init__(filename, **kwargs)
        if self._file_type == "SMI":
            self._file = SMILESRead(filename, **kwargs)
        elif self._file_type == "RDF":
            self._file = RDFRead(filename, indexable=True, **kwargs)
        elif self._file_type == "PB":
            self._file = _ORDReadAdapter(filename)
        else:
            raise ValueError("File type incompatible -", filename)


class ReactionWriter(Writer):
    def __init__(self, filename: str | Path, mapping: bool = True, **kwargs):
        """Class to write reaction files.

        :param filename: The path and name of the file.
        :param mapping: Whenever to save mapping or not.
        :return: None.
        """
        super().__init__(filename, mapping, **kwargs)
        self._rdf_header_written = False
        if self._file_type == "SMI":
            self._file = open(filename, "w", encoding="utf-8", **kwargs)
        elif self._file_type == "RDF":
            self._file = RDFWrite(filename, append=False, **kwargs)
        else:
            raise ValueError("File type incompatible -", filename)

    def write(self, reaction: ReactionContainer):
        """Function to write a specific reaction to the file.

        :param reaction: The path and name of the file.
        :return: None.
        """
        if self._file_type == "SMI":
            rea_str = to_reaction_smiles_record(reaction)
            self._file.write(rea_str + "\n")
        elif self._file_type == "RDF":
            self._file.write(reaction)

    def write_string(self, record: str):
        """Write a pre-serialized record directly, bypassing ReactionContainer.

        ``SMI`` records are written as SMILES lines. ``RDF`` records are written
        to the underlying text file, with the file-level header emitted on the
        first call.

        Use this when workers have already serialized reactions (via
        synplan.chem.data.pipeline.serialize_reaction) to avoid redundant work in
        the parent process.
        """
        s = record if record.endswith("\n") else record + "\n"
        if self._file_type == "SMI":
            self._file.write(s)
        elif self._file_type == "RDF":
            if not self._rdf_header_written:
                from time import strftime

                # chython exposes RDFWrite's wrapped text stream only via the
                # private ``_file._file`` path (verified with chython 2.x).
                # We write the file-level header here because ``write_string``
                # receives already serialized RDF records from worker processes,
                # so RDFWrite.write(rxn) is intentionally bypassed.
                self._file._file.write(strftime("$RDFILE 1\n$DATM    %m/%d/%y %H:%M\n"))
                # Remove the instance-level __write override so a subsequent
                # write(rxn) call does not re-emit the header.
                with contextlib.suppress(AttributeError):
                    del self._file.write
                self._rdf_header_written = True
            self._file._file.write(s)
        else:
            raise ValueError(f"write_string not supported for {self._file_type}")


class MoleculeReader(Reader):
    def __init__(self, filename: str | Path, **kwargs):
        """Class to read molecule files.

        :param filename: The path and name of the file.
        :return: None.
        """
        super().__init__(filename, **kwargs)
        if self._file_type == "SMI":
            self._file = SMILESRead(filename, ignore=True, **kwargs)
        elif self._file_type == "SDF":
            self._file = SDFRead(filename, indexable=True, **kwargs)
        else:
            raise ValueError("File type incompatible -", filename)


class MoleculeWriter(Writer):
    def __init__(self, filename: str | Path, mapping: bool = True, **kwargs):
        """Class to write molecule files.

        :param filename: The path and name of the file.
        :param mapping: Whenever to save mapping or not.
        :return: None.
        """
        super().__init__(filename, mapping, **kwargs)
        if self._file_type == "SMI":
            self._file = open(filename, "w", encoding="utf-8", **kwargs)
        elif self._file_type == "SDF":
            self._file = SDFWrite(filename, append=False, **kwargs)
        else:
            raise ValueError("File type incompatible -", filename)

    def write(self, molecule: MoleculeContainer):
        """Function to write a specific molecule to the file.

        :param molecule: The path and name of the file.
        :return: None.
        """
        if self._file_type == "SMI":
            mol_str = str(molecule)
            self._file.write(mol_str + "\n")
        elif self._file_type == "SDF":
            self._file.write(molecule)


def count_sdf_records(path: str | Path) -> int:
    """Count number of SDF records (by '$$$$' separators)."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip() == "$$$$")


def iter_sdf_text_blocks(path: str | Path, records_per_block: int) -> Iterator[str]:
    """Yield SDF text blocks containing up to `records_per_block` molecules.

    Records are delimited by lines equal to '$$$$'. The block is a concatenated
    string of lines that can be fed to StringIO and parsed by SDFRead.
    """
    p = Path(path)
    buf: list[str] = []
    count = 0
    step = max(1, records_per_block)
    with open(p, encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if line.strip() == "$$$$":
                count += 1
                if count % step == 0:
                    yield "".join(buf)
                    buf = []
    if buf:
        yield "".join(buf)


def open_text(path: str | Path) -> TextIO:
    """Open a text file that may be gzip-compressed.

    If the path ends with ".gz", the file is opened via gzip in text mode.
    """
    p = Path(path)
    if p.suffix.lower() == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", newline="")
    return open(p, encoding="utf-8", newline="")


def _resolve_csv_column(fieldnames: list[str] | None, column: str) -> str:
    """Resolve CSV column name with a case-insensitive fallback."""
    if not fieldnames:
        raise ValueError("CSV header is missing field names.")
    if column in fieldnames:
        return column

    matches = [name for name in fieldnames if name.lower() == column.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous CSV column name '{column}'. Case-insensitive matches: {matches}"
        )
    raise ValueError(f"Expected '{column}' column in CSV header, got {fieldnames}")


def count_smiles_records(path: str | Path) -> int:
    """Count number of non-empty SMILES records (lines)."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def iter_smiles(path: str | Path) -> Iterator[str]:
    """Yield first whitespace-delimited token (SMILES) per non-empty line."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            smiles_part, _source_fields = split_smiles_record(line)
            yield smiles_part


def iter_smiles_records(path: str | Path) -> Iterator[str]:
    """Yield complete non-empty SMI records, including metadata columns."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def iter_smiles_blocks(path: str | Path, records_per_block: int) -> Iterator[list[str]]:
    """Yield SMILES lists of up to `records_per_block` items from file."""
    step = max(1, records_per_block)
    block: list[str] = []
    for smi in iter_smiles(path):
        block.append(smi)
        if len(block) == step:
            yield block
            block = []
    if block:
        yield block


def iter_csv_smiles(
    path: str | Path,
    *,
    header: bool = True,
    delimiter: str = ",",
    smiles_column: str = "SMILES",
) -> Iterator[str]:
    """Yield SMILES strings from a CSV/CSV.GZ file.

    Parameters
    ----------
    path
        Path to the CSV file. If it ends with ".gz", it's treated as gzipped CSV.
    header
        If True, treat the first row as a header and read from `smiles_column`.
        If False, treat the first column as SMILES.
    delimiter
        CSV delimiter (default: ",").
    smiles_column
        Column name containing SMILES (used when `header=True`).
    """
    with open_text(path) as f:
        if header:
            reader = csv.DictReader(f, delimiter=delimiter)
            column = _resolve_csv_column(reader.fieldnames, smiles_column)
            for row in reader:
                smi = (row.get(column) or "").strip()
                if smi:
                    yield smi
        else:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if not row:
                    continue
                smi = (row[0] or "").strip()
                if smi:
                    yield smi


def iter_csv_smiles_blocks(
    path: str | Path,
    records_per_block: int,
    *,
    header: bool = True,
    delimiter: str = ",",
    smiles_column: str = "SMILES",
) -> Iterator[list[str]]:
    """Yield SMILES lists of up to `records_per_block` items from a CSV/CSV.GZ file."""
    step = max(1, records_per_block)
    block: list[str] = []
    for smi in iter_csv_smiles(
        path, header=header, delimiter=delimiter, smiles_column=smiles_column
    ):
        block.append(smi)
        if len(block) == step:
            yield block
            block = []
    if block:
        yield block


def count_rdf_records(path: str | Path) -> int:
    """Count number of RDF records.

    MDL permits two equivalent framings: ``$RFMT``/``$MFMT`` between records
    (each frames one ``$RXN``), or bare ``$RXN`` records under a single RDF
    header. We count whichever framing the file actually uses: if any
    ``$RFMT``/``$MFMT`` lines exist they delimit, otherwise ``$RXN`` lines do.
    """
    p = Path(path)
    rfmt_count = 0
    rxn_count = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.startswith(("$RFMT", "$MFMT")):
                rfmt_count += 1
            elif line.startswith("$RXN"):
                rxn_count += 1
    return rfmt_count if rfmt_count > 0 else rxn_count


def iter_rdf_text_blocks(path: str | Path, records_per_block: int) -> Iterator[str]:
    """Yield RDF text blocks of up to ``records_per_block`` records.

    Each block is a string containing one or more records, parseable via
    ``RDFRead(StringIO(block))``. Both legal MDL framings are supported:
    ``$RFMT``/``$MFMT``-framed records, and bare ``$RXN``-framed records
    (CHORISO-style). The delimiter is detected on the first record start.
    """
    p = Path(path)
    buf: list[str] = []
    count = 0
    step = max(1, records_per_block)
    in_header = True
    delimiters: tuple[str, ...] = ("$RFMT", "$MFMT")
    with open(p, encoding="utf-8") as f:
        for line in f:
            if in_header:
                if line.startswith(("$RFMT", "$MFMT", "$RXN")):
                    in_header = False
                    # If the first delimiter is a bare $RXN (no $RFMT/$MFMT
                    # wrapper anywhere ahead of it), records are split on
                    # $RXN. Otherwise $RFMT/$MFMT delimit.
                    if line.startswith("$RXN"):
                        delimiters = ("$RXN",)
                    buf.append(line)
                continue
            if line.startswith(delimiters):
                count += 1
                if count % step == 0:
                    yield "".join(buf)
                    buf = [line]
                    continue
            buf.append(line)
    if buf:
        yield "".join(buf)


def _check_mapping_status(status: str, mode: "AtomMappingCheck") -> None:
    if mode == "off":
        return
    if status == "unmapped":
        raise MappingError(
            "reaction has no shared atom numbers between reactants and products"
        )
    if status == "partially_mapped" and mode == "reject_partial":
        raise MappingError("reaction is only partially atom-mapped")


def parse_reaction(
    item: str | ReactionContainer,
    fmt: str = "smi",
    *,
    check_atom_mapping: "AtomMappingCheck" = "off",
) -> ReactionContainer:
    """Parse a raw string into a ReactionContainer.

    :param item: SMILES string, RDF text block, or ReactionContainer.
    :param fmt: ``"smi"`` or ``"rdf"``.
    :param check_atom_mapping: see :data:`synplan.chem.utils.AtomMappingCheck`.
        When enabled the parsed reaction is tagged via
        ``rxn.meta["mapping_status"]``.
    """
    if check_atom_mapping != "off":
        from synplan.chem.utils import (
            reaction_mapping_status,
            reaction_string_mapping_status,
        )

    if isinstance(item, ReactionContainer):
        rxn = item
        if check_atom_mapping != "off":
            status = reaction_mapping_status(rxn)
            _check_mapping_status(status, check_atom_mapping)
            rxn.meta["mapping_status"] = status
        return rxn
    if fmt == "smi":
        smiles_part, source_fields = split_smiles_record(item)
        # String-based check works for SMILES and rule SMARTS alike; the
        # SMARTS parser drops parsed_mapping so container check would lie.
        if check_atom_mapping != "off":
            status = reaction_string_mapping_status(smiles_part)
            _check_mapping_status(status, check_atom_mapping)
        rxn = smiles(smiles_part)
        rxn.meta["init_smiles"] = smiles_part
        _store_source_fields(rxn, source_fields)
        if check_atom_mapping != "off":
            rxn.meta["mapping_status"] = status
        return rxn
    else:  # rdf block
        with RDFRead(StringIO(item), ignore=True) as r:
            rxn = next(iter(r))
        if check_atom_mapping != "off":
            status = reaction_mapping_status(rxn)
            _check_mapping_status(status, check_atom_mapping)
            rxn.meta["mapping_status"] = status
        return rxn


# Process-pool worker globals (set via init_parse_worker, read in parse_one).
_parse_fmt = "smi"
_parse_check_atom_mapping: "AtomMappingCheck" = "off"


def init_parse_worker(fmt: str, check_atom_mapping: "AtomMappingCheck" = "off"):
    """Pool initializer: set format and mapping-check mode for ``parse_one``."""
    global _parse_fmt, _parse_check_atom_mapping
    _parse_fmt = fmt
    _parse_check_atom_mapping = check_atom_mapping


def parse_one(item: str):
    """Parse a single raw item → ``(ReactionContainer | None, error | None)``."""
    try:
        rxn = parse_reaction(
            item, fmt=_parse_fmt, check_atom_mapping=_parse_check_atom_mapping
        )
        if rxn is None:
            return None, "parse returned None"
        if not isinstance(rxn, ReactionContainer):
            return None, "not a reaction"
        return rxn, None
    except Exception as e:
        return None, f"parse: {e}"


class RawReactionReader:
    """Yields raw unparsed items: str lines for SMILES, str blocks for RDF."""

    def __init__(self, filename: str | Path, batch_size: int = 1):
        p = Path(filename)
        ext = p.suffix.lower()
        if ext in (".smi", ".smiles"):
            self.format = "smi"
        elif ext == ".rdf":
            self.format = "rdf"
        elif ext == ".pb":
            self.format = "pb"
        else:
            raise ValueError(f"Unsupported extension for raw reading: {ext}")
        self._path = p
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[str | ReactionContainer]:
        if self.format == "smi":
            yield from iter_smiles_records(self._path)
        elif self.format == "rdf":
            yield from iter_rdf_text_blocks(self._path, 1)
        elif self.format == "pb":
            from synplan.utils.ord.reader import iter_ord_reactions

            yield from iter_ord_reactions(self._path)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def count(self) -> int:
        """Count the number of records without parsing them."""
        if self.format == "smi":
            return count_smiles_records(self._path)
        elif self.format == "rdf":
            return count_rdf_records(self._path)
        elif self.format == "pb":
            from synplan.utils.ord.reader import iter_ord_reactions

            return sum(1 for _ in iter_ord_reactions(self._path))
        raise ValueError(f"Unsupported format: {self.format}")

    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[int, list]]:
        """Yield ``(offset, items)`` chunks of up to *chunk_size* records."""
        chunk: list = []
        offset = 0
        for item in self:
            chunk.append(item)
            if len(chunk) == chunk_size:
                yield offset, chunk
                offset += len(chunk)
                chunk = []
        if chunk:
            yield offset, chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def load_rule_index_mapping_tsv(tsv_path: str | Path) -> dict:
    """Load reaction-to-rule index mapping from a rules TSV file.

    The TSV is already sorted by descending popularity (same order as the
    pickle), so the rule index is simply the row number (0-based).

    :param tsv_path: Path to the TSV file with columns
        ``rule_smarts``, ``popularity``, ``reaction_indices``.
    :return: Dict mapping ``reaction_index → rule_index``.
    """
    reaction_rule_pairs = {}
    with open(tsv_path, encoding="utf-8") as f:
        f.readline()  # skip header
        for rule_i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            indices_str = parts[2]
            for reaction_id_str in indices_str.split(","):
                reaction_rule_pairs[int(reaction_id_str)] = rule_i
    return dict(sorted(reaction_rule_pairs.items()))


def to_reaction_smiles_record(reaction: ReactionContainer) -> str:
    """Converts the reaction to the SMILES record. Needed for reaction/molecule writers.

    :param reaction: The reaction to be written.
    :return: The SMILES record to be written.
    """

    if isinstance(reaction, str):
        return reaction

    reaction_record = [format(reaction, "m")]
    has_source_fields = any(key.startswith("source_") for key in reaction.meta)
    source_meta = [
        (key, value)
        for key, value in sorted(reaction.meta.items(), key=lambda item: item[0])
        if key.startswith("source_")
    ]
    other_meta = [
        (key, value)
        for key, value in sorted(reaction.meta.items(), key=lambda item: item[0])
        if not key.startswith("source_")
    ]
    ordered_meta = source_meta + other_meta if has_source_fields else other_meta
    for key, meta_info in ordered_meta:
        if has_source_fields and key == "init_smiles":
            continue
        meta_info = ";".join(str(meta_info).split("\n"))
        reaction_record.append(str(meta_info))
    return "\t".join(reaction_record)
