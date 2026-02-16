"""Module containing classes and functions needed for reactions/molecules data
reading/writing."""

import csv
import gzip
from io import StringIO
from os.path import splitext
from pathlib import Path
from typing import TextIO
from collections.abc import Iterable, Iterator

from chython import smiles
from chython.containers import CGRContainer, MoleculeContainer, ReactionContainer
from chython.files.RDFrw import RDFRead, RDFWrite
from chython.files.SDFrw import SDFRead, SDFWrite


class FileHandler:
    """General class to handle chemical files."""

    def __init__(self, filename: str | Path, **kwargs):
        """General class to handle chemical files.

        :param filename: The path and name of the file.
        :return: None.
        """
        self._file = None
        _, ext = splitext(filename)
        file_types = {".smi": "SMI", ".smiles": "SMI", ".rdf": "RDF", ".sdf": "SDF"}
        try:
            self._file_type = file_types[ext]
        except KeyError:
            raise ValueError("I don't know the file extension,", ext)

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
            x = smiles(line)
            if isinstance(x, (ReactionContainer, CGRContainer, MoleculeContainer)):
                x.meta["init_smiles"] = line
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


def iter_sdf_text_blocks(
    path: str | Path, records_per_block: int
) -> Iterator[str]:
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
            yield line.split()[0]


def iter_smiles_blocks(
    path: str | Path, records_per_block: int
) -> Iterator[list[str]]:
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
    """Count number of RDF records (by $RFMT/$MFMT markers)."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.startswith(("$RFMT", "$MFMT")))


def iter_rdf_text_blocks(
    path: str | Path, records_per_block: int
) -> Iterator[str]:
    """Yield RDF text blocks of up to `records_per_block` records.

    Each block is a string containing one or more $RFMT/$MFMT records,
    parseable via ``RDFRead(StringIO(block))``.
    """
    p = Path(path)
    buf: list[str] = []
    count = 0
    step = max(1, records_per_block)
    in_header = True
    with open(p, encoding="utf-8") as f:
        for line in f:
            if in_header:
                if line.startswith(("$RFMT", "$MFMT", "$RXN")):
                    in_header = False
                    buf.append(line)
                continue
            if line.startswith(("$RFMT", "$MFMT")):
                count += 1
                if count % step == 0:
                    yield "".join(buf)
                    buf = [line]
                    continue
            buf.append(line)
    if buf:
        yield "".join(buf)


def parse_reaction(
    item: str | ReactionContainer, fmt: str = "smi"
) -> ReactionContainer:
    """Parse a raw string into a ReactionContainer.

    If *item* is already a ReactionContainer it is returned as-is.

    :param item: A SMILES string, an RDF text block, or a ReactionContainer.
    :param fmt: ``"smi"`` for SMILES strings, ``"rdf"`` for RDF text blocks.
    :return: The parsed ReactionContainer.
    """
    if isinstance(item, ReactionContainer):
        return item
    if fmt == "smi":
        rxn = smiles(item)
        rxn.meta["init_smiles"] = item
        return rxn
    else:  # rdf block
        with RDFRead(StringIO(item), ignore=True) as r:
            rxn = next(iter(r))
        return rxn


class RawReactionReader:
    """Yields raw unparsed items: str lines for SMILES, str blocks for RDF."""

    def __init__(self, filename: str | Path, batch_size: int = 1):
        p = Path(filename)
        ext = p.suffix.lower()
        if ext in (".smi", ".smiles"):
            self.format = "smi"
        elif ext == ".rdf":
            self.format = "rdf"
        else:
            raise ValueError(f"Unsupported extension for raw reading: {ext}")
        self._path = p
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[str]:
        if self.format == "smi":
            yield from iter_smiles(self._path)
        else:
            yield from iter_rdf_text_blocks(self._path, 1)

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
    sorted_meta = sorted(reaction.meta.items(), key=lambda x: x[0])
    for _, meta_info in sorted_meta:
        meta_info = ""
        meta_info = ";".join(meta_info.split("\n"))
        reaction_record.append(str(meta_info))
    return "\t".join(reaction_record)
