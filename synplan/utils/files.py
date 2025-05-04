"""Module containing classes and functions needed for reactions/molecules data
reading/writing."""

from os.path import splitext
from pathlib import Path
from typing import Iterable, Union

from CGRtools import smiles
from CGRtools.containers import CGRContainer, MoleculeContainer, ReactionContainer
from CGRtools.files.RDFrw import RDFRead, RDFWrite
from CGRtools.files.SDFrw import SDFRead, SDFWrite


class FileHandler:
    """General class to handle chemical files."""

    def __init__(self, filename: Union[str, Path], **kwargs):
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
    def __init__(self, filename: Union[str, Path], **kwargs):
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
    def __init__(self, filename: Union[str, Path], **kwargs):
        """Simplified class to read files containing a SMILES (Molecules or Reaction)
        string per line.

        :param filename: The path and name of the SMILES file to parse.
        :return: None.
        """
        filename = str(Path(filename).resolve(strict=True))
        self._file = open(filename, "r", encoding="utf-8")
        self._data = self.__data()

    def __data(
        self,
    ) -> Iterable[Union[ReactionContainer, CGRContainer, MoleculeContainer]]:
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
    def __init__(self, filename: Union[str, Path], mapping: bool = True, **kwargs):
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
    def __init__(self, filename: Union[str, Path], **kwargs):
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
    def __init__(self, filename: Union[str, Path], mapping: bool = True, **kwargs):
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
    def __init__(self, filename: Union[str, Path], **kwargs):
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
    def __init__(self, filename: Union[str, Path], mapping: bool = True, **kwargs):
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
