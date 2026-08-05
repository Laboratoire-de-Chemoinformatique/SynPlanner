"""Molecule file input/output helpers."""

import logging
from io import StringIO

from chython import smiles as smiles_parser
from chython.files.SDFrw import SDFRead
from tqdm.auto import tqdm

from synplan.chem.molecule.standardization import safe_canonicalization
from synplan.utils.files import MoleculeReader, MoleculeWriter


def standardize_building_blocks(input_file: str, output_file: str) -> str:
    """Standardize a building-block file and return the output path."""
    if input_file == output_file:
        raise ValueError("input_file name and output_file name cannot be the same.")

    with (
        MoleculeReader(input_file) as inp_file,
        MoleculeWriter(output_file) as out_file,
    ):
        for molecule in tqdm(
            inp_file,
            desc="Number of building blocks processed: ",
            bar_format="{desc}{n} [{elapsed}]",
        ):
            try:
                molecule = safe_canonicalization(molecule)
            except Exception as error:
                logging.debug(error)
                continue
            out_file.write(molecule)
    return output_file


def _standardize_one_smiles(smiles_str: str) -> str | None:
    try:
        molecule = smiles_parser(smiles_str, ignore=True)
        molecule = safe_canonicalization(molecule)
        return str(molecule)
    except Exception:
        return None


def _standardize_sdf_range(filename: str, start: int, end: int) -> list[str]:
    output: list[str] = []
    sdf = SDFRead(filename, indexable=True)
    try:
        for index in range(start, end):
            try:
                molecule = safe_canonicalization(sdf[index])
                output.append(str(molecule))
            except Exception:
                pass
    finally:
        sdf.close()
    return output


def standardize_sdf_text(block: str) -> list[str]:
    """Standardize molecules from one or more SDF records in text."""
    output: list[str] = []
    with StringIO(block) as stream, SDFRead(stream) as sdf:
        for molecule in sdf:
            try:
                molecule = safe_canonicalization(molecule)
                output.append(str(molecule))
            except Exception:
                pass
    return output


def standardize_smiles_batch(batch: list[str]) -> list[str]:
    """Standardize a batch of SMILES strings and return valid results."""
    output: list[str] = []
    for smiles_str in batch:
        result = _standardize_one_smiles(smiles_str)
        if result:
            output.append(result)
    return output


__all__ = [
    "standardize_building_blocks",
    "standardize_sdf_text",
    "standardize_smiles_batch",
]
