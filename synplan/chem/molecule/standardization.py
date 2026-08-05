"""Parsing and standardization helpers for Chython molecules."""

import logging
from collections.abc import Iterable

from chython import smiles as smiles_parser
from chython.containers import MoleculeContainer
from chython.exceptions import InvalidAromaticRing


def _clean_molecule(
    molecule: MoleculeContainer,
    *,
    standardize: bool = True,
    clean_stereo: bool = True,
    clean2d: bool = True,
) -> MoleculeContainer:
    """Clean a Chython molecule on a copy while preserving failure semantics."""
    tmp = molecule.copy()
    try:
        tmp.remove_coordinate_bonds(keep_to_terminal=False)
        if standardize:
            tmp.canonicalize()
        if clean_stereo:
            tmp.clean_stereo()
        if clean2d:
            tmp.clean2d()
        return tmp
    except InvalidAromaticRing:
        logging.warning(
            "chython was not able to standardize molecule due to invalid aromatic ring"
        )
        return molecule


def mol_from_smiles(
    smiles: str,
    standardize: bool = True,
    clean_stereo: bool = True,
    clean2d: bool = True,
) -> MoleculeContainer:
    """Convert a SMILES string to a cleaned molecule container."""
    molecule = smiles_parser(smiles, ignore=True)
    if not isinstance(molecule, MoleculeContainer):
        raise ValueError("SMILES string was not processed by chython")
    return _clean_molecule(
        molecule,
        standardize=standardize,
        clean_stereo=clean_stereo,
        clean2d=clean2d,
    )


def unite_molecules(molecules: Iterable[MoleculeContainer]) -> MoleculeContainer:
    """Combine an iterable of molecules into one molecule."""
    new_mol = MoleculeContainer()
    for molecule in molecules:
        new_mol = new_mol.union(molecule)
    return new_mol


def safe_canonicalization(molecule: MoleculeContainer) -> MoleculeContainer:
    """Canonicalize a molecule, returning the input on invalid aromatic rings."""
    molecule._atoms = dict(sorted(molecule._atoms.items()))
    molecule_copy = molecule.copy()
    try:
        molecule_copy.remove_coordinate_bonds(keep_to_terminal=False)
        molecule_copy.canonicalize()
        molecule_copy.clean_stereo()
        return molecule_copy
    except InvalidAromaticRing:
        return molecule


__all__ = ["mol_from_smiles", "safe_canonicalization", "unite_molecules"]
