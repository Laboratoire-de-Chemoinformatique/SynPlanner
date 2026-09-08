"""Chython-only Standard InChIKey identity helpers."""

from __future__ import annotations

import re

from chython import inchi_key
from chython.containers import MoleculeContainer

from synplan.chem.stereo import has_stereo as molecule_has_stereo

_STANDARD_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{8}SA-[A-Z]$")


def validate_standard_inchikey(value: str, *, context: str = "value") -> str:
    """Validate and return one complete version-1 Standard InChIKey."""

    if not isinstance(value, str) or not _STANDARD_INCHIKEY.fullmatch(value):
        raise ValueError(f"{context}: invalid Standard InChIKey {value!r}")
    return value


def molecule_to_inchikey(molecule: MoleculeContainer) -> str:
    """Generate a full stereo- and isotope-sensitive Standard InChIKey.

    Chython may populate lazy caches while writing InChI. Work on a copy so an
    identity query never changes the caller's container or its pickle payload.
    """

    if not isinstance(molecule, MoleculeContainer):
        raise TypeError("molecule must be a Chython MoleculeContainer")
    return validate_standard_inchikey(
        inchi_key(molecule.copy(), ignore_stereo=False), context=str(molecule)
    )


__all__ = [
    "molecule_has_stereo",
    "molecule_to_inchikey",
    "validate_standard_inchikey",
]
