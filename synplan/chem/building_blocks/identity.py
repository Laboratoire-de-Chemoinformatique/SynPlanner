"""Standard InChI identity for ordinary building blocks.

The planner uses direct, full Standard InChIKeys for catalogue membership.
Standard InChI is generated separately when auditable reference data is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chython.containers import MoleculeContainer
from rdkit.Chem import rdinchi
from rdkit.Chem.rdchem import Mol

from synplan.chem.utils import safe_canonicalization

_STANDARD_INCHI_PREFIX = "InChI=1S/"
_STANDARD_INCHI_KEY = re.compile(r"^[A-Z]{14}-[A-Z]{8}SA-[A-Z]$")


class MoleculeIdentityError(ValueError):
    """Raised when a molecule cannot be represented by Standard InChI."""


@dataclass(frozen=True, slots=True)
class MoleculeIdentity:
    """Auditable representations produced for one molecule."""

    canonical_smiles: str
    standard_inchi: str
    inchi_key: str
    return_code: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _InchiResult:
    canonical_smiles: str
    standard_inchi: str
    return_code: int
    warnings: tuple[str, ...]


def _warning_messages(message: str) -> tuple[str, ...]:
    """Split RDKit's semicolon/newline warning field into stable messages."""
    return tuple(
        item.strip()
        for line in message.splitlines()
        for item in line.split(";")
        if item.strip()
    )


def _canonical_rdkit_molecule(molecule: MoleculeContainer) -> tuple[str, Mol]:
    if not isinstance(molecule, MoleculeContainer):
        raise TypeError("molecule must be a Chython MoleculeContainer")
    try:
        copied = safe_canonicalization(molecule, clean_stereo=False)
    except Exception as error:
        raise MoleculeIdentityError(
            f"cannot canonicalize molecule for identity conversion: {error}"
        ) from error
    canonical_smiles = str(copied)
    try:
        rdkit_molecule = copied.to_rdkit(keep_mapping=False)
    except Exception as error:
        raise MoleculeIdentityError(
            "cannot convert molecule to RDKit for identity conversion: "
            f"{canonical_smiles!r}: {error}"
        ) from error
    if rdkit_molecule is None:
        raise MoleculeIdentityError(
            "cannot convert molecule to RDKit for identity conversion: "
            f"{canonical_smiles!r}"
        )
    return canonical_smiles, rdkit_molecule


def _rdkit_molecule_to_inchi_result(
    canonical_smiles: str, rdkit_molecule: Mol
) -> _InchiResult:
    try:
        inchi, return_code, message, _log, _aux_info = rdinchi.MolToInchi(
            rdkit_molecule
        )
    except Exception as error:
        raise MoleculeIdentityError(
            f"RDKit failed to generate InChI for {canonical_smiles!r}: {error}"
        ) from error

    return_code = int(return_code)
    if return_code not in {0, 1}:
        detail = message.strip() or "no diagnostic message"
        raise MoleculeIdentityError(
            "RDKit InChI conversion failed for "
            f"{canonical_smiles!r} with return code {return_code}: {detail}"
        )
    if not inchi.startswith(_STANDARD_INCHI_PREFIX):
        raise MoleculeIdentityError(
            f"RDKit returned a non-Standard InChI for {canonical_smiles!r}: {inchi!r}"
        )
    return _InchiResult(
        canonical_smiles=canonical_smiles,
        standard_inchi=inchi,
        return_code=return_code,
        warnings=_warning_messages(message),
    )


def _molecule_to_inchi_result(molecule: MoleculeContainer) -> _InchiResult:
    canonical_smiles, rdkit_molecule = _canonical_rdkit_molecule(molecule)
    return _rdkit_molecule_to_inchi_result(canonical_smiles, rdkit_molecule)


def validate_standard_inchi_key(inchi_key: str) -> str:
    """Validate and return one complete, version-1 Standard InChIKey."""
    if not isinstance(inchi_key, str) or not _STANDARD_INCHI_KEY.fullmatch(inchi_key):
        raise MoleculeIdentityError(
            f"invalid Standard InChIKey (expected 27 characters): {inchi_key!r}"
        )
    return inchi_key


def _rdkit_molecule_to_inchi_key(canonical_smiles: str, rdkit_molecule: Mol) -> str:
    try:
        inchi_key = rdinchi.MolToInchiKey(rdkit_molecule)
    except Exception as error:
        raise MoleculeIdentityError(
            "RDKit failed to generate an InChIKey directly for "
            f"{canonical_smiles!r}: {error}"
        ) from error
    return validate_standard_inchi_key(inchi_key)


def molecule_to_inchi(molecule: MoleculeContainer) -> str:
    """Return the Standard InChI of a Chython molecule."""
    return _molecule_to_inchi_result(molecule).standard_inchi


def inchi_to_inchi_key(inchi: str) -> str:
    """Derive a full Standard InChIKey from a Standard InChI string."""
    if not isinstance(inchi, str) or not inchi.startswith(_STANDARD_INCHI_PREFIX):
        raise MoleculeIdentityError(f"expected a Standard InChI, got {inchi!r}")
    if inchi != inchi.strip() or "\n" in inchi or "\r" in inchi or "\t" in inchi:
        raise MoleculeIdentityError(
            "Standard InChI must not contain whitespace padding"
        )
    try:
        inchi_key = rdinchi.InchiToInchiKey(inchi)
    except Exception as error:
        raise MoleculeIdentityError(
            f"RDKit failed to generate an InChIKey from {inchi!r}: {error}"
        ) from error
    return validate_standard_inchi_key(inchi_key)


def molecule_to_inchi_key(molecule: MoleculeContainer) -> str:
    """Return the direct full Standard InChIKey of a Chython molecule."""
    canonical_smiles, rdkit_molecule = _canonical_rdkit_molecule(molecule)
    # Match AiZynthFinder stock lookup: do not materialize an intermediate InChI.
    return _rdkit_molecule_to_inchi_key(canonical_smiles, rdkit_molecule)


def molecule_identity(molecule: MoleculeContainer) -> MoleculeIdentity:
    """Return canonical SMILES, Standard InChI, and its full InChIKey."""
    canonical_smiles, rdkit_molecule = _canonical_rdkit_molecule(molecule)
    result = _rdkit_molecule_to_inchi_result(canonical_smiles, rdkit_molecule)
    return MoleculeIdentity(
        canonical_smiles=result.canonical_smiles,
        standard_inchi=result.standard_inchi,
        inchi_key=_rdkit_molecule_to_inchi_key(result.canonical_smiles, rdkit_molecule),
        return_code=result.return_code,
        warnings=result.warnings,
    )


__all__ = [
    "MoleculeIdentity",
    "MoleculeIdentityError",
    "inchi_to_inchi_key",
    "molecule_identity",
    "molecule_to_inchi",
    "molecule_to_inchi_key",
    "validate_standard_inchi_key",
]
