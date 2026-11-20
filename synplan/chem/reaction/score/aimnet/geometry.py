"""Validate sampled coordinates against the requested molecular structure."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from .config import MolecularCalculationError
from .species import canonical_species


def _with_coordinates(mol, positions):
    result = Chem.Mol(mol)
    result.RemoveAllConformers()
    conformer = Chem.Conformer(result.GetNumAtoms())
    conformer.Set3D(True)
    for index, xyz in enumerate(positions):
        conformer.SetAtomPosition(index, tuple(map(float, xyz)))
    result.AddConformer(conformer, assignId=True)
    return result


def validate_geometry(mol, positions, *, optimized: bool) -> None:
    """Check coordinates independently of the labels carried by the input graph."""
    positions = np.asarray(positions, dtype=float)
    stage = "optimization" if optimized else "generation"
    if positions.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(positions).all():
        raise MolecularCalculationError(
            stage, "nonfinite_geometry", "Coordinates must be finite with shape (N, 3)."
        )
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    if np.any(distances < 0.4):
        raise MolecularCalculationError(
            stage, "collapsed_geometry", "An interatomic distance is below 0.4 Å."
        )
    # The AIMNet2 compressed-H defect concerns geminal H atoms. A real H-H
    # bond in H2 (~0.74 Å) must pass the pre-optimization 0.8 Å check.
    for atom in mol.GetAtoms():
        hydrogens = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        for i, left in enumerate(hydrogens):
            for right in hydrogens[i + 1 :]:
                if distances[left, right] < (0.6 if optimized else 0.8):
                    raise MolecularCalculationError(
                        stage,
                        "collapsed_hydrogens",
                        "Geminal hydrogens are too close for AIMNet2.",
                    )
    geometry = _with_coordinates(mol, positions)
    inferred = Chem.Mol(geometry)
    rdDetermineBonds.DetermineConnectivity(inferred, useVdw=True, covFactor=1.3)

    def edges(m):
        return {
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in m.GetBonds()
        }

    if edges(inferred) != edges(mol):
        raise MolecularCalculationError(
            stage,
            "connectivity_changed",
            "Coordinates do not match the requested molecular connectivity.",
        )
    Chem.RemoveStereochemistry(geometry)
    Chem.AssignStereochemistryFrom3D(geometry, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(geometry, cleanIt=True, force=True)
    if canonical_species(geometry) != canonical_species(mol):
        raise MolecularCalculationError(
            stage,
            "stereo_mismatch",
            "3D stereochemistry does not match the requested stereoisomer.",
        )
