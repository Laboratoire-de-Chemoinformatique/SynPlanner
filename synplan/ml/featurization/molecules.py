"""Shared molecular features for NumPy inference and Torch training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from chython.containers import MoleculeContainer
from chython.exceptions import InvalidAromaticRing

if TYPE_CHECKING:
    from torch import Tensor
    from torch_geometric.data import Data


def atom_features(atom) -> np.ndarray:
    """Return the eight atomic descriptors shared by inference and training."""
    vector = np.zeros(8, dtype=np.uint8)
    period, group, shell, electrons = MENDEL_INFO[atom.atomic_symbol]
    vector[0] = atom.atomic_number
    vector[1] = period
    vector[2] = group
    vector[3] = electrons + atom.charge
    vector[4] = shell
    vector[5] = atom.total_hydrogens
    vector[6] = int(atom.in_ring)
    vector[7] = atom.neighbors
    return vector


def bond_features(molecule: MoleculeContainer, atom_ind: int) -> np.ndarray:
    """Count single, double and triple bonds adjacent to an atom."""
    vector = np.zeros(3, dtype=np.uint8)
    for _, b_order in molecule.bond_items(atom_ind):
        vector[int(b_order) - 1] += 1
    return vector


def molecule_features(molecule: MoleculeContainer) -> np.ndarray:
    """Return the 11 features per atom for a molecule numbered from one."""
    atoms_vectors = np.zeros((len(molecule), 11), dtype=np.uint8)
    for n, atom in molecule.atoms():
        atoms_vectors[n - 1][:8] = atom_features(atom)
        atoms_vectors[n - 1][8:] = bond_features(molecule, n)

    return atoms_vectors


def atom_to_vector(atom) -> Tensor:
    """Return atomic descriptors as a Torch tensor (legacy public API)."""
    import torch

    return torch.from_numpy(atom_features(atom))


def bonds_to_vector(molecule: MoleculeContainer, atom_ind: int) -> Tensor:
    """Return bond counts as a Torch tensor (legacy public API)."""
    import torch

    return torch.from_numpy(bond_features(molecule, atom_ind))


def mol_to_matrix(molecule: MoleculeContainer) -> Tensor:
    """Given a molecule, it returns a vector of shape (max_atoms, 11) where each row is
    an atom and each column is a feature.

    :param molecule: The molecule to be converted to a vector
    :return: The atoms vectors array.
    """

    import torch

    return torch.from_numpy(molecule_features(molecule))


def mol_to_pyg(molecule: MoleculeContainer, canonicalize: bool = True) -> Data | None:
    """Wrap the shared molecular features as a PyG graph for Torch models."""
    import torch
    from torch_geometric.data import Data

    arrays = mol_to_numpy(molecule, canonicalize=canonicalize)
    if arrays is None:
        return None
    return Data(**{key: torch.from_numpy(value) for key, value in arrays.items()})


def mol_to_numpy(molecule: MoleculeContainer, canonicalize: bool = True) -> dict | None:
    """Return atom features, directed edges and bond features as NumPy arrays.

    :param molecule: The molecule to featurize.
    :param canonicalize: If True, the input molecule is canonicalized.
    :return: Input arrays, or None for an unsupported molecular graph.
    """

    if len(molecule) == 1:  # to avoid a precursor to be a single atom
        return None

    tmp_molecule = molecule.copy()

    try:
        if canonicalize:
            tmp_molecule.canonicalize()
        tmp_molecule.remove_coordinate_bonds(keep_to_terminal=False)
        tmp_molecule.kekule()
        if tmp_molecule.check_valence():
            return None
    except InvalidAromaticRing:
        return None

    # remapping target for torch_geometric because
    # it is necessary that the elements in edge_index only hold nodes_idx in the range { 0, ..., num_nodes - 1}
    new_mappings = {n: i for i, (n, _) in enumerate(tmp_molecule.atoms(), 1)}
    tmp_molecule.remap(new_mappings)

    # get edge indexes and edge features from target mapping
    edge_index = []
    edge_attr = []
    for atom, neighbour, bond in tmp_molecule.bonds():
        edge_index.extend([[atom - 1, neighbour - 1], [neighbour - 1, atom - 1]])
        features = [
            float(bond.order == 1),
            float(bond.order == 2),
            float(bond.order == 3),
            float(bond.in_ring),
        ]
        edge_attr.extend([features, features])
    # Edgeless precursors (e.g. [NH4+].[OH-]) have no bonded fragment to expand;
    # disconnected salts still pass as long as one component has bonds.
    if not edge_index:
        return None
    edge_index = np.asarray(edge_index, dtype=np.int64)
    order = np.lexsort((edge_index[:, 1], edge_index[:, 0]))
    return {
        "x": molecule_features(tmp_molecule),
        "edge_index": np.ascontiguousarray(edge_index[order].T),
        "edge_attr": np.asarray(edge_attr, dtype=np.float32)[order],
    }


MENDEL_INFO = {
    "Ag": (5, 11, 1, 1),
    "Al": (3, 13, 2, 1),
    "Ar": (3, 18, 2, 6),
    "As": (4, 15, 2, 3),
    "B": (2, 13, 2, 1),
    "Ba": (6, 2, 1, 2),
    "Bi": (6, 15, 2, 3),
    "Br": (4, 17, 2, 5),
    "C": (2, 14, 2, 2),
    "Ca": (4, 2, 1, 2),
    "Ce": (6, None, 1, 2),
    "Cl": (3, 17, 2, 5),
    "Cr": (4, 6, 1, 1),
    "Cs": (6, 1, 1, 1),
    "Cu": (4, 11, 1, 1),
    "Dy": (6, None, 1, 2),
    "Er": (6, None, 1, 2),
    "F": (2, 17, 2, 5),
    "Fe": (4, 8, 1, 2),
    "Ga": (4, 13, 2, 1),
    "Gd": (6, None, 1, 2),
    "Ge": (4, 14, 2, 2),
    "Hg": (6, 12, 1, 2),
    "I": (5, 17, 2, 5),
    "In": (5, 13, 2, 1),
    "K": (4, 1, 1, 1),
    "La": (6, 3, 1, 2),
    "Li": (2, 1, 1, 1),
    "Mg": (3, 2, 1, 2),
    "Mn": (4, 7, 1, 2),
    "N": (2, 15, 2, 3),
    "Na": (3, 1, 1, 1),
    "Nd": (6, None, 1, 2),
    "O": (2, 16, 2, 4),
    "P": (3, 15, 2, 3),
    "Pb": (6, 14, 2, 2),
    "Pd": (5, 10, 3, 10),
    "Pr": (6, None, 1, 2),
    "Rb": (5, 1, 1, 1),
    "S": (3, 16, 2, 4),
    "Sb": (5, 15, 2, 3),
    "Se": (4, 16, 2, 4),
    "Si": (3, 14, 2, 2),
    "Sm": (6, None, 1, 2),
    "Sn": (5, 14, 2, 2),
    "Sr": (5, 2, 1, 2),
    "Te": (5, 16, 2, 4),
    "Ti": (4, 4, 1, 2),
    "Tl": (6, 13, 2, 1),
    "Yb": (6, None, 1, 2),
    "Zn": (4, 12, 1, 2),
}


__all__ = [
    "MENDEL_INFO",
    "atom_features",
    "atom_to_vector",
    "bond_features",
    "bonds_to_vector",
    "mol_to_matrix",
    "mol_to_numpy",
    "mol_to_pyg",
    "molecule_features",
]
