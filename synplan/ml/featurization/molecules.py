"""Torch tensorization of chython molecules into PyG graphs."""

from __future__ import annotations

from typing import Any

import torch
from chython.containers import MoleculeContainer
from chython.exceptions import InvalidAromaticRing
from torch import Tensor
from torch_geometric.data.data import Data
from torch_geometric.transforms import ToUndirected


def atom_to_vector(atom: Any) -> Tensor:
    """Given an atom, return a vector of length 8 with the following
    information:

    1. Atomic number
    2. Period
    3. Group
    4. Number of electrons + atom's charge
    5. Shell
    6. Total number of hydrogens
    7. Whether the atom is in a ring
    8. Number of neighbors

    :param atom: The atom object.

    :return: The vector of the atom.
    """
    vector = torch.zeros(8, dtype=torch.uint8)
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


def bonds_to_vector(molecule: MoleculeContainer, atom_ind: int) -> Tensor:
    """Takes a molecule and an atom index as input, and returns a vector representing
    the bond orders of the atom's bonds.

    :param molecule: The given molecule.
    :param atom_ind: The index of the atom in the molecule to be converted to the bond
        vector.
    :return: The torch tensor of size 3, with each element representing the order of
        bonds connected to the atom with the given index in the molecule.
    """

    vector = torch.zeros(3, dtype=torch.uint8)
    for b_order in molecule._bonds[atom_ind].values():
        vector[int(b_order) - 1] += 1
    return vector


def mol_to_matrix(molecule: MoleculeContainer) -> Tensor:
    """Given a molecule, it returns a vector of shape (max_atoms, 12) where each row is
    an atom and each column is a feature.

    :param molecule: The molecule to be converted to a vector
    :return: The atoms vectors array.
    """

    atoms_vectors = torch.zeros((len(molecule), 11), dtype=torch.uint8)
    for n, atom in molecule.atoms():
        atoms_vectors[n - 1][:8] = atom_to_vector(atom)
    for n, _ in molecule.atoms():
        atoms_vectors[n - 1][8:] = bonds_to_vector(molecule, n)

    return atoms_vectors


def mol_to_pyg(molecule: MoleculeContainer, canonicalize: bool = True) -> Data | None:
    """Takes a list of molecules and returns a list of PyTorch Geometric graphs, a one-
    hot encoded vectors of the atoms, and a matrices of the bonds.

    :param molecule: The molecule to be converted to PyTorch Geometric graph.
    :param canonicalize: If True, the input molecule is canonicalized.
    :return: The list of PyGraph objects.
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
        edge_index.append([atom - 1, neighbour - 1])
        edge_attr.append(
            [
                float(bond.order == 1),
                float(bond.order == 2),
                float(bond.order == 3),
                float(bond.in_ring),
            ]
        )
    # Edgeless precursors (e.g. [NH4+].[OH-]) have no bonded fragment to expand;
    # disconnected salts still pass as long as one component has bonds.
    if not edge_index:
        return None
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = mol_to_matrix(tmp_molecule)

    mol_pyg_graph = Data(
        x=x,
        edge_index=edge_index.t().contiguous(),
        edge_attr=edge_attr,
    )
    mol_pyg_graph = ToUndirected()(mol_pyg_graph)

    assert mol_pyg_graph.is_undirected()

    return mol_pyg_graph


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
    "atom_to_vector",
    "bonds_to_vector",
    "mol_to_matrix",
    "mol_to_pyg",
]
