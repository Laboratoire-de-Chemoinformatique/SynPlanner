"""Module containing functions for preparation of the training sets for policy and value
network."""

import logging
import os
from abc import ABC
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import ray
import torch
from chython import smiles
from chython.containers import MoleculeContainer
from chython.exceptions import InvalidAromaticRing
from chython.reactor import Reactor
from ray.util.queue import Empty, Queue
from torch import Tensor
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.data.makedirs import makedirs
from torch_geometric.transforms import ToUndirected
from tqdm.auto import tqdm

from synplan.utils.cache import load_pyg_dataset, save_pyg_dataset
from synplan.utils.loading import load_reaction_rules
from synplan.utils.parallel import default_num_workers, process_pool_map_stream


class ValueNetworkDataset(InMemoryDataset, ABC):
    """Value network dataset."""

    def __init__(self, extracted_precursor: dict[str, float]) -> None:
        """Initializes a value network dataset object.

        :param extracted_precursor: The dictionary with the extracted from the built
            search trees precursor and their labels.
        """
        super().__init__(None, None, None)

        if extracted_precursor:
            self.data, self.slices = self.graphs_from_extracted_precursor(
                extracted_precursor
            )

    @staticmethod
    def mol_to_graph(molecule: MoleculeContainer, label: float) -> Data | None:
        """Takes a molecule as input, and converts the molecule to a PyTorch geometric
        graph, assigns the reward value (label) to the graph, and returns the graph.

        :param molecule: The input molecule.
        :param label: The label (solved/unsolved routes in the tree) of the molecule
            (precursor).
        :return: A PyTorch Geometric graph representation of a molecule.
        """
        if len(molecule) > 2:
            pyg = mol_to_pyg(molecule)
            if pyg:
                pyg.y = torch.tensor([label])
                return pyg

        return None

    def graphs_from_extracted_precursor(
        self, extracted_precursor: dict[str, float]
    ) -> tuple[Data, dict]:
        """Converts the extracted from the search trees precursor to the PyTorch geometric
        graphs.

        :param extracted_precursor: The dictionary with the extracted from the built
            search trees precursor and their labels.
        :return: The PyTorch geometric graphs and slices.
        """
        processed_data = []
        for smi, label in extracted_precursor.items():
            mol = smiles(smi)
            pyg = self.mol_to_graph(mol, label)
            if pyg:
                processed_data.append(pyg)
        data, slices = self.collate(processed_data)
        return data, slices


def _convert_ranking_item(item: tuple[str, str]) -> tuple[Data, str] | None:
    """Convert a (product_smiles, rule_id_str) pair into a labelled PyG graph.

    Module-level function so it can be pickled by :class:`ProcessPoolExecutor`.
    Returns ``(pyg_graph, product_smiles)`` or ``None`` on failure.
    """
    product_smi, rule_id_str = item
    try:
        molecule = smiles(product_smi)
        pyg_graph = mol_to_pyg(molecule)
        if pyg_graph is not None:
            pyg_graph.y_rules = torch.tensor([int(rule_id_str)], dtype=torch.long)
            return pyg_graph, product_smi
    except Exception:
        return None
    return None


class RankingPolicyDataset(InMemoryDataset):
    """Ranking policy network dataset.

    Reads a policy training mapping file (TSV with ``product_smiles`` and
    ``rule_id`` columns) produced by :func:`extract_rules_from_reactions`.
    """

    def __init__(self, policy_data_path: str, output_path: str, num_workers: int = 0):
        """Initializes a ranking policy network dataset.

        :param policy_data_path: Path to the policy training mapping file
            (``*_policy_data.tsv``) generated during rule extraction.
        :param output_path: Path where the cached PyG dataset will be saved.
        :param num_workers: CPU workers for parallel graph conversion.
            ``0`` auto-detects via :func:`default_num_workers`.
        """
        super().__init__(None, None, None)

        self.policy_data_path = policy_data_path
        self.output_path = output_path
        self.num_workers = num_workers
        self._product_keys: list[str] | None = None

        if output_path and os.path.exists(output_path):
            data, slices, product_keys, self._sf_handle = load_pyg_dataset(output_path)
            self.data, self.slices = data, slices
            self._product_keys = product_keys
        else:
            self.data, self.slices = self.prepare_data()

    @property
    def num_classes(self) -> int:
        return self._infer_num_classes(self._data.y_rules)

    def prepare_data(self) -> tuple[Data, dict[str, Tensor]]:
        """Reads the policy mapping file and converts product SMILES to PyG graphs.

        :return: The collated PyTorch Geometric data and slices.
        """
        # 1. Read TSV into a list of (smiles, rule_id) pairs
        items: list[tuple[str, str]] = []
        with open(self.policy_data_path, encoding="utf-8") as f:
            f.readline()  # skip header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                items.append((parts[0], parts[1]))

        # 2. Convert molecules to PyG graphs
        # Parallel only helps for large datasets; for small ones the process
        # spawn + PyG Data pickling overhead dominates (especially on macOS).
        _MIN_ITEMS_FOR_PARALLEL = 5000
        workers = self.num_workers if self.num_workers > 0 else default_num_workers()
        if len(items) < _MIN_ITEMS_FOR_PARALLEL:
            workers = 1
        list_of_graphs: list[Data] = []
        product_keys: list[str] = []

        def _collect_serial():
            for item in tqdm(
                items,
                desc="Building policy dataset: ",
                bar_format="{desc}{n}/{total} [{elapsed}]",
            ):
                result = _convert_ranking_item(item)
                if result is not None:
                    list_of_graphs.append(result[0])
                    product_keys.append(result[1])

        if workers <= 1:
            _collect_serial()
        else:
            try:
                for result in tqdm(
                    process_pool_map_stream(
                        items,
                        _convert_ranking_item,
                        max_workers=workers,
                    ),
                    total=len(items),
                    desc=f"Building policy dataset ({workers} workers): ",
                    bar_format="{desc}{n}/{total} [{elapsed}]",
                ):
                    if result is not None:
                        list_of_graphs.append(result[0])
                        product_keys.append(result[1])
            except BrokenProcessPool:
                logging.warning(
                    "A worker process crashed — falling back to serial. "
                    "This is usually caused by a malformed SMILES triggering "
                    "a C-level error in chython."
                )
                list_of_graphs.clear()
                product_keys.clear()
                _collect_serial()

        self._product_keys = product_keys
        data, slices = self.collate(list_of_graphs)
        if self.output_path:
            save_pyg_dataset(self.output_path, data, slices, product_keys=product_keys)

        return data, slices


class FilteringPolicyDataset(InMemoryDataset):
    """Filtering policy network dataset."""

    def __init__(
        self,
        molecules_path: str,
        reaction_rules_path: str,
        output_path: str,
        num_cpus: int,
    ) -> None:
        """Initializes a policy network dataset object.

        :param molecules_path: The path to the file containing the molecules for
            reaction rule appliance.
        :param reaction_rules_path: The path to the file containing the reaction rules.
        :param output_path: The output path to the file where policy network dataset
            will be stored.
        :param num_cpus: The number of CPUs to be used for the dataset preparation.
        :return: None.
        """
        super().__init__(None, None, None)

        self.molecules_path = molecules_path
        self.reaction_rules_path = reaction_rules_path
        self.output_path = output_path
        self.num_cpus = num_cpus
        self.batch_size = 100

        if output_path and os.path.exists(output_path):
            data, slices, _, self._sf_handle = load_pyg_dataset(output_path)
            self.data, self.slices = data, slices
        else:
            self.data, self.slices = self.prepare_data()

    @property
    def num_classes(self) -> int:
        return self._data.y_rules.shape[1]

    def prepare_data(self) -> tuple[Data, dict]:
        """Prepares data by loading reaction rules, initializing Ray, preprocessing the
        molecules, collating the data, and returning the data and slices.

        :return: The PyTorch geometric graphs and slices.
        """

        ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True)
        try:
            reaction_rules = load_reaction_rules(self.reaction_rules_path)
            reaction_rules_ids = ray.put(reaction_rules)

            to_process = Queue(maxsize=self.batch_size * self.num_cpus)
            processed_data = []
            results_ids = [
                preprocess_filtering_policy_molecules.remote(
                    to_process, reaction_rules_ids
                )
                for _ in range(self.num_cpus)
            ]

            with open(self.molecules_path, encoding="utf-8") as inp_data:
                for molecule in tqdm(
                    inp_data.read().splitlines(),
                    desc="Number of molecules processed: ",
                    bar_format="{desc}{n} [{elapsed}]",
                ):

                    to_process.put(molecule)

            results = [graph for res in ray.get(results_ids) if res for graph in res]
            processed_data.extend(results)
        finally:
            ray.shutdown()

        for pyg in processed_data:
            pyg.y_rules = pyg.y_rules.to_dense()
            pyg.y_priority = pyg.y_priority.to_dense()

        data, slices = self.collate(processed_data)
        if self.output_path:
            save_pyg_dataset(self.output_path, data, slices)

        return data, slices


def reaction_rules_appliance(
    molecule: MoleculeContainer, reaction_rules: list[Reactor]
) -> tuple[list[int], list[int]]:
    """Applies each reaction rule from the list of reaction rules to a given molecule
    and returns the indexes of the successfully applied regular and prioritized reaction
    rules.

    :param molecule: The input molecule.
    :param reaction_rules: The list of reaction rules.
    :return: The two lists of indexes of successfully applied regular reaction rules and
        priority reaction rules.
    """

    applied_rules, priority_rules = [], []
    for i, rule in enumerate(reaction_rules):

        rule_applied = False
        rule_prioritized = False

        try:
            for reaction in rule([molecule]):
                for prod in reaction.products:
                    tmp_prod = prod.copy()
                    tmp_prod.remove_coordinate_bonds(keep_to_terminal=False)
                    tmp_prod.kekule()
                    if tmp_prod.check_valence():
                        break
                    rule_applied = True

                    # check priority rules
                    if len(reaction.products) > 1:
                        # check coupling retro manual
                        if all(len(mol) > 6 for mol in reaction.products):
                            if (
                                sum(len(mol) for mol in reaction.products)
                                - len(reaction.reactants[0])
                                < 6
                            ):
                                rule_prioritized = True
                    else:
                        # check cyclization retro manual
                        if sum(len(mol.sssr) for mol in reaction.products) < sum(
                            len(mol.sssr) for mol in reaction.reactants
                        ):
                            rule_prioritized = True
            #
            if rule_applied:
                applied_rules.append(i)
                #
                if rule_prioritized:
                    priority_rules.append(i)
        except Exception as e:
            logging.debug(e)
            continue

    return applied_rules, priority_rules


@ray.remote
def preprocess_filtering_policy_molecules(
    to_process: Queue, reaction_rules: list[Reactor]
) -> list[Data | None]:
    """Preprocesses a list of molecules by applying reaction rules and converting
    molecules into PyTorch geometric graphs. Successfully applied reaction rules are
    converted to binary vectors for policy network training.

    :param to_process: The queue containing SMILES of molecules to be converted to the
        training data.
    :param reaction_rules: The list of reaction rules.
    :return: The list of PyGraph objects.
    """

    pyg_graphs = []
    while True:
        try:
            molecule = smiles(to_process.get(timeout=30))
            if not isinstance(molecule, MoleculeContainer):
                continue

            # reaction reaction_rules application
            applied_rules, priority_rules = reaction_rules_appliance(
                molecule, reaction_rules
            )

            y_rules = torch.sparse_coo_tensor(
                [applied_rules],
                torch.ones(len(applied_rules)),
                (len(reaction_rules),),
                dtype=torch.uint8,
            )
            y_priority = torch.sparse_coo_tensor(
                [priority_rules],
                torch.ones(len(priority_rules)),
                (len(reaction_rules),),
                dtype=torch.uint8,
            )

            y_rules = torch.unsqueeze(y_rules, 0)
            y_priority = torch.unsqueeze(y_priority, 0)

            pyg_graph = mol_to_pyg(molecule)
            if not pyg_graph:
                continue
            pyg_graph.y_rules = y_rules
            pyg_graph.y_priority = y_priority
            pyg_graphs.append(pyg_graph)

        except Empty:
            break

    return pyg_graphs


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
