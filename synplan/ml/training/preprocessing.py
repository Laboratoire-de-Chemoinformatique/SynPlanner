"""Module containing functions for preparation of the training sets for policy and value
network."""

import logging
import os
from abc import ABC
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import torch
from chython import smiles
from chython.containers import MoleculeContainer
from chython.exceptions import InvalidAromaticRing
from chython.reactor import Reactor
from torch import Tensor
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.transforms import ToUndirected
from tqdm.auto import tqdm

from synplan.utils.cache import load_pyg_dataset, save_pyg_dataset
from synplan.utils.loading import load_reaction_rules
from synplan.utils.parallel import chunked, default_num_workers, process_pool_map_stream

logger = logging.getLogger(__name__)

_RANKING_POLICY_PARALLEL_BATCH_SIZE = 1000
RankingGraphPayload = tuple[
    list[list[int]],
    list[list[int]],
    list[list[float]],
    int,
    str,
]


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


def _ranking_graph_to_payload(graph: Data, product_smi: str) -> RankingGraphPayload:
    """Serialize a ranking graph without torch objects for process-pool IPC.

    PyTorch tensors crossing a multiprocessing boundary are transferred via
    shared-memory file descriptors. Large ranking batches can therefore hit
    ``Too many open files`` while unpickling results in the parent process.
    Keep worker payloads as plain Python lists and reconstruct tensors in the
    parent instead.
    """
    return (
        graph.x.tolist(),
        graph.edge_index.tolist(),
        graph.edge_attr.tolist(),
        int(graph.y_rules.reshape(-1)[0].item()),
        product_smi,
    )


def _ranking_payload_to_graph(payload: RankingGraphPayload) -> tuple[Data, str]:
    """Rebuild a PyG graph from the plain-Python worker payload."""
    x, edge_index, edge_attr, rule_id, product_smi = payload
    graph = Data(
        x=torch.tensor(x, dtype=torch.uint8),
        edge_index=torch.tensor(edge_index, dtype=torch.long).reshape(2, -1),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float).reshape(-1, 4),
    )
    graph.y_rules = torch.tensor([rule_id], dtype=torch.long)
    return graph, product_smi


def _convert_ranking_batch(
    batch: list[tuple[str, str]],
) -> tuple[int, list[RankingGraphPayload]]:
    """Convert a batch of ranking policy rows inside one worker process.

    Returning batches amortizes ProcessPool IPC overhead.
    The first tuple element is the number of input rows, so the parent can update
    progress even when some rows fail conversion and produce no graph.
    """
    converted: list[RankingGraphPayload] = []
    for item in batch:
        result = _convert_ranking_item(item)
        if result is not None:
            converted.append(_ranking_graph_to_payload(result[0], result[1]))
    return len(batch), converted


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
                batches = chunked(items, _RANKING_POLICY_PARALLEL_BATCH_SIZE)
                results = process_pool_map_stream(
                    batches,
                    _convert_ranking_batch,
                    max_workers=workers,
                    max_pending=max(1, workers * 2),
                )
                with tqdm(
                    total=len(items),
                    desc=f"Building policy dataset ({workers} workers): ",
                    bar_format="{desc}{n}/{total} [{elapsed}]",
                ) as progress:
                    for processed_count, batch_results in results:
                        progress.update(processed_count)
                        for payload in batch_results:
                            graph, product_smi = _ranking_payload_to_graph(payload)
                            list_of_graphs.append(graph)
                            product_keys.append(product_smi)
            except BrokenProcessPool as exc:
                raise RuntimeError(
                    "Parallel ranking policy dataset build failed before "
                    "completion. Refusing silent serial fallback because it "
                    "hides worker crashes and can turn this stage into a "
                    "single-core run; rerun with --workers 1 only for "
                    "diagnostics."
                ) from exc

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
        """Preprocesses molecules by applying reaction rules and converting them to
        PyTorch geometric graphs using a process pool.

        :return: The PyTorch geometric graphs and slices.
        """

        with open(self.molecules_path, encoding="utf-8") as inp_data:
            molecules = inp_data.read().splitlines()

        workers = self.num_cpus if self.num_cpus > 0 else default_num_workers()

        def _collect_serial():
            # Ensure module-level worker state is set for the serial path.
            global _worker_state
            _worker_state = load_reaction_rules(self.reaction_rules_path)
            try:
                results = []
                for mol_smi in tqdm(
                    molecules,
                    desc="Number of molecules processed: ",
                    bar_format="{desc}{n} [{elapsed}]",
                ):
                    result = _preprocess_filtering_policy_molecule(mol_smi)
                    if result is not None:
                        results.append(result)
                return results
            finally:
                _worker_state = None

        if workers <= 1:
            processed_data = _collect_serial()
        else:
            try:
                processed_data = []
                for result in tqdm(
                    process_pool_map_stream(
                        molecules,
                        _preprocess_filtering_policy_molecule,
                        max_workers=workers,
                        initializer=_init_filtering_worker,
                        initargs=(self.reaction_rules_path,),
                    ),
                    total=len(molecules),
                    desc=f"Number of molecules processed ({workers} workers): ",
                    bar_format="{desc}{n} [{elapsed}]",
                ):
                    if result is not None:
                        processed_data.append(result)
            except BrokenProcessPool as exc:
                raise RuntimeError(
                    "Parallel filtering policy dataset build failed before "
                    "completion. Refusing silent serial fallback because it "
                    "hides worker crashes and can turn this stage into a "
                    "single-core run; rerun with --num_cpus 1 only for "
                    "diagnostics."
                ) from exc

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
            logger.debug(e)
            continue

    return applied_rules, priority_rules


# ---------------------------------------------------------------------------
# Worker state and functions for FilteringPolicyDataset parallel processing
# ---------------------------------------------------------------------------

_worker_state = None


def _init_filtering_worker(reaction_rules_path: str) -> None:
    """Initializer for each worker process in the filtering policy pool.

    Loads reaction rules once per worker and stores them in module-level
    ``_worker_state`` so that :func:`_preprocess_filtering_policy_molecule`
    can access them without re-loading on every call.
    """
    global _worker_state
    _worker_state = load_reaction_rules(reaction_rules_path)


def _preprocess_filtering_policy_molecule(molecule_smi: str) -> Data | None:
    """Preprocess a single molecule SMILES for filtering policy training.

    Applies all reaction rules loaded into ``_worker_state`` (set by
    :func:`_init_filtering_worker`) to the molecule, builds the sparse
    label tensors, and converts the molecule to a PyG graph.

    When running serially (no initializer), the function falls back to using
    the ``_worker_state`` that must have been set before calling.

    :param molecule_smi: SMILES string for the molecule.
    :return: A PyG :class:`Data` object with ``y_rules`` and ``y_priority``
        attributes, or ``None`` if parsing / conversion fails.
    """
    reaction_rules = _worker_state
    if reaction_rules is None:
        return None

    try:
        molecule = smiles(molecule_smi)
        if not isinstance(molecule, MoleculeContainer):
            return None

        # reaction rules application
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
            return None
        pyg_graph.y_rules = y_rules
        pyg_graph.y_priority = y_priority
        return pyg_graph

    except Exception:
        return None


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
