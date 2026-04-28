"""RDKit compatibility layer for SynPlanner.

Provides conversion functions so RDKit users can pass RDKit Mol objects as
input (target molecules, building blocks) and receive RDKit Mol objects as
output (route molecules) without working with chython directly.

Requires rdkit to be installed.
"""

import itertools
import logging
from collections.abc import Iterable

from chython.containers import MoleculeContainer
from chython.exceptions import InvalidAromaticRing

from synplan.chem.utils import safe_canonicalization

logger = logging.getLogger(__name__)


def target_from_rdkit(
    rdkit_mol,
    standardize: bool = True,
    clean_stereo: bool = True,
    clean2d: bool = True,
) -> MoleculeContainer:
    """Convert an RDKit Mol to a chython MoleculeContainer for use as a
    search target.

    Applies the same standardization as :func:`~synplan.chem.utils.mol_from_smiles`:
    canonicalization, stereo cleaning, and 2D coordinate generation.

    :param rdkit_mol: An RDKit Mol or RWMol object.
    :param standardize: Whether to canonicalize the molecule.
    :param clean_stereo: Whether to remove stereo marks.
    :param clean2d: Whether to generate clean 2D coordinates.
    :return: A standardized MoleculeContainer ready for Tree search.
    """
    molecule = MoleculeContainer.from_rdkit(rdkit_mol)
    tmp = molecule.copy()
    try:
        tmp.remove_coordinate_bonds(keep_to_terminal=False)
        if standardize:
            tmp.canonicalize()
        if clean_stereo:
            tmp.clean_stereo()
        if clean2d:
            tmp.clean2d()
        molecule = tmp
    except InvalidAromaticRing:
        logging.warning(
            "chython was not able to standardize molecule due to invalid aromatic ring"
        )
    return molecule


def building_blocks_from_rdkit(rdkit_mols: Iterable) -> frozenset[str]:
    """Convert RDKit Mol objects to a building block set of canonical SMILES.

    The returned frozenset is compatible with ``Tree(building_blocks=...)``.

    :param rdkit_mols: Iterable of RDKit Mol objects.
    :return: Frozenset of canonical SMILES strings.
    """
    smiles_set: set[str] = set()
    for rdmol in rdkit_mols:
        mol = MoleculeContainer.from_rdkit(rdmol)
        mol = safe_canonicalization(mol)
        smiles_set.add(str(mol))
    return frozenset(smiles_set)


def route_to_rdkit(tree, node_id: int, keep_mapping: bool = True) -> list[dict]:
    """Convert a synthesis route to a flat list of retrosynthetic steps
    with RDKit Mol objects.

    Each step is a dict::

        {
            "target": rdkit.Chem.Mol,       # molecule being disconnected
            "precursors": [Mol, ...],        # resulting fragments
            "in_stock": [bool, ...],         # building block status per precursor
            "rule_id": int | None,           # reaction rule applied
            "rule_source": str | None,       # rule source namespace
            "rule_key": str | None,          # collision-safe "<source>:<id>"
            "policy_rank": int | None,       # 1-indexed Top-N policy position
            "depth": int,                    # step depth in the tree
        }

    :param tree: A solved Tree object.
    :param node_id: Winning node ID (must be in ``tree.winning_nodes``).
    :param keep_mapping: Preserve atom map numbers on RDKit atoms. Required
        to build atom-mapped RDKit ChemicalReaction objects from the molecules.
    :return: List of step dicts ordered from target to building blocks.
    """
    # collect node IDs along the path (root → winning node)
    path_ids: list[int] = []
    nid = node_id
    while nid:
        path_ids.append(nid)
        nid = tree.parents[nid]
    path_ids.reverse()

    steps = []
    for before_id, after_id in itertools.pairwise(path_ids):
        before_node = tree.nodes[before_id]
        after_node = tree.nodes[after_id]

        target_mol = before_node.curr_precursor.molecule.to_rdkit(
            keep_mapping=keep_mapping
        )

        precursor_mols = []
        in_stock = []
        for p in after_node.new_precursors:
            precursor_mols.append(p.molecule.to_rdkit(keep_mapping=keep_mapping))
            in_stock.append(
                p.is_building_block(tree.building_blocks, tree.config.min_mol_size)
            )

        steps.append(
            {
                "target": target_mol,
                "precursors": precursor_mols,
                "in_stock": in_stock,
                "rule_id": tree.nodes_rules.get(after_id),
                "rule_source": tree.nodes_rule_source.get(after_id),
                "rule_key": tree.nodes_rule_key.get(after_id),
                "policy_rank": tree.nodes_policy_rank.get(after_id),
                "depth": tree.nodes_depth.get(after_id, 0),
            }
        )

    return steps


def extract_routes_rdkit(tree, keep_mapping: bool = True) -> list[dict]:
    """Extract all winning routes as nested dicts with RDKit Mol objects.

    Mirrors :func:`~synplan.utils.visualisation.extract_routes` but each
    molecule node also carries an ``"mol"`` key with an RDKit Mol object.

    Structure::

        {
            "type": "mol",
            "smiles": str,
            "mol": rdkit.Chem.Mol,
            "in_stock": bool,
            "children": [{
                "type": "reaction",
                "children": [<mol nodes>, ...]
            }]
        }

    :param tree: A solved Tree object.
    :param keep_mapping: Preserve atom map numbers on RDKit atoms.
    :return: List of route tree dicts, one per winning node.
    """
    target_mol = tree.nodes[1].precursors_to_expand[0].molecule
    target_in_stock = tree.nodes[1].curr_precursor.is_building_block(
        tree.building_blocks, tree.config.min_mol_size
    )

    if not tree.winning_nodes:
        return [
            {
                "type": "mol",
                "smiles": str(target_mol),
                "mol": target_mol.to_rdkit(keep_mapping=keep_mapping),
                "in_stock": target_in_stock,
                "children": [],
            }
        ]

    routes_block = []
    for winning_node in tree.winning_nodes:
        # build molecule graph (same logic as extract_routes)
        graph: dict[MoleculeContainer, dict[str, object]] = {}
        mol_cache: dict[int, object] = {}  # id(MoleculeContainer) → RDKit Mol

        path_ids: list[int] = []
        nid = winning_node
        while nid:
            path_ids.append(nid)
            nid = tree.parents[nid]
        path_ids.reverse()

        for before_id, after_id in itertools.pairwise(path_ids):
            before_mol = tree.nodes[before_id].curr_precursor.molecule
            after_mols = [x.molecule for x in tree.nodes[after_id].new_precursors]
            graph[before_mol] = {
                "children": after_mols,
                "rule_key": tree.nodes_rule_key.get(after_id),
                "policy_rank": tree.nodes_policy_rank.get(after_id),
            }

            if id(before_mol) not in mol_cache:
                mol_cache[id(before_mol)] = before_mol.to_rdkit(
                    keep_mapping=keep_mapping
                )
            for m in after_mols:
                if id(m) not in mol_cache:
                    mol_cache[id(m)] = m.to_rdkit(keep_mapping=keep_mapping)

        def _build_node(
            molecule: MoleculeContainer, _graph=graph, _mol_cache=mol_cache
        ) -> dict:
            smi = str(molecule)
            rdkit_mol = _mol_cache.get(
                id(molecule),
                molecule.to_rdkit(keep_mapping=keep_mapping),
            )
            node = {
                "type": "mol",
                "smiles": smi,
                "mol": rdkit_mol,
                "in_stock": smi in tree.building_blocks
                or len(molecule) <= tree.config.min_mol_size,
            }
            reaction = graph.get(molecule)
            if reaction is not None:
                children = [_build_node(p) for p in reaction["children"]]
                reaction_node = {"type": "reaction", "children": children}
                if reaction.get("rule_key"):
                    reaction_node["rule_key"] = reaction["rule_key"]
                if reaction.get("policy_rank") is not None:
                    reaction_node["policy_rank"] = reaction["policy_rank"]
                node["children"] = [reaction_node]
            return node

        routes_block.append(_build_node(target_mol))

    return routes_block
