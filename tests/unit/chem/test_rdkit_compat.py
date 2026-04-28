"""Tests for synplan.chem.rdkit_compat — RDKit compatibility layer."""

import pytest
from chython import smiles
from chython.containers import MoleculeContainer
from rdkit import Chem

from synplan.chem.rdkit_compat import (
    building_blocks_from_rdkit,
    extract_routes_rdkit,
    route_to_rdkit,
    target_from_rdkit,
)

# ---------------------------------------------------------------------------
# target_from_rdkit
# ---------------------------------------------------------------------------


class TestTargetFromRdkit:
    def test_basic_molecule(self):
        rdmol = Chem.MolFromSmiles("CCO")
        result = target_from_rdkit(rdmol)
        assert isinstance(result, MoleculeContainer)
        assert len(result) == 3  # C, C, O (heavy atoms)

    def test_aromatic_molecule(self):
        rdmol = Chem.MolFromSmiles("c1ccccc1")
        result = target_from_rdkit(rdmol)
        assert isinstance(result, MoleculeContainer)
        assert len(result) == 6

    def test_complex_drug_molecule(self):
        smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
        rdmol = Chem.MolFromSmiles(smi)
        result = target_from_rdkit(rdmol)
        assert isinstance(result, MoleculeContainer)
        assert len(result) == 13  # heavy atoms

    def test_charged_molecule(self):
        rdmol = Chem.MolFromSmiles("[NH3+]CC([O-])=O")  # glycine zwitterion
        result = target_from_rdkit(rdmol)
        assert isinstance(result, MoleculeContainer)

    def test_stereo_molecule(self):
        rdmol = Chem.MolFromSmiles("C/C=C/[C@@H](O)F")
        result = target_from_rdkit(rdmol)
        assert isinstance(result, MoleculeContainer)

    def test_roundtrip_preserves_structure(self):
        """RDKit → chython → RDKit should give the same canonical SMILES."""
        original_smi = "c1ccc(CC(=O)O)cc1"
        rdmol = Chem.MolFromSmiles(original_smi)
        chython_mol = target_from_rdkit(rdmol)
        rdmol_back = chython_mol.to_rdkit(keep_mapping=False)
        assert Chem.MolToSmiles(rdmol) == Chem.MolToSmiles(rdmol_back)

    def test_no_standardization(self):
        rdmol = Chem.MolFromSmiles("CCO")
        result = target_from_rdkit(
            rdmol, standardize=False, clean_stereo=False, clean2d=False
        )
        assert isinstance(result, MoleculeContainer)


# ---------------------------------------------------------------------------
# building_blocks_from_rdkit
# ---------------------------------------------------------------------------


class TestBuildingBlocksFromRdkit:
    def test_basic(self):
        mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CC(=O)O", "c1ccccc1"]]
        result = building_blocks_from_rdkit(mols)
        assert isinstance(result, frozenset)
        assert len(result) == 3

    def test_deduplication(self):
        """Same molecule in different SMILES forms should deduplicate."""
        mols = [
            Chem.MolFromSmiles("CCO"),
            Chem.MolFromSmiles("OCC"),  # same molecule
        ]
        result = building_blocks_from_rdkit(mols)
        assert len(result) == 1

    def test_empty_input(self):
        result = building_blocks_from_rdkit([])
        assert result == frozenset()

    def test_smiles_are_canonical(self):
        """Output SMILES should match chython canonical form."""
        rdmol = Chem.MolFromSmiles("OCC")
        result = building_blocks_from_rdkit([rdmol])
        chython_mol = smiles("OCC")
        chython_mol.canonicalize()
        chython_mol.clean_stereo()
        assert str(chython_mol) in result


# ---------------------------------------------------------------------------
# route_to_rdkit — uses a mock tree
# ---------------------------------------------------------------------------


class _MockConfig:
    min_mol_size = 6


class _MockTree:
    """Minimal mock of Tree for testing route export.

    Builds a 2-step route:
        aspirin (node 1) → [salicylic acid, acetic anhydride] (node 2)
        salicylic acid (node 2) → [phenol, CO2] (node 3)
    """

    def __init__(self):
        from synplan.chem.precursor import Precursor

        self.config = _MockConfig()
        self.building_blocks = frozenset(
            {str(smiles("O=C(O)C")), str(smiles("Oc1ccccc1")), str(smiles("O=C=O"))}
        )

        # molecules
        aspirin = smiles("CC(=O)Oc1ccccc1C(=O)O")
        salicylic = smiles("OC(=O)c1ccccc1O")
        acetic_anh = smiles("CC(=O)OC(=O)C")
        phenol = smiles("Oc1ccccc1")
        co2 = smiles("O=C=O")

        # precursors
        p_aspirin = Precursor(aspirin)
        p_salicylic = Precursor(salicylic)
        p_acetic = Precursor(acetic_anh)
        p_phenol = Precursor(phenol)
        p_co2 = Precursor(co2)

        # Node 1: target = aspirin, to_expand = [aspirin]
        from synplan.mcts.node import Node

        node1 = Node(
            precursors_to_expand=(p_aspirin,),
            new_precursors=(),
        )
        # Node 2: target was aspirin, produced salicylic + acetic_anh
        # salicylic still needs expansion
        node2 = Node(
            precursors_to_expand=(p_salicylic,),
            new_precursors=(p_salicylic, p_acetic),
        )
        # Node 3: target was salicylic, produced phenol + co2
        # both are building blocks → solved
        node3 = Node(
            precursors_to_expand=(),
            new_precursors=(p_phenol, p_co2),
        )

        self.nodes = {1: node1, 2: node2, 3: node3}
        self.parents = {1: 0, 2: 1, 3: 2}
        self.children = {1: {2}, 2: {3}, 3: set()}
        self.winning_nodes = [3]
        self.nodes_rules = {2: 42, 3: 7}
        self.nodes_depth = {1: 0, 2: 1, 3: 2}

    def route_to_node(self, node_id):
        path = []
        nid = node_id
        while nid:
            path.append(nid)
            nid = self.parents[nid]
        return [self.nodes[nid] for nid in reversed(path)]


class TestRouteToRdkit:
    @pytest.fixture
    def mock_tree(self):
        return _MockTree()

    def test_returns_list_of_steps(self, mock_tree):
        steps = route_to_rdkit(mock_tree, 3)
        assert isinstance(steps, list)
        assert len(steps) == 2  # 2-step route

    def test_step_has_rdkit_mols(self, mock_tree):
        steps = route_to_rdkit(mock_tree, 3)
        for step in steps:
            assert isinstance(step["target"], Chem.rdchem.Mol)
            for mol in step["precursors"]:
                assert isinstance(mol, Chem.rdchem.Mol)

    def test_step_has_metadata(self, mock_tree):
        steps = route_to_rdkit(mock_tree, 3)
        assert steps[0]["rule_id"] == 42
        assert steps[0]["depth"] == 1
        assert steps[1]["rule_id"] == 7
        assert steps[1]["depth"] == 2

    def test_in_stock_flags(self, mock_tree):
        steps = route_to_rdkit(mock_tree, 3)
        # step 0: aspirin → [salicylic, acetic_anh]
        # salicylic is not small (>6 atoms) but not in BB set as written
        # acetic_anh: also >6 atoms, check BB set
        assert isinstance(steps[0]["in_stock"], list)
        assert len(steps[0]["in_stock"]) == 2

    def test_keep_mapping_true(self, mock_tree):
        steps = route_to_rdkit(mock_tree, 3, keep_mapping=True)
        # with mapping, atoms should have map numbers
        target = steps[0]["target"]
        has_mapping = any(a.GetAtomMapNum() > 0 for a in target.GetAtoms())
        assert has_mapping

    def test_keep_mapping_false(self, mock_tree):
        steps = route_to_rdkit(mock_tree, 3, keep_mapping=False)
        target = steps[0]["target"]
        has_mapping = any(a.GetAtomMapNum() > 0 for a in target.GetAtoms())
        assert not has_mapping


class TestExtractRoutesRdkit:
    @pytest.fixture
    def mock_tree(self):
        return _MockTree()

    def test_returns_list(self, mock_tree):
        routes = extract_routes_rdkit(mock_tree)
        assert isinstance(routes, list)
        assert len(routes) == 1  # one winning node

    def test_root_node_structure(self, mock_tree):
        routes = extract_routes_rdkit(mock_tree)
        root = routes[0]
        assert root["type"] == "mol"
        assert "smiles" in root
        assert "mol" in root
        assert isinstance(root["mol"], Chem.rdchem.Mol)
        assert "in_stock" in root
        assert "children" in root

    def test_nested_structure(self, mock_tree):
        routes = extract_routes_rdkit(mock_tree)
        root = routes[0]
        # root should have children (reaction node)
        assert len(root["children"]) == 1
        reaction = root["children"][0]
        assert reaction["type"] == "reaction"
        # reaction should have mol children
        for child in reaction["children"]:
            assert child["type"] == "mol"
            assert isinstance(child["mol"], Chem.rdchem.Mol)

    def test_all_mol_nodes_have_rdkit_mol(self, mock_tree):
        """Recursively check that every 'mol' node has an RDKit Mol."""
        routes = extract_routes_rdkit(mock_tree)

        def check_node(node):
            if node["type"] == "mol":
                assert isinstance(
                    node["mol"], Chem.rdchem.Mol
                ), f"Missing RDKit Mol for {node['smiles']}"
            for child in node.get("children", []):
                check_node(child)

        for route in routes:
            check_node(route)
