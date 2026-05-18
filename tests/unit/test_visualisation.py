from chython import smiles as read_smiles
from chython.containers import MoleculeContainer

from synplan.chem.precursor import Precursor
from synplan.chem.reaction_routes.io import make_json
from synplan.mcts.node import Node
from synplan.utils.visualisation import get_route_svg, get_route_svg_from_json


def make_mol(n: int) -> MoleculeContainer:
    molecule = MoleculeContainer()
    prev = None
    for _ in range(n):
        atom = molecule.add_atom("C")
        if prev is not None:
            molecule.add_bond(prev, atom, 1)
        prev = atom
    return molecule


class _MockConfig:
    min_mol_size = 6


class _MockTree:
    def __init__(self):
        target = Precursor(make_mol(7))
        intermediate = Precursor(make_mol(8))

        self.config = _MockConfig()
        self.building_blocks = frozenset()
        self.nodes = {
            1: Node(
                precursors_to_expand=(target,),
                new_precursors=(target,),
            ),
            2: Node(
                precursors_to_expand=(intermediate,),
                new_precursors=(intermediate,),
                rule_key="policy:0",
            ),
        }
        self.parents = {1: 0, 2: 1}
        self.winning_nodes = []


class _MockRouteMetadataTree:
    def route_details(self, node_id: int) -> dict:
        assert node_id == 7
        return {
            "steps": [
                {
                    "node_id": 2,
                    "rule_id": 42,
                    "rule_source": "policy",
                    "rule_key": "policy:42",
                },
                {
                    "node_id": 7,
                    "rule_id": 0,
                    "rule_source": "priority",
                    "rule_key": "priority:0",
                },
            ]
        }


def test_get_route_svg_unsolved_is_opt_in():
    tree = _MockTree()

    assert get_route_svg(tree, 2) is None

    svg = get_route_svg(tree, 2, labeled=True, allow_unsolved=True)
    assert svg is not None
    assert "<svg" in svg
    assert "policy:0" in svg


def test_make_json_attaches_rule_metadata_from_tree():
    routes_dict = {
        7: {
            0: read_smiles("[CH4:1].[OH2:2]>>[CH3:1][OH:2]"),
            1: read_smiles("[CH3:1][OH:2].[NH3:3]>>[CH3:1][NH2:3].[OH2:2]"),
        }
    }

    routes_json = make_json(routes_dict, tree=_MockRouteMetadataTree())
    root = routes_json[7]
    root_reaction = root["children"][0]
    expanded_child = next(
        child for child in root_reaction["children"] if child.get("children")
    )
    nested_reaction = expanded_child["children"][0]

    assert root_reaction["step_id"] == 1
    assert root_reaction["rule_source"] == "policy"
    assert root_reaction["rule_key"] == "policy:42"
    assert nested_reaction["step_id"] == 0
    assert nested_reaction["rule_source"] == "priority"
    assert nested_reaction["rule_key"] == "priority:0"


def test_get_route_svg_from_json_can_render_rule_labels():
    routes_dict = {
        7: {
            0: read_smiles("[CH4:1].[OH2:2]>>[CH3:1][OH:2]"),
            1: read_smiles("[CH3:1][OH:2].[NH3:3]>>[CH3:1][NH2:3].[OH2:2]"),
        }
    }

    routes_json = make_json(routes_dict, tree=_MockRouteMetadataTree())
    svg = get_route_svg_from_json(routes_json, 7, labeled=True)

    assert "<svg" in svg
    assert "policy:42" in svg
    assert "priority:0" in svg
