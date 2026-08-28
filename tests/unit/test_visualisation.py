from chython import smiles as read_smiles
from chython.containers import MoleculeContainer

from synplan.chem.precursor import Precursor
from synplan.chem.reaction.routes import Route
from synplan.chem.reaction.routes.io import make_json
from synplan.mcts.node import Node
from synplan.utils.visualisation import extract_routes


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


def test_a_route_read_back_from_json_keeps_the_rules_that_made_it():
    """The rule keys `make_json` writes are what a reader needs to know which chemistry a
    step stands for, so they have to survive the trip back into a `Route`."""
    routes_dict = {
        7: {
            0: read_smiles("[CH4:1].[OH2:2]>>[CH3:1][OH:2]"),
            1: read_smiles("[CH3:1][OH:2].[NH3:3]>>[CH3:1][NH2:3].[OH2:2]"),
        }
    }

    routes_json = make_json(routes_dict, tree=_MockRouteMetadataTree())
    route = Route.from_json(routes_json[7])

    assert [step.origin.rule_key for step in route] == ["priority:0", "policy:42"]
    assert "<svg" in route.svg()


def test_extract_routes_uses_root_to_terminal_steps():
    target = Precursor(make_mol(7))
    product = Precursor(make_mol(8))
    tree = type("RouteTree", (), {})()
    tree.config = _MockConfig()
    tree.building_blocks = frozenset()
    tree.nodes = {
        1: Node(
            precursors_to_expand=(target,),
            new_precursors=(target,),
        ),
        2: Node(
            precursors_to_expand=(),
            new_precursors=(product,),
            rule_key="policy:1",
        ),
    }
    tree.parents = {1: 0, 2: 1}
    tree.winning_nodes = [2]

    routes = extract_routes(tree)

    assert routes[0]["smiles"] == str(target.molecule)
    reaction = routes[0]["children"][0]
    assert reaction["type"] == "reaction"
    assert reaction["children"][0]["smiles"] == str(product.molecule)
