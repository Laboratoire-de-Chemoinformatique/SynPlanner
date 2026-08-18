from chython import smiles as read_smiles
from chython.containers import MoleculeContainer

from synplan.chem.building_blocks import BuildingBlockStock, molecule_to_inchi_key
from synplan.chem.precursor import Precursor
from synplan.chem.reaction.routes.io import make_tree_json
from synplan.mcts.node import Node
from synplan.utils.visualisation import (
    extract_routes,
    get_route_svg,
    get_route_svg_from_json,
    render_svg,
)


def make_mol(n: int) -> MoleculeContainer:
    molecule = MoleculeContainer()
    prev = None
    for _ in range(n):
        atom = molecule.add_atom("C")
        if prev is not None:
            molecule.add_bond(prev, atom, 1)
        prev = atom
    return molecule


def test_render_svg_can_use_transparent_boxes_with_thicker_borders():
    box_colors = {"target": "#98EEFF"}

    solid_molecule = read_smiles("CCO")
    solid_molecule.meta["status"] = "target"
    solid_svg = render_svg((), [[solid_molecule]], box_colors)

    transparent_molecule = read_smiles("CCO")
    transparent_molecule.meta["status"] = "target"
    transparent_svg = render_svg(
        (), [[transparent_molecule]], box_colors, box_solid=False
    )

    assert (
        'stroke="#98EEFF" stroke-width=".005" '
        'fill="#98EEFF" fill-opacity="0.30"' in solid_svg
    )
    assert 'stroke="#98EEFF" stroke-width=".005" fill="none"' in transparent_svg
    assert 'fill="#98EEFF"' not in transparent_svg


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

    routes_json = make_tree_json(_MockRouteMetadataTree(), reactions=routes_dict)
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

    routes_json = make_tree_json(_MockRouteMetadataTree(), reactions=routes_dict)
    svg = get_route_svg_from_json(
        routes_json,
        7,
        labeled=True,
        box_solid=False,
    )

    assert "<svg" in svg
    assert "policy:42" in svg
    assert "priority:0" in svg
    assert 'stroke="#98EEFF" stroke-width=".005" fill="none"' in svg
    assert 'fill="#98EEFF"' not in svg


def test_extract_routes_uses_root_to_terminal_steps():
    target = Precursor(make_mol(7))
    product = Precursor(make_mol(8))
    tree = _MockTree()
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


def test_make_json_uses_tree_inchikey_stock_for_leaf_flags():
    class StockTree:
        def __init__(self):
            stocked = read_smiles("CCCCCCC")
            self.config = _MockConfig()
            self.building_blocks = BuildingBlockStock(
                frozenset({molecule_to_inchi_key(stocked)}),
                "inchikey",
            )

        def route_details(self, node_id: int) -> dict:
            assert node_id == 7
            return {"steps": [{"node_id": 2, "rule_id": 1, "rule_source": "policy"}]}

    routes_dict = {7: {0: read_smiles("CCCCCCC.CCCCCCCO>>CCCCCCCCCCCCCCO")}}

    route = make_tree_json(StockTree(), reactions=routes_dict)[7]
    leaves = route["children"][0]["children"]

    assert sorted(node["in_stock"] for node in leaves) == [False, True]
