from types import SimpleNamespace

import pytest
from chython import smiles as read_smiles

from synplan.chem.precursor import Precursor
from synplan.chem.reaction.routes.representation.container import RouteCGRContainer
from synplan.chem.reaction.routes.route import Route
from synplan.mcts.node import Node
from synplan.mcts.tree import Tree

# Ethyl benzoate from benzoic acid and ethanol, the acid in turn from toluene.
# Atom maps are shared across the steps, as a reactor's precursors would be.
TARGET = "[CH3:1][CH2:2][O:3][C:4](=[O:5])[c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
ACID = "[OH:3][C:4](=[O:5])[c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
ETHANOL = "[CH3:1][CH2:2][OH:12]"
TOLUENE = "[CH3:4][c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
OXYGEN = "[O:3]=[O:5]"


class _StubTree:
    """Route state only; the Tree methods under test are bound from the real class."""

    synthesis_route = Tree.synthesis_route
    route_score = Tree.route_score
    route_details = Tree.route_details
    routes = Tree.routes

    def __init__(self, solved: bool = True):
        target = Precursor(read_smiles(TARGET))
        acid = Precursor(read_smiles(ACID))
        self.nodes = {
            1: Node(
                precursors_to_expand=(target,),
                new_precursors=(target,),
                total_value=0.5,
            ),
            2: Node(
                precursors_to_expand=(acid,),
                new_precursors=(acid, Precursor(read_smiles(ETHANOL))),
                depth=1,
                total_value=0.4,
                rule_id=1,
                rule_source="policy",
                rule_key="policy:1",
            ),
        }
        self.parents = {1: 0, 2: 1}
        self.children = {1: {2}, 2: set()}
        self.winning_nodes = []
        if solved:
            self.nodes[3] = Node(
                precursors_to_expand=(),
                new_precursors=(
                    Precursor(read_smiles(TOLUENE)),
                    Precursor(read_smiles(OXYGEN)),
                ),
                depth=2,
                total_value=0.9,
                rule_id=2,
                rule_source="policy",
                rule_key="policy:2",
            )
            self.parents[3] = 2
            self.children[2] = {3}
            self.children[3] = set()
            self.winning_nodes = [3]
        self.building_blocks = frozenset({str(read_smiles(TOLUENE))})
        self.config = SimpleNamespace(min_mol_size=6)
        self._route_scorer = None
        self._rescore_cache = {}


@pytest.fixture
def solved_route() -> Route:
    return Route.from_tree(_StubTree(), 3)


@pytest.fixture
def unsolved_route() -> Route:
    return Route.from_tree(_StubTree(solved=False), 2)


def test_route_from_tree_carries_steps_stock_and_score(solved_route):
    tree = _StubTree()

    assert len(solved_route) == 2
    assert str(solved_route.target) == str(read_smiles(TARGET))
    assert solved_route.route_id == 3
    assert solved_route.score == pytest.approx(tree.route_score(3))
    assert solved_route.steps == tuple(tree.synthesis_route(3))
    assert solved_route.step_meta[0]["rule_key"] == "policy:2"
    assert solved_route.step_meta[1]["rule_key"] == "policy:1"


def test_route_from_tree_rejects_unknown_node():
    with pytest.raises(KeyError):
        Route.from_tree(_StubTree(), 99)


def test_solved_route_has_no_dead_ends(solved_route):
    assert solved_route.solved
    assert solved_route.dead_ends == ()
    assert "solved" in repr(solved_route)


def test_unsolved_route_dead_ends_are_the_precursors_left_to_expand(unsolved_route):
    tree = _StubTree(solved=False)

    assert not unsolved_route.solved
    assert [str(mol) for mol in unsolved_route.dead_ends] == [
        str(precursor) for precursor in tree.nodes[2].precursors_to_expand
    ]
    assert "unsolved (1 dead end)" in repr(unsolved_route)


def test_tree_hands_back_route_objects_best_score_first():
    tree = _StubTree()
    routes = tree.routes()

    assert [route.route_id for route in routes] == [3]
    assert all(isinstance(route, Route) for route in routes)
    assert [route.score for route in routes] == sorted(
        (route.score for route in routes), reverse=True
    )


def test_tree_routes_can_widen_to_unsolved_search_leaves():
    tree = _StubTree(solved=False)

    assert tree.routes() == []
    unsolved = tree.routes(solved_only=False)
    assert [route.route_id for route in unsolved] == [2]
    assert not unsolved[0].solved


def test_unsolved_route_draws_its_dead_end_red(unsolved_route, solved_route):
    svg = unsolved_route.svg(align=False)

    assert svg.startswith("<svg")
    assert "NOT IN STOCK" in svg
    assert 'fill="#fdf2f2" stroke="#d06a6a"' in svg
    assert "#d06a6a" not in solved_route.svg(align=False)


def test_route_json_round_trip_keeps_steps_metadata_and_stock_verdicts(
    solved_route, unsolved_route
):
    for route in (solved_route, unsolved_route):
        restored = Route.from_json(route.to_json(), route_id=route.route_id)

        assert [str(step) for step in restored.steps] == [
            str(step) for step in route.steps
        ]
        assert restored.solved is route.solved
        assert [str(mol) for mol in restored.dead_ends] == [
            str(mol) for mol in route.dead_ends
        ]
        assert restored.step_meta[0]["rule_key"] == route.step_meta[0]["rule_key"]
        assert restored.route_id == route.route_id


def test_route_json_round_trip_survives_atom_mapping_reconciliation(solved_route):
    restored = Route.from_json(solved_route.to_json(reconcile_atom_mapping=True))

    assert len(restored) == len(solved_route)
    assert isinstance(restored.route_cgr(), RouteCGRContainer)


def test_route_cgr_delegates_to_the_route_cgr_container(solved_route):
    cgr = solved_route.route_cgr()

    assert isinstance(cgr, RouteCGRContainer)
    assert len(cgr) >= len(solved_route.target)


def test_reactions_dict_is_the_routes_dict_adapter(solved_route):
    assert solved_route.reactions_dict == dict(enumerate(solved_route.steps))
