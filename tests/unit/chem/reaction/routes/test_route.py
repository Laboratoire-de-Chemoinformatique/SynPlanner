import json
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from chython import smiles as read_smiles

from synplan.chem.precursor import Precursor
from synplan.chem.reaction.reactor import Reaction
from synplan.chem.reaction.routes.io import read_routes_json, write_routes_json
from synplan.chem.reaction.routes.io.json import molecule_key
from synplan.chem.reaction.routes.representation.container import RouteCGRContainer
from synplan.chem.reaction.routes.route import MoleculePosition, Route, Step
from synplan.chem.utils import safe_canonicalization
from synplan.mcts.node import Node
from synplan.mcts.tree import Tree
from synplan.utils.routedraw import ROLE_STYLE

# Ethyl benzoate from benzoic acid and ethanol, the acid in turn from toluene.
# Atom maps are shared across the steps, as a reactor's precursors would be.
TARGET = "[CH3:1][CH2:2][O:3][C:4](=[O:5])[c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
ACID = "[OH:3][C:4](=[O:5])[c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
ETHANOL = "[CH3:1][CH2:2][OH:12]"
TOLUENE = "[CH3:4][c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
OXYGEN = "[O:3]=[O:5]"

_ROUTES_JSON = Path(__file__).resolve().parents[4] / "data" / "routes_mol_1.json"


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


# Two branches of two steps each, merged and then capped. Given in the order the
# search produces -- branches interleaved -- so re-linearisation has work to do.
# Only which product feeds which step matters here, not the chemistry.
_CONVERGENT = (
    (("C", "CC"), "CCC"),
    (("N", "CN"), "CCN"),
    (("CCC", "CCCC"), "CCCCC"),
    (("CCN", "CCCN"), "CCCCN"),
    (("CCCCC", "CCCCN"), "CCCCCN"),
    (("CCCCCN", "O"), "CCCCCCN"),
)

_DISC = re.compile(
    r'<circle cx="(-?[\d.]+)" cy="(-?[\d.]+)" r="10.5"[^>]*/>'
    r'<text x="-?[\d.]+" y="-?[\d.]+" class="sp-num">(\d+)</text>'
)
_TARGET_CAPTION = re.compile(
    r'<text x="(-?[\d.]+)" y="(-?[\d.]+)" class="sp-tag" fill="[^"]*">TARGET</text>'
)
_RECT = re.compile(
    r'<rect x="(-?[\d.]+)" y="(-?[\d.]+)" width="-?[\d.]+" height="([\d.]+)"[^>]*/>'
)


def convergent_steps() -> tuple[Step, ...]:
    """Fresh molecules each call: drawing rewrites their coordinates in place.

    One object per molecule, shared between the step that makes it and the step
    that consumes it -- the link the search hands out and this route relies on.
    """

    mols = {
        name: read_smiles(name) for name in {n for r, p in _CONVERGENT for n in (*r, p)}
    }
    return tuple(
        Step(
            Reaction([mols[name] for name in reactants], [mols[product]]),
            mols[product],
        )
        for reactants, product in _CONVERGENT
    )


def feeders(steps) -> dict[int, list[int]]:
    """``{index: indices of the steps whose product it consumes}``."""

    by_product = {id(step.product): i for i, step in enumerate(steps)}
    return {
        i: [
            by_product[id(mol)]
            for mol in step.reaction.reactants
            if id(mol) in by_product
        ]
        for i, step in enumerate(steps)
    }


def branch(sources: dict[int, list[int]], index: int) -> set[int]:
    """``index`` and everything feeding it, however deep."""

    return {index}.union(*(branch(sources, s) for s in sources[index]), set())


@pytest.fixture
def convergent_route() -> Route:
    return Route(steps=convergent_steps())


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
    assert solved_route.provenance.tree_node_id == 3
    assert solved_route.provenance.search_score == pytest.approx(tree.route_score(3))
    assert [step.reaction for step in solved_route] == list(tree.synthesis_route(3))
    assert solved_route.steps[0].origin.rule_key == "policy:2"
    assert solved_route.steps[1].origin.rule_key == "policy:1"


def test_a_step_disconnects_a_product_of_its_own_reaction():
    """The one fact the route never guesses: which product the step is about."""

    reaction = Reaction([read_smiles("CCO")], [read_smiles("CC=O")])
    with pytest.raises(ValueError):
        Step(reaction, read_smiles("CC=O"))  # equal, but not the same object


def test_route_from_tree_rejects_unknown_node():
    with pytest.raises(KeyError):
        Route.from_tree(_StubTree(), 99)


def test_a_solved_route_has_nothing_unresolved(solved_route):
    assert solved_route.solved
    assert solved_route.unresolved == frozenset()
    assert "solved" in repr(solved_route)


def test_the_unresolved_leaves_are_the_precursors_left_to_expand(unsolved_route):
    tree = _StubTree(solved=False)

    assert not unsolved_route.solved
    assert unsolved_route.unresolved == frozenset(
        molecule_key(precursor.molecule)
        for precursor in tree.nodes[2].precursors_to_expand
    )
    assert "unsolved (1 unresolved)" in repr(unsolved_route)


def test_tree_hands_back_route_objects_best_score_first():
    tree = _StubTree()
    routes = tree.routes()

    assert [route.provenance.tree_node_id for route in routes] == [3]
    assert all(isinstance(route, Route) for route in routes)
    scores = [route.provenance.search_score for route in routes]
    assert scores == sorted(scores, reverse=True)


def test_tree_routes_can_widen_to_unsolved_search_leaves():
    tree = _StubTree(solved=False)

    assert tree.routes() == []
    unsolved = tree.routes(solved_only=False)
    assert [route.provenance.tree_node_id for route in unsolved] == [2]
    assert not unsolved[0].solved


def test_unsolved_route_draws_its_unresolved_leaf_red(unsolved_route, solved_route):
    fill, stroke, _ = ROLE_STYLE["oos"]
    svg = unsolved_route.svg(align=False)

    assert svg.startswith("<svg")
    assert svg.count("NOT IN STOCK") == len(unsolved_route.unresolved)
    assert f'fill="{fill}" stroke="{stroke}"' in svg
    assert "NOT IN STOCK" not in solved_route.svg(align=False)


def test_route_json_round_trip_keeps_steps_origins_and_stock_verdicts(
    solved_route, unsolved_route
):
    for route in (solved_route, unsolved_route):
        restored = Route.from_json(route.to_json())

        assert [str(step.reaction) for step in restored] == [
            str(step.reaction) for step in route
        ]
        assert restored.solved is route.solved
        assert restored.unresolved == route.unresolved
        assert restored.steps[0].origin == route.steps[0].origin


def test_route_json_round_trip_survives_atom_mapping_reconciliation(solved_route):
    restored = Route.from_json(solved_route.to_json(reconcile_atom_mapping=True))

    assert len(restored) == len(solved_route)
    assert isinstance(restored.route_cgr(), RouteCGRContainer)


def test_route_cgr_delegates_to_the_route_cgr_container(solved_route):
    cgr = solved_route.route_cgr()

    assert isinstance(cgr, RouteCGRContainer)
    assert len(cgr) >= len(solved_route.target)


def test_reactions_dict_is_the_routes_dict_adapter(solved_route):
    assert solved_route.reactions_dict == dict(
        enumerate(step.reaction for step in solved_route)
    )


def test_a_search_that_never_expanded_yields_no_route():
    """The childless root is not a zero-step route."""

    tree = _StubTree()
    tree.children = {node_id: set() for node_id in tree.nodes}
    tree.winning_nodes = []

    routes = tree.routes(solved_only=False)

    assert 1 not in [route.provenance.tree_node_id for route in routes]
    assert all(len(route) > 0 for route in routes)


def test_a_route_needs_a_step():
    with pytest.raises(ValueError):
        Route(steps=())


def test_steps_that_do_not_make_one_molecule_are_not_a_route():
    """Two unconsumed products is two routes, not one."""

    first = read_smiles("CCO")
    second = read_smiles("CCN")
    steps = (
        Step(Reaction([read_smiles("CC=O")], [first]), first),
        Step(Reaction([read_smiles("CC#N")], [second]), second),
    )
    with pytest.raises(ValueError):
        Route(steps=steps)


def test_position_names_where_a_molecule_sits(solved_route, unsolved_route):
    acid, ethanol, toluene = (read_smiles(s) for s in (ACID, ETHANOL, TOLUENE))

    assert solved_route.position(solved_route.target) is MoleculePosition.TARGET
    assert solved_route.position(acid) is MoleculePosition.INTERMEDIATE
    assert solved_route.position(toluene) is MoleculePosition.LEAF
    assert solved_route.position(ethanol) is MoleculePosition.LEAF

    assert unsolved_route.position(unsolved_route.target) is MoleculePosition.TARGET
    assert unsolved_route.position(acid) is MoleculePosition.LEAF
    assert molecule_key(acid) in unsolved_route.unresolved
    assert molecule_key(ethanol) not in unsolved_route.unresolved

    with pytest.raises(KeyError):
        solved_route.position(read_smiles("CCCCCCCCCCCC"))


def test_position_takes_a_molecule_spelled_another_way(solved_route):
    """The route spells its molecules canonically; the question need not."""

    kekule = read_smiles("CCOC(=O)C1=CC=CC=C1")

    assert str(kekule) != str(solved_route.target)
    assert molecule_key(kekule) == molecule_key(solved_route.target)
    assert solved_route.position(kekule) is MoleculePosition.TARGET


def test_json_round_trip_answers_position_identically(solved_route, unsolved_route):
    for route in (solved_route, unsolved_route):
        restored = Route.from_json(route.to_json())
        molecules = [
            mol
            for step in route
            for mol in (*step.reaction.reactants, *step.reaction.products)
        ]

        assert [restored.position(mol) for mol in molecules] == [
            route.position(mol) for mol in molecules
        ]


def test_every_step_is_fed_by_leaves_or_lower_numbered_steps(
    convergent_route, solved_route
):
    """The topological invariant: nothing is consumed before it is made."""

    for route in (convergent_route, solved_route):
        for index, sources in feeders(route.steps).items():
            assert all(source < index for source in sources)


def test_each_branch_of_a_convergent_route_is_contiguous(convergent_route):
    raw = convergent_steps()
    sources = feeders(raw)
    assert any(
        sorted(branch(sources, i)) != list(range(min(b), max(b) + 1))
        for i in sources
        if (b := branch(sources, i))
    ), "the search order under test is already contiguous, so this proves nothing"

    sources = feeders(convergent_route.steps)
    for index in sources:
        block = sorted(branch(sources, index))
        assert block == list(range(block[0], block[-1] + 1))


def test_step_position_and_the_disc_all_carry_the_number(convergent_route):
    svg = convergent_route.svg(align=False)
    discs = _DISC.findall(svg)

    assert sorted(int(d[2]) for d in discs) == list(range(1, len(convergent_route) + 1))

    caption = _TARGET_CAPTION.search(svg)
    box_x, box_y = float(caption.group(1)) - 1, float(caption.group(2)) + 5
    height = next(
        float(r[2])
        for r in _RECT.findall(svg)
        if abs(float(r[0]) - box_x) < 0.11 and abs(float(r[1]) - box_y) < 0.11
    )
    on_target = [d for d in discs if abs(float(d[1]) - (box_y + height / 2)) < 0.11]
    assert [int(d[2]) for d in on_target] == [len(convergent_route)]


def test_a_pickled_route_does_not_carry_the_catalogue():
    tree = _StubTree()
    tree.building_blocks = frozenset(f"C{'C' * n}O" for n in range(20_000))

    assert len(pickle.dumps(Route.from_tree(tree, 3))) < 100_000


# --------------------------------------------------------------------------- #
# the routes file, in routes and out
# --------------------------------------------------------------------------- #


def test_routes_write_themselves_and_read_back(tmp_path, solved_route, unsolved_route):
    """No dict of dicts on either side of the file."""

    path = tmp_path / "routes.json"
    result = write_routes_json([solved_route, unsolved_route], path)
    assert result.diagnostics == ()

    back = read_routes_json(path, as_routes=True)
    assert [route.solved for route in back] == [True, False]
    for one, other in zip((solved_route, unsolved_route), back):
        assert [str(step.reaction) for step in other] == [
            str(step.reaction) for step in one
        ]
        assert other.unresolved == one.unresolved
        assert other.steps[0].origin == one.steps[0].origin


def test_a_route_carries_its_own_origins(tmp_path, solved_route):
    with pytest.raises(TypeError, match="already carries its step origins"):
        write_routes_json([solved_route], tmp_path / "routes.json", tree=_StubTree())


def test_the_reader_hands_back_one_shape(tmp_path, solved_route):
    path = tmp_path / "routes.json"
    write_routes_json([solved_route], path)

    with pytest.raises(ValueError, match="one shape"):
        read_routes_json(path, to_dict=True, as_routes=True)


def test_a_routes_dict_keyed_by_position_says_what_it_wanted(tmp_path, solved_route):
    """The keys of a routes_dict exported with a tree are its node ids."""

    with pytest.raises(ValueError, match="not by its position"):
        write_routes_json(
            {0: solved_route.reactions_dict},
            tmp_path / "routes.json",
            tree=_StubTree(),
        )


# --------------------------------------------------------------------------- #
# the guesses the route used to make
# --------------------------------------------------------------------------- #


def test_a_multi_product_route_file_reads_back_with_its_own_root_as_target():
    """`products[0]` picked whichever fragment the SMILES happened to start with."""

    routes = json.loads(_ROUTES_JSON.read_text(encoding="utf-8"))
    assert routes

    for route_id, route_json in routes.items():
        route = Route.from_json(route_json)
        assert route.target == read_smiles(route_json["smiles"]), route_id


def test_a_file_in_another_tools_spelling_reads_back_with_the_right_verdicts():
    """The verdict rides with its own molecule node, not with a SMILES string.

    Nothing here is written the way SynPlanner writes it: the reaction is
    Kekule, the acid starts from its hydroxyl, and the children are in the
    opposite order to the reactants. The purchasable leaf is still the toluene.
    """

    route = Route.from_json(
        {
            "type": "mol",
            "smiles": "OC(=O)c1ccccc1",
            "children": [
                {
                    "type": "reaction",
                    "smiles": "C1=CC=CC=C1C.O=O>>OC(=O)C1=CC=CC=C1",
                    "children": [
                        {"type": "mol", "smiles": "O=O", "in_stock": False},
                        {"type": "mol", "smiles": "C1=CC=CC=C1C", "in_stock": True},
                    ],
                }
            ],
        }
    )

    purchasable = {
        str(leaf)
        for leaf in route.leaves()
        if molecule_key(leaf) not in route.unresolved
    }
    assert purchasable == {"c1cc(ccc1)C"}
    assert route.unresolved == frozenset({"O=O"})
    assert not route.solved
    assert str(route.target) == "c1cc(ccc1)C(O)=O"


def test_a_molecule_chython_cannot_canonicalise_is_kept_as_the_file_wrote_it():
    """Reading is a trust boundary, not a validator: keep it, count it, draw it."""

    route = Route.from_json(
        {
            "type": "mol",
            "smiles": "Brc1cccc1",
            "children": [
                {
                    "type": "reaction",
                    "smiles": "c1cccc1.BrBr>>Brc1cccc1",
                    "children": [
                        {"type": "mol", "smiles": "c1cccc1", "in_stock": True},
                        {"type": "mol", "smiles": "BrBr", "in_stock": False},
                    ],
                }
            ],
        }
    )

    assert {str(leaf) for leaf in route.leaves()} == {"c1cccc1", "BrBr"}
    assert route.provenance.uncanonical == 2  # the ring and the product it makes
    assert route.unresolved == frozenset({"BrBr"})
    assert route.svg()


def symmetric_route() -> Route:
    """Biphenyl from two bromobenzenes, each brominated from benzene.

    Two steps, one product SMILES: only identity tells them apart.
    """

    biphenyl = read_smiles("c1ccc(-c2ccccc2)cc1")
    steps = []
    halves = []
    for _ in range(2):
        half = read_smiles("Brc1ccccc1")
        halves.append(half)
        steps.append(
            Step(Reaction([read_smiles("c1ccccc1"), read_smiles("BrBr")], [half]), half)
        )
    steps.append(Step(Reaction(halves, [biphenyl]), biphenyl))
    return Route(steps=tuple(steps))


def test_a_symmetric_disconnection_keeps_its_target_and_gets_a_disc_per_step():
    route = symmetric_route()

    assert len(route) == 3
    assert str(route.target) == str(read_smiles("c1ccc(-c2ccccc2)cc1"))
    discs = sorted(int(d[2]) for d in _DISC.findall(route.svg(align=False)))
    assert discs == [1, 2, 3]


def kekulized_route() -> Route:
    """A solved route whose aromatics are written in Kekule form."""

    target = read_smiles("c1ccc(C(=O)OCC)cc1")
    acid = read_smiles("c1ccc(C(=O)O)cc1")
    toluene = read_smiles("Cc1ccccc1")
    for mol in (target, acid, toluene):
        mol.kekule()
    return Route(
        steps=(
            Step(Reaction([toluene, read_smiles("O=O")], [acid]), acid),
            Step(Reaction([acid, read_smiles("CCO")], [target]), target),
        )
    )


def test_a_kekulized_route_stays_solved_through_json():
    route = kekulized_route()
    assert route.solved

    restored = Route.from_json(route.to_json())

    assert restored.solved, sorted(restored.unresolved)
    # reading canonicalises, so a Kekule file comes back aromatic: the same
    # molecule, spelled the one way the route now keys everything by.
    assert restored.target == safe_canonicalization(route.target.copy())
    assert str(restored.target) != str(route.target)


def test_to_json_leaves_the_route_it_was_called_on_alone():
    route = kekulized_route()
    before = [
        str(mol)
        for step in route
        for mol in (*step.reaction.reactants, *step.reaction.products)
    ]

    route.to_json()

    assert [
        str(mol)
        for step in route
        for mol in (*step.reaction.reactants, *step.reaction.products)
    ] == before
