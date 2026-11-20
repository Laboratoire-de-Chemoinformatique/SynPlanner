"""MCTS route-lineage allocation and origin-aware persistence."""

import json

from chython import smarts, smiles

from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.reaction.reactor import (
    ATOM_ORIGINS_META,
    ROUTE_ATOM_MAPPING_CONTRACT,
)
from synplan.chem.reaction.routes.representation.route_cgr import build_route_cgr
from synplan.mcts.config import TreeConfig
from synplan.mcts.evaluation import EvaluationStrategy
from synplan.mcts.record import read_search_record, write_search_record
from synplan.mcts.tree import Tree


class _Policy:
    def __init__(self, candidates):
        self.candidates = candidates
        self.top_rules = len(candidates)

    def predict_reaction_rules(self, precursor, reaction_rules):
        yield from self.candidates


class _Evaluation(EvaluationStrategy):
    def _evaluate_node(self, node, node_id, nodes) -> float:
        return 0.0


def _reactor(rule_text: str, *, native: bool = True):
    rule = smarts(rule_text)
    return CanonicalRetroReactor(
        patterns=tuple(rule.reactants),
        products=tuple(rule.products),
        delete_atoms=False,
        reconcile_mapping=native,
    )


def _tree(reactors, target="CCCCCCC", building_blocks=(), forward_assessor=None):
    candidates = [
        (1.0 - index / 10, reactor, index) for index, reactor in enumerate(reactors)
    ]
    return Tree(
        target=smiles(target),
        config=TreeConfig(
            algorithm="breadth_first",
            max_iterations=3,
            max_tree_size=50,
            max_time=20,
            max_depth=3,
            min_mol_size=0,
            silent=True,
            enable_pruning=False,
            search_strategy="expansion_first",
        ),
        reaction_rules=reactors,
        building_blocks=set(building_blocks),
        expansion_function=_Policy(candidates),
        evaluation_function=_Evaluation(),
        forward_assessor=forward_assessor,
    )


class _ForwardAssessor:
    def __init__(self):
        self.last_stats = {}
        self.last_inputs = set()

    def assess_reactions(self, reactants, expected_products):
        assert all(
            set(molecule.meta[ATOM_ORIGINS_META]) == set(molecule)
            for candidate in reactants
            for molecule in candidate
        )
        assert all(
            set(molecule.meta[ATOM_ORIGINS_META]) == set(molecule)
            for molecule in expected_products
        )
        self.last_stats = {
            "cache_hits": 0,
            "cache_misses": 1,
            "model_predictions": 1,
            "inference_seconds": 0.0,
        }
        self.last_inputs = {"native-mapping-test"}
        return [{"alert": False, "rank": 1} for _ in reactants]


def test_alternative_children_reuse_the_parent_allocation_range():
    chloride = _reactor("[C;D1:1]-[C:2]>>[C;D1:1]-[Cl:3].[C:2]")
    bromide = _reactor("[C;D1:1]-[C:2]>>[C;D1:1]-[Br:3].[C:2]")
    tree = _tree([chloride, bromide])

    tree._expand_node(1)
    children = [tree.nodes[node_id] for node_id in sorted(tree.children[1])]

    assert tree.atom_mapping_contract == ROUTE_ATOM_MAPPING_CONTRACT
    assert tree.nodes[1].atom_mapping_state.map_ceiling == 7
    assert len(children) == 2
    assert {child.atom_mapping_state.map_ceiling for child in children} == {8}
    assert {child.atom_mapping_state.origin_ceiling for child in children} == {8}


def test_native_mapping_and_forward_prediction_share_child_state():
    reactor = _reactor("[C;D1:1]-[C:2]>>[C;D1:1]-[Cl:3].[C:2]")
    tree = _tree([reactor], forward_assessor=_ForwardAssessor())

    tree._expand_node(1)

    children = [tree.nodes[node_id] for node_id in sorted(tree.children[1])]
    assert children
    assert all(child.atom_mapping_state is not None for child in children)
    assert all(child.forward_prediction["rank"] == 1 for child in children)


def test_tree_rejects_mixed_baseline_and_native_rules():
    native = _reactor("[C:1]-[C:2]>>[C:1].[C:2]")
    baseline = _reactor("[C:1]-[C:2]>>[C:1].[C:2]", native=False)

    try:
        _tree([native, baseline])
    except ValueError as error:
        assert "cannot mix" in str(error)
    else:
        raise AssertionError("mixed mapping modes were accepted")


def test_native_search_record_round_trip_retains_direct_composition(tmp_path):
    reactor = _reactor("[c:1]-[Cl:2]>>[c:1]-[O:2]")
    phenol = str(smiles("c1ccccc1O"))
    tree = _tree([reactor], target="c1ccccc1Cl", building_blocks={phenol})
    tree.run()
    assert tree.winning_nodes

    record_path = write_search_record(tree, tmp_path / "native-search.json")
    raw = json.loads(record_path.read_text())
    record = read_search_record(record_path)
    route_id = record.winning_nodes[0]
    result = build_route_cgr(record, route_id, reconcile_atom_mapping=False)

    assert record.atom_mapping_contract == ROUTE_ATOM_MAPPING_CONTRACT
    assert all(
        not isinstance(stored, dict) or set(stored) == {"smiles", "origins"}
        for stored in raw["molecules"]
    )
    assert all("atom_mapping_state" in node for node in raw["nodes"])
    assert result.ok, result.diagnostic
    assert all(
        set(precursor.molecule.meta[ATOM_ORIGINS_META]) == set(precursor.molecule)
        for node in record.nodes.values()
        for precursor in node.new_precursors
    )
