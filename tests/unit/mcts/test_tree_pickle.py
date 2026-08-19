from __future__ import annotations

import pickle

from chython import smarts, smiles

from synplan.chem.precursor import Precursor
from synplan.chem.reaction import CanonicalRetroReactor
from synplan.mcts.node import Node
from synplan.mcts.tree import Tree
from synplan.utils.config import TreeConfig


class _Policy:
    def __init__(self, rule):
        self.rule = rule

    def predict_reaction_rules(self, _precursor, _reaction_rules):
        yield 1.0, self.rule, 0


class _Evaluator:
    def evaluate_node(self, **_kwargs):
        return 0.0


def _reactor(rule_smarts):
    rule = smarts(rule_smarts)
    return CanonicalRetroReactor(
        patterns=tuple(rule.reactants),
        products=tuple(rule.products),
        delete_atoms=False,
    )


def test_tree_save_pickle_disables_tqdm_and_round_trips(tmp_path):
    tree = Tree.__new__(Tree)
    tree._tqdm = object()
    file_path = tmp_path / "tree.pkl"

    assert tree.save_pickle(file_path) is None
    assert tree._tqdm is None
    assert file_path.is_file()

    with file_path.open("rb") as file:
        loaded_tree = pickle.load(file)

    assert isinstance(loaded_tree, Tree)
    assert loaded_tree._tqdm is None
    assert loaded_tree.bond_constraints.active is False
    assert loaded_tree.bonds_state == {}


def test_legacy_precursor_and_node_pickles_backfill_empty_constraint_state():
    precursor = Precursor(smiles("CCCC"))
    del precursor.target_atom_provenance
    node = Node(precursors_to_expand=(precursor,), new_precursors=(precursor,))
    del node.remaining_required_bonds

    loaded_node = pickle.loads(pickle.dumps(node))

    assert loaded_node.remaining_required_bonds == frozenset()
    assert loaded_node.curr_precursor.target_atom_provenance.pairs == frozenset()


def test_pre_feature_tree_pickle_resumes_unconstrained():
    target = smiles("[CH3:1][CH2:2][CH2:3][OH:4]")
    cleavage = _reactor("[C;D2:1]-[O;D1:2]>>[C;D1:1].[O;D0:2]")
    tree = Tree(
        target=target,
        config=TreeConfig(
            algorithm="breadth_first",
            max_iterations=2,
            max_tree_size=20,
            max_time=5,
            max_depth=2,
            min_mol_size=10,
            silent=True,
            enable_pruning=False,
            search_strategy="expansion_first",
        ),
        reaction_rules=[cleavage],
        building_blocks=set(),
        expansion_function=_Policy(cleavage),
        evaluation_function=_Evaluator(),
    )

    del tree._bond_constraints
    seen_precursors = set()
    for node in tree.nodes.values():
        del node.remaining_required_bonds
        for precursor in (*node.precursors_to_expand, *node.new_precursors):
            if id(precursor) in seen_precursors:
                continue
            seen_precursors.add(id(precursor))
            del precursor.target_atom_provenance
        for precursor in node.curr_precursor.prev_precursors:
            if id(precursor) in seen_precursors:
                continue
            seen_precursors.add(id(precursor))
            del precursor.target_atom_provenance

    loaded_tree = pickle.loads(pickle.dumps(tree))
    loaded_tree.run()

    assert loaded_tree.bonds_state == {}
    assert loaded_tree.winning_nodes
