"""Likelihood ranking, root allocation, and discovery telemetry contracts."""

from math import log
from types import SimpleNamespace

import pytest

from synplan.chem.reaction.routes.quality import PolicyLikelihoodScorer
from synplan.chem.reaction.routes.route import RouteProvenance
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.config import TreeConfig
from synplan.mcts.evaluation import RandomEvaluationStrategy
from synplan.mcts.node import Node
from synplan.mcts.record import read_search_record, write_search_record
from synplan.mcts.tree import Tree


class EmptyPolicy:
    def predict_reaction_rules(self, *args):
        return iter(())


def make_tree(**kwargs):
    return Tree(
        mol_from_smiles("CCCCCC", clean2d=False),
        TreeConfig(silent=True, **kwargs),
        [],
        set(),
        EmptyPolicy(),
        RandomEvaluationStrategy(),
    )


def add_child(tree, parent, probability, *, solved=False):
    precursor = tree.nodes[1].curr_precursor
    child = Node(
        precursors_to_expand=() if solved else (precursor,), new_precursors=(precursor,)
    )
    tree._add_node(
        parent,
        child,
        policy_prob=probability * 3,
        rule_id=0,
        rule_source="policy",
        policy_probability=probability,
    )
    return tree.curr_tree_size - 1


def test_likelihood_uses_raw_probabilities_and_rejects_missing():
    tree = make_tree()
    first = add_child(tree, 1, 0.4)
    last = add_child(tree, first, 0.2, solved=True)
    assert tree.route_log_likelihood(last) == pytest.approx(log(0.4) + log(0.2))
    tree.nodes[last].policy_probability = None
    with pytest.raises(ValueError, match="policy probabilities"):
        tree.route_log_likelihood(last)


def test_likelihood_scorer_uses_policy_instead_of_search_values():
    low = SimpleNamespace(
        provenance=RouteProvenance(search_score=1.0, policy_log_likelihood=log(0.1))
    )
    high = SimpleNamespace(
        provenance=RouteProvenance(search_score=0.0, policy_log_likelihood=log(0.9))
    )
    scorer = PolicyLikelihoodScorer()
    assert scorer.rank([low, high])[:1] == [high]
    with pytest.raises(ValueError, match="recorded policy probabilities"):
        scorer.score(SimpleNamespace(provenance=RouteProvenance()))


def test_root_balanced_search_serves_each_root_before_revisiting(monkeypatch):
    tree = make_tree(algorithm="root_balanced", max_depth=5)
    expanded = []

    def expand(node_id):
        expanded.append(node_id)
        if node_id == 1:
            add_child(tree, node_id, 0.9)
            add_child(tree, node_id, 0.01)
        else:
            add_child(tree, node_id, 0.99)

    monkeypatch.setattr(tree, "_expand_node", expand)
    for _ in range(4):
        tree.algorithm.step()
    assert expanded[:3] == [1, 2, 3]
    assert expanded[3] == 4
    assert len(set(expanded)) == len(expanded)


def test_root_balanced_marks_all_solved_children_and_stops(monkeypatch):
    tree = make_tree(algorithm="root_balanced")

    def expand(node_id):
        add_child(tree, node_id, 0.4, solved=True)
        add_child(tree, node_id, 0.3, solved=True)

    monkeypatch.setattr(tree, "_expand_node", expand)
    tree.run()
    assert tree.winning_nodes == [2, 3]
    assert len(tree.stats.routes_found_at) == 2


def test_repeated_solution_notifications_are_not_new_discoveries(monkeypatch):
    tree = make_tree(max_iterations=4)
    child = add_child(tree, 1, 0.5, solved=True)

    def step():
        tree.algorithm._mark_solved(child)
        return True, [child]

    monkeypatch.setattr(tree.algorithm, "step", step)
    tree.run()
    assert len(tree.stats.routes_found_at) == 1
    assert tree.stats.iterations_without_expansion == 4


def test_raw_probability_round_trips_in_search_record(tmp_path):
    tree = make_tree()
    child = add_child(tree, 1, 0.25)
    path = tmp_path / "search.json.gz"
    write_search_record(tree, path)
    restored = read_search_record(path)
    assert restored.nodes[child].policy_probability == 0.25
    assert restored.route_log_likelihood(child) == log(0.25)
