import pytest

from synplan.chem.reaction_rules import POLICY_SOURCE_NAME
from synplan.mcts.evaluation import EvaluationStrategy
from synplan.mcts.node import Node


class FixedEvaluationStrategy(EvaluationStrategy):
    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    def _evaluate_node(self, node, node_id, nodes) -> float:
        self.calls += 1
        return self.score


def make_node(rule_source: str | None) -> Node:
    return Node(
        precursors_to_expand=(),
        new_precursors=(),
        rule_source=rule_source,
    )


def test_priority_rule_node_receives_highest_score():
    strategy = FixedEvaluationStrategy(score=0.25)
    node = make_node(rule_source="ugi")

    assert strategy.evaluate_node(node=node, node_id=2, nodes={2: node}) == 1.0
    assert strategy.calls == 0


@pytest.mark.parametrize("rule_source", [None, POLICY_SOURCE_NAME])
def test_non_priority_node_uses_configured_evaluator(rule_source):
    strategy = FixedEvaluationStrategy(score=0.25)
    node = make_node(rule_source=rule_source)

    assert strategy.evaluate_node(node=node, node_id=2, nodes={2: node}) == 0.25
    assert strategy.calls == 1
