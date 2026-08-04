import pytest

from synplan.chem.reaction_rules import POLICY_SOURCE_NAME
from synplan.mcts import evaluation as evaluation_module
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


def test_shipped_strategies_are_instantiable():
    """A strategy overriding ``evaluate_node`` instead of ``_evaluate_node``
    stays abstract and raises TypeError only when someone actually plans."""
    for name in dir(evaluation_module):
        cls = getattr(evaluation_module, name)
        if (
            isinstance(cls, type)
            and issubclass(cls, EvaluationStrategy)
            and cls is not EvaluationStrategy
        ):
            assert not cls.__abstractmethods__, f"{name} is abstract"


def _shipped_strategies():
    return [
        cls
        for name in dir(evaluation_module)
        if isinstance(cls := getattr(evaluation_module, name), type)
        and issubclass(cls, EvaluationStrategy)
        and cls is not EvaluationStrategy
    ]


def test_no_strategy_overrides_the_public_entry_point():
    """The priority-rule short-circuit lives in the base ``evaluate_node``.

    Overriding ``_evaluate_node`` is the contract. Overriding ``evaluate_node``
    itself leaves priority-rule nodes to the configured evaluator instead of
    scoring them 1.0 — and unlike the abstract case, a subclass that overrides
    both instantiates cleanly and gets it wrong in silence.
    """
    offenders = [c.__name__ for c in _shipped_strategies() if "evaluate_node" in c.__dict__]
    assert not offenders, (
        f"{offenders} override evaluate_node; override _evaluate_node instead, "
        "or priority rules lose their score"
    )


@pytest.mark.parametrize("rule_source", [None, POLICY_SOURCE_NAME])
def test_non_priority_node_uses_configured_evaluator(rule_source):
    strategy = FixedEvaluationStrategy(score=0.25)
    node = make_node(rule_source=rule_source)

    assert strategy.evaluate_node(node=node, node_id=2, nodes={2: node}) == 0.25
    assert strategy.calls == 1
