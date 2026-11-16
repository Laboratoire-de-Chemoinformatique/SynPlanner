"""Wall-clock budgets are optional independently of search strategy."""

import pytest
from test_tree_stats import build_tree

from synplan.mcts.config import TreeConfig
from synplan.mcts.tree import ALGORITHMS


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_null_time_limit_across_algorithms(algorithm):
    tree = build_tree(algorithm=algorithm, max_iterations=2)
    tree.config.max_time = None
    tree.run()
    assert tree.curr_iteration <= 2
    assert not tree.time_limit_reached()


def test_finite_deadline_stops_before_another_iteration(monkeypatch):
    import synplan.mcts.tree as module

    monkeypatch.setattr(module, "time", lambda: 100.0)
    tree = build_tree()
    assert not tree.time_limit_reached()
    tree.start_time = 80.0
    assert tree.time_limit_reached()
    tree.config.max_time = None
    assert not tree.time_limit_reached()
    tree.config.max_time = 10
    with pytest.raises(StopIteration, match="Time limit exceeded"):
        next(tree)
    assert tree.curr_iteration == 0
    assert tree.stop_reason == "max_time"


def test_default_limit_and_iteration_stopping_reason():
    assert TreeConfig().max_time == 600
    tree = build_tree(max_iterations=2)
    tree.config.max_time = None
    tree.run()
    assert tree.stop_reason == "max_iterations"
    assert tree.to_stats_dict()["stop_reason"] == "max_iterations"
