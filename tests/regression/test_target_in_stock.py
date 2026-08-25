"""A target already in the catalogue must be reported, not planned."""

import csv
from pathlib import Path

from synplan.mcts.search import run_search


def test_purchasable_target_is_skipped_and_flagged(tmp_path, monkeypatch):
    """run_search reports a catalogue hit instead of spending a search on it."""
    calls = []

    class _NoTree:
        def __init__(self, **kwargs):
            calls.append(kwargs["target"])
            raise AssertionError("a purchasable target must not reach the search")

    monkeypatch.setattr("synplan.mcts.search.Tree", _NoTree)
    monkeypatch.setattr(
        "synplan.mcts.search.load_building_blocks", lambda *a, **k: {"CCN"}
    )
    monkeypatch.setattr("synplan.mcts.search.load_reaction_rules", lambda *a, **k: [])
    monkeypatch.setattr("synplan.mcts.search.load_policy_function", lambda *a, **k: None)
    monkeypatch.setattr(
        "synplan.mcts.search.load_evaluation_function", lambda *a, **k: None
    )

    targets = tmp_path / "targets.smi"
    targets.write_text("CCN\n")
    run_search(
        targets_path=str(targets),
        search_config={"max_iterations": 1, "silent": True},
        policy_config=None,
        evaluation_config=None,
        reaction_rules_path="unused",
        building_blocks_path="unused",
        results_root=str(tmp_path / "out"),
    )

    assert not calls, "the search was built for a target that is already purchasable"
    stats = list(csv.DictReader((tmp_path / "out" / "tree_search_stats.csv").open()))
    assert stats[0]["target_in_stock"] == "True"
    assert stats[0]["num_routes"] == "0"
