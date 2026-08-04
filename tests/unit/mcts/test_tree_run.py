"""Tests for ``Tree.run()``.

``run()`` replaces the ``list(tree)`` idiom. These pin the three things callers
rely on: it returns the tree, it actually searches, and it does the same search
iterating by hand does.
"""

from test_tree_stats import build_tree

from synplan.mcts.tree import Tree


def test_run_returns_the_tree():
    """So ``extract_routes(Tree(...).run())`` chains."""
    tree = build_tree()
    assert tree.run() is tree


def test_tree_does_nothing_until_run():
    """Constructing a Tree must not search — the whole reason run() exists."""
    tree = build_tree()
    assert tree.curr_iteration == 0

    tree.run()
    assert tree.curr_iteration > 0


def test_run_matches_manual_iteration():
    """run() is exhausting the iterator, nothing more."""
    ran = build_tree()
    ran.run()

    iterated = build_tree()
    for _ in iterated:
        pass

    assert ran.curr_iteration == iterated.curr_iteration
    assert ran.curr_tree_size == iterated.curr_tree_size
    assert ran.found_a_route == iterated.found_a_route


def test_iteration_still_yields_per_step_results():
    """The documented early-stopping form must keep working."""
    tree = build_tree()

    steps = 0
    for is_solved, node_ids in tree:
        steps += 1
        assert isinstance(is_solved, bool)
        assert isinstance(node_ids, list)
        if steps == 3:
            break

    assert steps == 3
    assert tree.curr_iteration == 3


def test_run_respects_max_iterations():
    """run() stops on the configured limit rather than spinning."""
    tree: Tree = build_tree(max_iterations=5)
    tree.run()
    assert tree.curr_iteration <= 5
