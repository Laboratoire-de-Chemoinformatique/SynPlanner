from types import SimpleNamespace

import pytest

from synplan.mcts.tree import Tree


class _TreeLike:
    """Route state only; the walks under test are bound from the real class."""

    route_node_ids = Tree.route_node_ids
    route_steps = Tree.route_steps
    route_to_node = Tree.route_to_node

    def __init__(self, parents):
        self.parents = parents
        self.nodes = {node_id: SimpleNamespace(node_id=node_id) for node_id in parents}


@pytest.fixture
def tree_like():
    return _TreeLike({1: 0, 2: 1, 3: 2})


def test_route_node_ids_are_root_to_terminal(tree_like):
    assert tree_like.route_node_ids(3) == (1, 2, 3)


def test_route_nodes_and_steps_preserve_route_order(tree_like):
    nodes = tree_like.route_to_node(3)
    steps = list(tree_like.route_steps(3))

    assert [node.node_id for node in nodes] == [1, 2, 3]
    assert [(before.node_id, after.node_id) for before, after in steps] == [
        (1, 2),
        (2, 3),
    ]


def test_root_only_and_empty_routes_have_no_steps(tree_like):
    assert [node.node_id for node in tree_like.route_to_node(1)] == [1]
    assert list(tree_like.route_steps(1)) == []
    assert tree_like.route_to_node(0) == []
    assert list(tree_like.route_steps(0)) == []


def test_missing_parent_preserves_key_error(tree_like):
    del tree_like.parents[2]
    with pytest.raises(KeyError):
        tree_like.route_node_ids(3)
