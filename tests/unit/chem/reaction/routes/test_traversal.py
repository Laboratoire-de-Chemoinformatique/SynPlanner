from types import SimpleNamespace

import pytest

from synplan.chem.reaction.routes.traversal import (
    iter_route_nodes,
    iter_route_steps,
    route_node_ids,
)


@pytest.fixture
def tree_like():
    nodes = {node_id: SimpleNamespace(node_id=node_id) for node_id in (1, 2, 3)}
    return SimpleNamespace(nodes=nodes, parents={1: 0, 2: 1, 3: 2})


def test_route_node_ids_are_root_to_terminal():
    assert route_node_ids({1: 0, 2: 1, 3: 2}, 3) == (1, 2, 3)


def test_iter_route_nodes_and_steps_preserve_route_order(tree_like):
    nodes = list(iter_route_nodes(tree_like, 3))
    steps = list(iter_route_steps(tree_like, 3))

    assert [node.node_id for node in nodes] == [1, 2, 3]
    assert [(before.node_id, after.node_id) for before, after in steps] == [
        (1, 2),
        (2, 3),
    ]


def test_root_only_and_empty_routes_have_no_steps(tree_like):
    assert [node.node_id for node in iter_route_nodes(tree_like, 1)] == [1]
    assert list(iter_route_steps(tree_like, 1)) == []
    assert list(iter_route_nodes(tree_like, 0)) == []
    assert list(iter_route_steps(tree_like, 0)) == []


def test_missing_parent_preserves_key_error():
    with pytest.raises(KeyError):
        route_node_ids({1: 0, 2: 1}, 3)
