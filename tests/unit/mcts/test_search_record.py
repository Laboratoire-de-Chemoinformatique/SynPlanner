"""The search record: a finished search written out, and read back into routes."""

import itertools
import json

import pytest
from test_tree_stats import FakeReactor, build_tree, make_mol

from synplan.mcts.record import read_search_record, write_search_record
from synplan.mcts.tree import Tree


@pytest.fixture(scope="module")
def searched() -> Tree:
    tree = build_tree()
    tree.run()
    assert tree.winning_nodes, "the record tests need a search that found something"
    return tree


@pytest.fixture(scope="module")
def record(searched, tmp_path_factory):
    path = tmp_path_factory.mktemp("record") / "tree.json"
    return read_search_record(write_search_record(searched, path))


def twin_tree() -> Tree:
    """A search whose every disconnection makes the same molecule twice.

    Two equal precursors are two precursors: the route that consumes the second
    one is not the route that consumes the first, and interning the file by
    SMILES must not fuse them on the way back.
    """

    sizes = itertools.count(8)

    def twins():
        size = next(sizes)
        return [make_mol(size), make_mol(size)]

    tree = build_tree(
        rules=[(0.5, FakeReactor(twins), 0)], expand_deeper=True, max_iterations=8
    )
    return tree.run()


def test_the_record_rebuilds_the_routes_the_search_found(searched, record):
    live, back = searched.routes(), record.routes()

    assert [route.provenance.tree_node_id for route in back] == [
        route.provenance.tree_node_id for route in live
    ]
    for one, other in zip(live, back):
        assert [str(step.reaction) for step in other] == [
            str(step.reaction) for step in one
        ]
        assert [step.origin for step in other] == [step.origin for step in one]
        assert other.unresolved == one.unresolved
        assert other.solved is one.solved
        # unrounded values in the file: the same route, scored the same
        assert other.provenance.search_score == one.provenance.search_score


def test_the_record_answers_the_readouts_the_search_did(searched, record):
    assert record.target == searched.nodes[1].curr_precursor.molecule
    assert record.winning_nodes == searched.winning_nodes
    assert record.winning_rule_ranks() == searched.winning_rule_ranks()
    assert record.stats["branching_profile"] == searched.branching_profile()
    assert record.stats["routes_found_at"] == searched.stats.routes_found_at
    for key, value in searched.to_stats_dict().items():
        assert record.stats[key] == value
    for node_id in searched.winning_nodes:
        assert record.route_details(node_id) == searched.route_details(node_id)


def test_the_record_is_not_a_tree(record):
    """It records what happened; it cannot search."""

    assert not isinstance(record, Tree)
    assert not hasattr(record, "expansion_function")
    assert not hasattr(record, "building_blocks")


def test_the_record_carries_the_graph_and_no_routes(searched, tmp_path):
    written = json.loads(write_search_record(searched, tmp_path / "t.json").read_text())

    assert set(written) == {
        "schema",
        "target",
        "molecules",
        "nodes",
        "winning",
        "stats",
    }
    assert len(written["nodes"]) == len(searched.nodes)
    # molecules interned once, nodes holding indices into them
    assert all(isinstance(smiles, str) for smiles in written["molecules"])
    assert len(set(written["molecules"])) == len(written["molecules"])
    assert all(
        index < len(written["molecules"])
        for node in written["nodes"]
        for index in node["new"] + node["expand"]
    )


def test_two_equal_precursors_come_back_as_two(tmp_path):
    tree = twin_tree()
    record = read_search_record(write_search_record(tree, tmp_path / "twins.json"))

    made = record.nodes[2].new_precursors
    assert [str(p.molecule) for p in made] == [str(p.molecule) for p in made[:1]] * 2
    assert made[0] is not made[1]
    # the child expands the second one, and it is the very object its parent made
    assert record.nodes[3].precursors_to_expand[0] is made[1]
    assert tree.nodes[3].precursors_to_expand[0] is tree.nodes[2].new_precursors[1]


def test_the_record_gzips_itself(searched, tmp_path):
    plain = write_search_record(searched, tmp_path / "t.json")
    zipped = write_search_record(searched, tmp_path / "t.json.gz")

    assert zipped.stat().st_size < plain.stat().st_size
    assert (
        read_search_record(zipped).nodes.keys()
        == read_search_record(plain).nodes.keys()
    )


def test_a_file_that_is_not_a_search_record_says_so(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema": "synplan-routes/1", "routes": []}))

    with pytest.raises(ValueError, match="search record"):
        read_search_record(path)
