from __future__ import annotations

import pickle

from synplan.mcts.tree import Tree


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
