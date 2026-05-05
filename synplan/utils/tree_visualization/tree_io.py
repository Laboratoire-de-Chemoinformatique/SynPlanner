"""Pickle loading helpers for visualization entrypoints."""

from __future__ import annotations

import pickle
from pathlib import Path

from synplan.chem.reaction_routes.io import TreeWrapper
from synplan.mcts.tree import Tree


def load_pickle(path: Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_tree(tree_pkl: Path) -> Tree:
    loaded = load_pickle(tree_pkl)
    if isinstance(loaded, TreeWrapper):
        return loaded.tree
    if isinstance(loaded, Tree):
        return loaded
    raise TypeError("Tree pickle must contain a Tree or TreeWrapper.")


def load_clusters(clusters_pkl: Path | None) -> dict[str, dict]:
    return {} if clusters_pkl is None else load_pickle(clusters_pkl)
