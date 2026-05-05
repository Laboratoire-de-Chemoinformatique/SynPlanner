"""Pickle loading helpers for visualization entrypoints."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from synplan.mcts.tree import Tree


def ends_with_pickle_stop(path: Path) -> bool:
    size = path.stat().st_size
    if size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        return handle.read(1) == b"."


def load_tree(tree_pkl: Path) -> Tree:
    if not tree_pkl.exists():
        raise FileNotFoundError(f"Tree pickle not found: {tree_pkl}")
    if tree_pkl.stat().st_size == 0:
        raise ValueError(f"Tree pickle is empty: {tree_pkl}")
    if not ends_with_pickle_stop(tree_pkl):
        raise ValueError(
            "Tree pickle appears truncated (missing STOP opcode). "
            "Re-save the tree and try again."
        )

    try:
        with tree_pkl.open("rb") as handle:
            loaded = pickle.load(handle)
    except EOFError as exc:
        raise ValueError(
            "Tree pickle is incomplete or corrupted (unexpected EOF). "
            "Re-save the tree and try again."
        ) from exc
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Tree pickle could not be unpickled. Re-save the tree and try again."
        ) from exc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency while unpickling. "
            "Run this in the same environment where the tree was saved."
        ) from exc

    if hasattr(loaded, "tree"):
        return loaded.tree
    return loaded


def load_clusters(clusters_pkl: Optional[Path]) -> dict[str, dict]:
    if not clusters_pkl:
        return {}
    clusters_pkl = Path(clusters_pkl)
    if not clusters_pkl.exists():
        raise FileNotFoundError(f"Clusters pickle not found: {clusters_pkl}")
    if clusters_pkl.stat().st_size == 0:
        raise ValueError(f"Clusters pickle is empty: {clusters_pkl}")

    try:
        with clusters_pkl.open("rb") as handle:
            loaded = pickle.load(handle)
    except EOFError as exc:
        raise ValueError(
            "Clusters pickle is incomplete or corrupted (unexpected EOF). "
            "Re-save the clusters and try again."
        ) from exc
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Clusters pickle could not be unpickled. Re-save the clusters and try again."
        ) from exc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency while unpickling clusters. "
            "Run this in the same environment where the clusters were saved."
        ) from exc

    if not isinstance(loaded, dict):
        raise TypeError("Clusters pickle must contain a dict.")
    return loaded
