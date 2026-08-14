from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from synplan.chem.building_blocks import BuildingBlockStock, inchi_to_inchi_key
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


def test_tree_unpickle_migrates_legacy_smiles_stock_without_rebinding_rollout():
    legacy = frozenset({"CCO"})
    evaluator = SimpleNamespace(rollout=SimpleNamespace(building_blocks=legacy))
    tree = Tree.__new__(Tree)

    tree.__setstate__({"building_blocks": legacy, "evaluator": evaluator})

    assert tree.building_blocks == BuildingBlockStock(legacy, "smiles")
    assert tree.evaluator.rollout.building_blocks is legacy


def test_tree_unpickle_rejects_raw_inchi_stock():
    tree = Tree.__new__(Tree)

    with pytest.raises(
        ValueError, match="raw InChI building-block stocks are unsupported"
    ):
        tree.__setstate__({"building_blocks": frozenset({"InChI=1S/H2O/h1H2"})})


def test_tree_unpickle_rejects_mixed_legacy_identity_stock():
    tree = Tree.__new__(Tree)

    with pytest.raises(
        ValueError, match="raw InChI building-block stocks are unsupported"
    ):
        tree.__setstate__({"building_blocks": frozenset({"InChI=1S/H2O/h1H2", "CCO"})})


def test_tree_typed_stock_pickle_round_trip_preserves_format_and_shared_object():
    stock = BuildingBlockStock(
        frozenset({inchi_to_inchi_key("InChI=1S/H2O/h1H2")}), "inchikey"
    )
    tree = Tree.__new__(Tree)
    tree._tqdm = None
    tree.building_blocks = stock
    tree.evaluator = SimpleNamespace(rollout=SimpleNamespace(building_blocks=stock))

    loaded = pickle.loads(pickle.dumps(tree))

    assert loaded.building_blocks.identity_format == "inchikey"
    assert loaded.building_blocks == stock
    assert loaded.evaluator.rollout.building_blocks is loaded.building_blocks


def test_tree_setstate_retains_existing_typed_stock_instance():
    stock = BuildingBlockStock(frozenset({"CCO"}), "smiles")
    tree = Tree.__new__(Tree)

    tree.__setstate__({"building_blocks": stock})

    assert tree.building_blocks is stock
