"""Tests for safetensors PyG dataset cache helpers."""

import torch
from torch_geometric.data import Data

from synplan.utils.cache import load_pyg_dataset, save_pyg_dataset


def test_pyg_dataset_cache_roundtrip_uses_safetensors_keys(tmp_path):
    """Cache loading should work with safetensors safe_open.keys()."""
    data = Data(
        x=torch.tensor([[1], [2]], dtype=torch.uint8),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y_rules=torch.tensor([3], dtype=torch.long),
    )
    slices = {
        "x": torch.tensor([0, 2], dtype=torch.long),
        "edge_index": torch.tensor([0, 2], dtype=torch.long),
        "y_rules": torch.tensor([0, 1], dtype=torch.long),
    }
    cache_path = tmp_path / "dataset.safetensors"

    save_pyg_dataset(cache_path, data, slices, product_keys=["CCO"])
    loaded_data, loaded_slices, product_keys, handle = load_pyg_dataset(cache_path)

    assert torch.equal(loaded_data.x, data.x)
    assert torch.equal(loaded_data.edge_index, data.edge_index)
    assert torch.equal(loaded_data.y_rules, data.y_rules)
    assert loaded_slices.keys() == slices.keys()
    assert product_keys == ["CCO"]
    assert handle is not None
