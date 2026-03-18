"""Safetensors persistence for PyG InMemoryDataset (data, slices) pairs.

Replaces pickle-based torch.save/torch.load with safetensors for:
- Security: no arbitrary code execution on load
- Speed: memory-mapped loading (zero-copy on CPU)
"""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch_geometric.data.data import Data


def cache_digest(*input_paths: str | Path, extra: str = "") -> str:
    """Compute a short hex digest identifying the input data.

    Uses file basenames, line counts, and byte sizes — a fast O(n)
    scan that catches file renames, additions, removals, and most edits
    without reading full contents.
    """
    h = hashlib.sha256()
    for p in sorted(str(x) for x in input_paths):
        path = Path(p)
        h.update(path.name.encode())
        stat = path.stat()
        h.update(str(stat.st_size).encode())
        with open(path, "rb") as f:
            line_count = sum(1 for _ in f)
        h.update(str(line_count).encode())
    if extra:
        h.update(extra.encode())
    return h.hexdigest()[:16]


def _flatten(data: Data, slices: dict) -> dict[str, torch.Tensor]:
    """Flatten a collated (data, slices) pair into a flat {str: Tensor} dict.

    Keys are namespaced: ``data/<attr>`` for Data tensors,
    ``slices/<attr>`` for slice tensors.
    """
    tensors = {}
    for key, value in data:
        if isinstance(value, torch.Tensor):
            tensors[f"data/{key}"] = value
    for key, value in slices.items():
        if isinstance(value, torch.Tensor):
            tensors[f"slices/{key}"] = value
    return tensors


def _unflatten(tensors: dict[str, torch.Tensor]) -> tuple[Data, dict]:
    """Inverse of _flatten."""
    data_dict = {}
    slices_dict = {}
    for key, value in tensors.items():
        namespace, attr = key.split("/", 1)
        if namespace == "data":
            data_dict[attr] = value
        elif namespace == "slices":
            slices_dict[attr] = value
    return Data(**data_dict), slices_dict


def save_pyg_dataset(
    path: str | Path,
    data: Data,
    slices: dict,
    *,
    product_keys: list[str] | None = None,
) -> None:
    """Save a collated PyG dataset to safetensors.

    String metadata (e.g. ``product_keys``) is stored in a sidecar
    ``.meta.json`` because safetensors only stores tensors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tensors = _flatten(data, slices)
    save_file(tensors, str(path))

    if product_keys is not None:
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps({"product_keys": product_keys}))


def load_pyg_dataset(
    path: str | Path,
) -> tuple[Data, dict, list[str] | None, safe_open | None]:
    """Load a collated PyG dataset from safetensors with memory mapping.

    Tensors are memory-mapped: the OS pages in data on demand instead of
    loading the entire file into RAM.  The returned ``handle`` keeps the
    mmap alive — the caller **must** hold a reference to it for as long as
    the tensors are used.

    Returns ``(data, slices, product_keys_or_None, handle)``.
    """
    path = Path(path)

    handle = safe_open(str(path), framework="pt", device="cpu")
    tensors = {key: handle.get_tensor(key) for key in handle}
    data, slices = _unflatten(tensors)

    meta_path = path.with_suffix(".meta.json")
    product_keys = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        product_keys = meta.get("product_keys")

    return data, slices, product_keys, handle
