"""Stock caches follow source content and the active chemistry contract."""

from synplan.chem import stock as stock_module
from synplan.chem.utils import standardize_smiles_batch
from synplan.utils.loading import load_building_blocks
from synplan.utils.provenance import atomic_json, read_json


def test_cache_records_failures_and_equivalent_spellings(tmp_path):
    path = tmp_path / "stock.csv"
    path.write_text("SMILES\nCCO\nOCC\nnot-smiles\n")
    stock, metadata = stock_module.load_stock_cache(path)
    assert stock == set(standardize_smiles_batch(["CCO"]))
    assert metadata["input_rows"] == 3
    assert len(metadata["failures"]) == 1
    assert metadata["failures"][0]["record"] == 3
    assert stock_module.load_stock_cache(path) == (stock, metadata)


def test_source_edit_and_normalization_change_invalidate_cache(tmp_path, monkeypatch):
    path = tmp_path / "stock.csv"
    path.write_text("SMILES\nCCO\n")
    _, before = stock_module.load_stock_cache(path)
    path.write_text("SMILES\nCCO\nCCN\n")
    stock, after = stock_module.load_stock_cache(path)
    assert len(stock) == 2
    assert before["cache_path"] != after["cache_path"]
    identity = {**stock_module.normalization_identity(), "name": "next-contract"}
    monkeypatch.setattr(stock_module, "normalization_identity", lambda: identity)
    _, updated = stock_module.load_stock_cache(path)
    assert after["cache_path"] != updated["cache_path"]


def test_tampered_cache_is_rebuilt(tmp_path):
    path = tmp_path / "stock.csv"
    path.write_text("SMILES\nCCO\n")
    expected, metadata = stock_module.load_stock_cache(path)
    from pathlib import Path

    cached_path = Path(metadata["cache_path"])
    record = read_json(cached_path)
    record["smiles"] = ["CCN"]
    atomic_json(cached_path, record)
    assert stock_module.load_stock_cache(path)[0] == expected


def test_cache_keeps_safe_canonicalization_fallbacks(tmp_path):
    path = tmp_path / "stock.csv"
    path.write_text("SMILES\nc1cccc1\nCCO\n")
    expected = load_building_blocks(path, standardize=True, num_workers=1)
    actual, metadata = stock_module.load_stock_cache(path)
    assert actual == expected
    assert metadata["failures"][0]["retained"] is True
