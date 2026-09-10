import gzip
from pathlib import Path
from unittest.mock import Mock

import pytest

from synplan.chem.building_blocks import load_building_blocks as catalogue_loader
from synplan.chem.utils import standardize_smiles_batch
from synplan.utils.loading import (
    download_preset,
    load_building_blocks,
    load_policy_function,
)


@pytest.mark.parametrize("onnx_available", [True, False])
def test_download_preset_prefers_onnx_when_available(
    tmp_path, monkeypatch, onnx_available
):
    preset = tmp_path / "preset.yaml"
    preset.write_text(
        "files:\n"
        "  ranking_policy: policy/v1/ranking_policy.ckpt\n"
        "  reaction_rules: policy/v1/reaction_rules.tsv\n"
        "  other_policy: policy/v2/ranking_policy.onnx\n",
        encoding="utf-8",
    )

    def download(*, repo_id, filename, subfolder, local_dir):
        assert repo_id == "test/models"
        if subfolder == "presets":
            assert filename == "test-preset.yaml"
            return preset
        return Path(local_dir) / subfolder / filename

    downloads = Mock(side_effect=download)
    exists = Mock(return_value=onnx_available)
    monkeypatch.setattr("synplan.utils.loading.hf_hub_download", downloads)
    monkeypatch.setattr("synplan.utils.loading.file_exists", exists)
    paths = download_preset("test-preset", tmp_path, repo_id="test/models")
    extension = "onnx" if onnx_available else "ckpt"
    assert paths == {
        "ranking_policy": tmp_path / f"policy/v1/ranking_policy.{extension}",
        "reaction_rules": tmp_path / "policy/v1/reaction_rules.tsv",
        "other_policy": tmp_path / "policy/v2/ranking_policy.onnx",
    }
    exists.assert_called_once_with(
        repo_id="test/models", filename="policy/v1/ranking_policy.onnx"
    )
    assert downloads.call_count == 4


@pytest.mark.parametrize("extension", [".csv", ".csv.gz", ".tsv", ".tsv.gz"])
@pytest.mark.parametrize("standardize", [False, True])
def test_load_building_blocks_table_header(tmp_path, extension, standardize):
    assert load_building_blocks is catalogue_loader
    path = tmp_path / ("bbs" + extension)
    data = "SMILES,ID\nCCO,1\n,2\nCCO,3\nCCN,4\n"
    if extension.startswith(".tsv"):
        data = data.replace(",", "\t")
    path.write_bytes(
        gzip.compress(data.encode()) if extension.endswith(".gz") else data.encode()
    )

    bbs = load_building_blocks(path, standardize=standardize, num_workers=1)
    assert isinstance(bbs, frozenset)
    assert bbs == frozenset({"CCO", "CCN"})


def test_load_building_blocks_csv_header_case_insensitive_column(tmp_path):
    path = tmp_path / "bbs.csv"
    path.write_text("smiles\nCCO\nCCN\n", encoding="utf-8")

    # Default smiles_column="SMILES" should match "smiles" in a case-insensitive way.
    bbs = load_building_blocks(path, standardize=False, silent=True)
    assert bbs == frozenset({"CCO", "CCN"})


@pytest.mark.parametrize("extension", [".csv", ".tsv"])
def test_load_building_blocks_table_no_header(tmp_path, extension):
    path = tmp_path / ("bbs" + extension)
    delimiter = "\t" if extension == ".tsv" else ","
    path.write_text(f"CCO{delimiter}1\nCCN{delimiter}2\n\n", encoding="utf-8")

    bbs = load_building_blocks(path, standardize=False, silent=True, header=False)
    assert bbs == frozenset({"CCO", "CCN"})


def test_load_building_blocks_csv_gz(tmp_path):
    path = tmp_path / "bbs.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write("SMILES\nCCO\nCCN\n")

    bbs = load_building_blocks(path, standardize=False, silent=True)
    assert bbs == frozenset({"CCO", "CCN"})


@pytest.mark.parametrize("extension", [".csv", ".tsv"])
@pytest.mark.parametrize("workers", [1, 2])
def test_load_building_blocks_table_standardization(tmp_path, extension, workers):
    path = tmp_path / ("bbs" + extension)
    path.write_text("SMILES\nOCC\nCCN\n", encoding="utf-8")

    expected = frozenset(standardize_smiles_batch(["OCC", "CCN"]))
    bbs = load_building_blocks(path, standardize=True, silent=True, num_workers=workers)
    assert bbs == expected


def test_standardized_stock_rejects_unsupported_stereo(tmp_path):
    unsupported = "Cc1ccccc1-c1ccccc1C |wU:1.6|"
    failures = []
    assert standardize_smiles_batch(["CCO", unsupported], failures=failures) == ["CCO"]
    assert failures[0]["record"] == 2
    assert failures[0]["smiles"] == unsupported
    assert "unsupported" in failures[0]["error"].lower()
    path = tmp_path / "stock.smi"
    path.write_text(f"CCO\n{unsupported}\n")
    assert load_building_blocks(path, standardize=True, num_workers=1) == {"CCO"}


def test_load_policy_function_weights_path_applies_overrides(monkeypatch):
    captured = {}

    def dummy_build_policy_from_config(policy_config):
        captured["policy_config"] = policy_config

    monkeypatch.setattr(
        "synplan.utils.loading.build_policy_from_config",
        dummy_build_policy_from_config,
    )

    load_policy_function(
        weights_path="policy.ckpt",
        top_rules=500,
        rule_prob_threshold=0.0,
    )

    policy_config = captured["policy_config"]
    assert policy_config.weights_path == "policy.ckpt"
    assert policy_config.top_rules == 500
    assert policy_config.rule_prob_threshold == 0.0
