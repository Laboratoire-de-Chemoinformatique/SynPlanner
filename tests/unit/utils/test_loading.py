import gzip

from synplan.chem.utils import standardize_smiles_batch
from synplan.utils.loading import (
    SASCORE_BENCHMARK_FILES,
    SASCORE_BENCHMARK_SUBFOLDER,
    download_sascore_benchmark,
    load_building_block,
    load_building_blocks,
    load_policy_function,
)


def test_download_sascore_benchmark_uses_published_subset(monkeypatch, tmp_path):
    observed = {}

    def fake_download_selected_files(files_to_get, **kwargs):
        observed["files_to_get"] = files_to_get
        observed.update(kwargs)
        return tmp_path

    monkeypatch.setattr(
        "synplan.utils.loading.download_selected_files", fake_download_selected_files
    )

    paths = download_sascore_benchmark(tmp_path, repo_id="test/repo")

    assert observed["files_to_get"] == [
        (SASCORE_BENCHMARK_SUBFOLDER, name) for name in SASCORE_BENCHMARK_FILES
    ]
    assert observed["extract_zips"] is False
    assert observed["repo_id"] == "test/repo"
    assert paths == [
        tmp_path / SASCORE_BENCHMARK_SUBFOLDER / name
        for name in SASCORE_BENCHMARK_FILES
    ]


def test_load_building_block_returns_typed_canonical_stock(tmp_path):
    path = tmp_path / "stock.smi"
    path.write_text("OCC\n", encoding="utf-8")

    stock = load_building_block(path)

    assert stock.identity_format == "smiles"
    assert stock.keys == frozenset({"CCO"})


def test_load_building_blocks_csv_header(tmp_path):
    path = tmp_path / "bbs.csv"
    path.write_text(
        "SMILES,ID\nCCO,1\n,2\nCCO,3\nCCN,4\n",
        encoding="utf-8",
    )

    bbs = load_building_blocks(path, standardize=False, silent=True)
    assert bbs == frozenset({"CCO", "CCN"})


def test_load_building_blocks_csv_header_case_insensitive_column(tmp_path):
    path = tmp_path / "bbs.csv"
    path.write_text("smiles\nCCO\nCCN\n", encoding="utf-8")

    # Default smiles_column="SMILES" should match "smiles" in a case-insensitive way.
    bbs = load_building_blocks(path, standardize=False, silent=True)
    assert bbs == frozenset({"CCO", "CCN"})


def test_load_building_blocks_csv_no_header(tmp_path):
    path = tmp_path / "bbs.csv"
    path.write_text("CCO,1\nCCN,2\n\n", encoding="utf-8")

    bbs = load_building_blocks(path, standardize=False, silent=True, header=False)
    assert bbs == frozenset({"CCO", "CCN"})


def test_load_building_blocks_csv_gz(tmp_path):
    path = tmp_path / "bbs.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write("SMILES\nCCO\nCCN\n")

    bbs = load_building_blocks(path, standardize=False, silent=True)
    assert bbs == frozenset({"CCO", "CCN"})


def test_load_building_blocks_csv_standardize_true_runs(tmp_path):
    path = tmp_path / "bbs.csv"
    path.write_text("SMILES\nOCC\nCCN\n", encoding="utf-8")

    expected = frozenset(standardize_smiles_batch(["OCC", "CCN"]))
    bbs = load_building_blocks(path, standardize=True, silent=True, num_workers=1)
    assert bbs == expected


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
