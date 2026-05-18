import gzip

from synplan.chem.utils import _standardize_smiles_batch
from synplan.utils.loading import load_building_blocks, load_policy_function


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

    expected = frozenset(_standardize_smiles_batch(["OCC", "CCN"]))
    bbs = load_building_blocks(path, standardize=True, silent=True, num_workers=1)
    assert bbs == expected


def test_load_policy_function_weights_path_applies_overrides(monkeypatch):
    captured = {}

    class DummyPolicyNetworkFunction:
        def __init__(self, policy_config):
            captured["policy_config"] = policy_config

    monkeypatch.setattr(
        "synplan.mcts.expansion.PolicyNetworkFunction",
        DummyPolicyNetworkFunction,
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
