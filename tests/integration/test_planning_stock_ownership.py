"""Planning CLI and search stock-ownership regressions."""

from __future__ import annotations

import inspect

from click.testing import CliRunner

import synplan.interfaces.cli as cli_module
import synplan.mcts.search as search_module
from synplan.chem.building_blocks import BuildingBlockStock
from synplan.utils.config import PolicyNetworkConfig


def test_planning_cli_loads_and_shares_stock_once(tmp_path, monkeypatch):
    stock = BuildingBlockStock(frozenset({"CC"}))
    stock_loads = []
    observed = {}

    def fake_load_building_block(path, *, config):
        stock_loads.append((path, config))
        return stock

    def fake_run_search(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(cli_module, "load_building_block", fake_load_building_block)
    monkeypatch.setattr(cli_module, "load_policy_function", lambda **_kwargs: object())
    monkeypatch.setattr(cli_module, "load_reaction_rules", lambda _path: [])
    monkeypatch.setattr(cli_module, "run_search", fake_run_search)

    config_path = tmp_path / "planning.yaml"
    config_path.write_text(
        "building_blocks:\n"
        "  identity_format: smiles\n"
        "tree:\n"
        "  min_mol_size: 6\n"
        "  max_depth: 6\n"
        "node_expansion: {}\n"
        "node_evaluation:\n"
        "  evaluation_type: rollout\n",
        encoding="utf-8",
    )
    inputs = {
        name: tmp_path / name
        for name in ("targets.smi", "rules.tsv", "stock.smi", "policy.ckpt")
    }
    for path in inputs.values():
        path.touch()

    result = CliRunner().invoke(
        cli_module.synplan,
        [
            "planning",
            "--config",
            str(config_path),
            "--targets",
            str(inputs["targets.smi"]),
            "--reaction_rules",
            str(inputs["rules.tsv"]),
            "--building_blocks",
            str(inputs["stock.smi"]),
            "--policy_network",
            str(inputs["policy.ckpt"]),
            "--results_dir",
            str(tmp_path / "results"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(stock_loads) == 1
    assert stock_loads[0][0] == str(inputs["stock.smi"])
    assert stock_loads[0][1].identity_format == "smiles"
    assert observed["building_block_stock"] is stock
    assert observed["evaluation_config"].building_blocks is stock
    assert "building_blocks_path" not in observed
    assert "building_block_stock_config" not in observed


def test_run_search_accepts_and_forwards_loaded_stock(tmp_path, monkeypatch):
    parameters = inspect.signature(search_module.run_search).parameters
    assert "building_block_stock" in parameters
    assert "building_blocks_path" not in parameters
    assert "building_block_stock_config" not in parameters

    stock = BuildingBlockStock(frozenset({"CC"}))
    evaluator = object()
    observed = {}

    class FakeTree:
        winning_nodes = ()

        def __init__(self, **kwargs):
            observed.update(kwargs)

        def run(self):
            pass

        def newickify(self, visits_threshold=0):
            return "", {}

        def to_stats_dict(self):
            return {}

    monkeypatch.setattr(search_module, "Tree", FakeTree)
    monkeypatch.setattr(
        search_module, "load_policy_function", lambda **_kwargs: object()
    )
    monkeypatch.setattr(search_module, "load_reaction_rules", lambda _path: [])
    monkeypatch.setattr(
        search_module, "load_evaluation_function", lambda _config: evaluator
    )

    targets = tmp_path / "targets.smi"
    targets.write_text("CCO\n", encoding="utf-8")
    search_module.run_search(
        targets_path=str(targets),
        search_config={},
        policy_config=PolicyNetworkConfig(),
        evaluation_config=object(),
        reaction_rules_path="rules.tsv",
        building_block_stock=stock,
        results_root=str(tmp_path / "results"),
    )

    assert observed["building_blocks"] is stock
    assert observed["evaluation_function"] is evaluator
