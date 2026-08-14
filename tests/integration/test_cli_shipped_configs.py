"""Every shipped config must reach the CLI subcommand it is written for.

tests/unit/utils/test_shipped_configs.py only proves the YAML parses. A config
can parse and still be unusable: ``planning`` reads its own top-level sections,
so a config whose sections do not match is a KeyError the first time a user
runs the documented command.

The heavy worker each subcommand calls is stubbed out; what is under test is
CLI wiring plus config parsing, not the pipeline behind it.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

import synplan.interfaces.cli as cli

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

# Worker functions the subcommands call once parsing succeeded.
WORKERS = (
    "standardize_reactions_from_file",
    "filter_reactions_from_file",
    "map_reactions_from_file",
    "extract_rules_from_reactions",
    "create_policy_dataset",
    "run_policy_training",
    "run_mhn_network_tuning",
    "run_updating",
    "run_search",
    "load_policy_function",
    "load_reaction_rules",
    "load_building_block",
    "classify_file",
    "synthonise_file",
    "fragment_file",
    "enumerate_file",
    "scaffolds_file",
)

# Input files touched in the isolated filesystem before each invocation.
INPUTS = ("in.smi", "policy.tsv", "rules.tsv", "bb.smi", "net.ckpt", "value.ckpt")

# (subcommand, config file, arguments other than --config)
CASES = [
    (
        "reaction_standardizing",
        "reactions_standardization.yaml",
        ["--input", "in.smi", "--output", "out.smi"],
    ),
    (
        "reaction_filtering",
        "reactions_filtration.yaml",
        ["--input", "in.smi", "--output", "out.smi"],
    ),
    (
        "rule_extracting",
        "rules_extraction.yaml",
        ["--input", "in.smi", "--output", "out.tsv"],
    ),
    (
        "rule_extracting",
        "extraction_functional_groups.yaml",
        ["--input", "in.smi", "--output", "out.tsv"],
    ),
    (
        "ranking_policy_training",
        "policy_training.yaml",
        ["--policy_data", "policy.tsv"],
    ),
    (
        "ranking_policy_training",
        "mhn_ranking_policy_training.yaml",
        ["--policy_data", "policy.tsv"],
    ),
    (
        "filtering_policy_training",
        "policy_training.yaml",
        ["--molecule_data", "in.smi", "--reaction_rules", "rules.tsv"],
    ),
    (
        "mhn_network_tuning",
        "mhn_ranking_policy_training.yaml",
        ["--policy_network", "net.ckpt", "--new_policy_data", "policy.tsv"],
    ),
    (
        "value_network_tuning",
        "tuning.yaml",
        [
            "--targets",
            "in.smi",
            "--reaction_rules",
            "rules.tsv",
            "--building_blocks",
            "bb.smi",
            "--policy_network",
            "net.ckpt",
        ],
    ),
    (
        "planning",
        "planning_standard.yaml",
        [
            "--targets",
            "in.smi",
            "--reaction_rules",
            "rules.tsv",
            "--building_blocks",
            "bb.smi",
            "--policy_network",
            "net.ckpt",
        ],
    ),
    (
        "planning",
        "planning_value.yaml",
        [
            "--targets",
            "in.smi",
            "--reaction_rules",
            "rules.tsv",
            "--building_blocks",
            "bb.smi",
            "--policy_network",
            "net.ckpt",
            "--value_network",
            "value.ckpt",
        ],
    ),
    pytest.param(
        "planning",
        "planning_combined_policies.yaml",
        [
            "--targets",
            "in.smi",
            "--reaction_rules",
            "rules.tsv",
            "--building_blocks",
            "bb.smi",
            "--policy_network",
            "net.ckpt",
        ],
    ),
    (
        "bb_classifying",
        "synthonisation.yaml",
        ["--input", "in.smi", "--output", "out.tsv"],
    ),
    (
        "bb_synthonizing",
        "synthonisation.yaml",
        ["--input", "in.smi", "--output", "out.smi"],
    ),
    (
        "synthon_fragment",
        "synthonisation.yaml",
        ["--input", "in.smi", "--output", "out.tsv"],
    ),
    (
        "synthon_enumerate",
        "synthonisation.yaml",
        ["--input", "in.smi", "--output", "out.smi", "--stock", "bb.smi"],
    ),
    (
        "bb_scaffolds",
        "synthonisation.yaml",
        ["--input", "in.smi", "--output", "out.tsv"],
    ),
]

# Configs consumed by the Python API only, never passed to a subcommand.
PYTHON_API_ONLY = {
    "building_blocks_stock.yaml",
    "combined_ranking_filtering_policy.yaml",
}


def test_every_shipped_config_has_a_cli_case():
    # getattr unwraps pytest.param; a plain tuple indexes directly
    covered = {getattr(case, "values", case)[1] for case in CASES}
    assert {p.name for p in CONFIGS.glob("*.yaml")} == covered | PYTHON_API_ONLY


@pytest.fixture
def stubbed_workers(monkeypatch):
    """Replace every worker with a recorder, so only CLI wiring runs."""
    calls: list[str] = []
    for name in WORKERS:
        monkeypatch.setattr(cli, name, lambda *a, _n=name, **kw: calls.append(_n))

    def record_run_updating(**kwargs):
        assert "building_block_stock" in kwargs
        assert "building_blocks_path" not in kwargs
        assert "building_block_stock_config" not in kwargs
        calls.append("run_updating")

    monkeypatch.setattr(cli, "run_updating", record_run_updating)
    return calls


@pytest.mark.parametrize(("command", "config", "args"), CASES)
def test_shipped_config_reaches_its_command(command, config, args, stubbed_workers):
    runner = CliRunner()
    with runner.isolated_filesystem():
        for name in INPUTS:
            Path(name).touch()
        result = runner.invoke(
            cli.synplan, [command, "--config", str(CONFIGS / config), *args]
        )

    assert result.exit_code == 0, f"{command} {config}: {result.exception!r}"
    assert stubbed_workers, f"{command} {config} exited without doing any work"
