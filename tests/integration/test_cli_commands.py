from click.testing import CliRunner

import synplan.interfaces.cli as cli


def test_synplan_help():
    runner = CliRunner()
    result = runner.invoke(cli.synplan, ["--help"])
    assert result.exit_code == 0
    assert "SynPlanner command line interface." in result.output


def test_reaction_standardizing_cli_shows_progress_by_default(monkeypatch):
    observed = {}

    def fake_standardize_reactions_from_file(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(
        cli, "standardize_reactions_from_file", fake_standardize_reactions_from_file
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("config.yaml", "w", encoding="utf-8") as config:
            config.write("deduplicate: true\n")
        with open("input.smi", "w", encoding="utf-8") as input_file:
            input_file.write("")

        result = runner.invoke(
            cli.synplan,
            [
                "reaction_standardizing",
                "--config",
                "config.yaml",
                "--input",
                "input.smi",
                "--output",
                "output.smi",
            ],
        )

    assert result.exit_code == 0
    assert observed["silent"] is False


def test_reaction_standardizing_cli_can_suppress_progress(monkeypatch):
    observed = {}

    def fake_standardize_reactions_from_file(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(
        cli, "standardize_reactions_from_file", fake_standardize_reactions_from_file
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("config.yaml", "w", encoding="utf-8") as config:
            config.write("deduplicate: true\n")
        with open("input.smi", "w", encoding="utf-8") as input_file:
            input_file.write("")

        result = runner.invoke(
            cli.synplan,
            [
                "reaction_standardizing",
                "--config",
                "config.yaml",
                "--input",
                "input.smi",
                "--output",
                "output.smi",
                "--silent",
            ],
        )

    assert result.exit_code == 0
    assert observed["silent"] is True


def test_ranking_policy_training_cli_accepts_litlogger(monkeypatch):
    observed = {}

    def fake_create_policy_dataset(**kwargs):
        observed["dataset_kwargs"] = kwargs
        return object()

    def fake_run_policy_training(datamodule, *, config, results_path):
        observed["datamodule"] = datamodule
        observed["config"] = config
        observed["results_path"] = results_path

    monkeypatch.setattr(cli, "create_policy_dataset", fake_create_policy_dataset)
    monkeypatch.setattr(cli, "run_policy_training", fake_run_policy_training)

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("config.yaml", "w", encoding="utf-8") as config:
            config.write(
                "vector_dim: 16\n"
                "num_conv_layers: 1\n"
                "learning_rate: 0.001\n"
                "dropout: 0.1\n"
                "num_epoch: 1\n"
                "batch_size: 2\n"
            )
        with open("policy.tsv", "w", encoding="utf-8") as policy_data:
            policy_data.write("product_smiles\trule_id\nCC\t0\n")

        result = runner.invoke(
            cli.synplan,
            [
                "ranking_policy_training",
                "--config",
                "config.yaml",
                "--policy_data",
                "policy.tsv",
                "--results_dir",
                "out",
                "--logger",
                "litlogger",
            ],
        )

    assert result.exit_code == 0
    assert observed["config"].logger == {"type": "litlogger"}


def test_mhn_ranking_policy_training_cli_uses_policy_data_only(monkeypatch):
    observed = {}

    monkeypatch.setattr(cli, "create_policy_dataset", lambda **_kwargs: object())

    def fake_run_policy_training(datamodule, *, config, results_path):
        observed["datamodule"] = datamodule
        observed["config"] = config
        observed["results_path"] = results_path

    monkeypatch.setattr(cli, "run_policy_training", fake_run_policy_training)

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("config.yaml", "w", encoding="utf-8") as config:
            config.write("architecture: mhn_ranking\n")
        with open("policy.tsv", "w", encoding="utf-8") as policy_data:
            policy_data.write("product_smiles\trule_id\nCC\t0\n")

        result = runner.invoke(
            cli.synplan,
            [
                "ranking_policy_training",
                "--config",
                "config.yaml",
                "--policy_data",
                "policy.tsv",
            ],
        )

    assert result.exit_code == 0
    assert observed["config"].architecture == "mhn_ranking"


def test_mhn_network_tuning_cli_passes_checkpoint_and_new_policy_data(monkeypatch):
    observed = {}

    def fake_run_mhn_network_tuning(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(cli, "run_mhn_network_tuning", fake_run_mhn_network_tuning)

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("policy_network.ckpt", "w", encoding="utf-8") as checkpoint:
            checkpoint.write("")
        with open("config.yaml", "w", encoding="utf-8") as config:
            config.write(
                "architecture: mhn_ranking\n"
                "num_epoch: 3\n"
                "learning_rate: 0.0001\n"
                "batch_size: 4\n"
                "logger:\n"
                "  type: csv\n"
            )
        with open("new_reaction_rules_policy_data.tsv", "w", encoding="utf-8") as data:
            data.write("product_smiles\trule_id\nCC\t0\n")

        result = runner.invoke(
            cli.synplan,
            [
                "mhn_network_tuning",
                "--config",
                "config.yaml",
                "--policy_network",
                "policy_network.ckpt",
                "--new_policy_data",
                "new_reaction_rules_policy_data.tsv",
                "--results_dir",
                "mhn_tuned",
                "--workers",
                "2",
            ],
        )

    assert result.exit_code == 0
    assert observed["config"].architecture == "mhn_ranking"
    assert observed["config"].num_epoch == 3
    assert observed["config"].learning_rate == 0.0001
    assert observed["config"].batch_size == 4
    assert observed["config"].logger == {"type": "csv"}
    assert observed == {
        "policy_network_path": "policy_network.ckpt",
        "new_policy_data_path": "new_reaction_rules_policy_data.tsv",
        "results_path": "mhn_tuned",
        "config": observed["config"],
        "num_workers": 2,
        "cache": True,
    }


def test_building_block_cli_preserves_legacy_two_path_defaults(monkeypatch):
    observed = {}

    def fake_prepare_building_blocks(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(cli, "prepare_building_blocks", fake_prepare_building_blocks)
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("input.smi", "w", encoding="utf-8") as input_file:
            input_file.write("CCO\n")
        result = runner.invoke(
            cli.synplan,
            [
                "building_blocks_standardizing",
                "--input",
                "input.smi",
                "--output",
                "stock.smi",
            ],
        )

    assert result.exit_code == 0, result.output
    assert observed["input_file"] == "input.smi"
    assert observed["output_file"] == "stock.smi"
    assert not observed["config"].deprotect
    assert not observed["config"].write_inchikey_stock
    assert not observed["config"].write_audit_files


def test_building_block_cli_loads_yaml_configuration(monkeypatch):
    observed = {}

    def fake_prepare_building_blocks(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(cli, "prepare_building_blocks", fake_prepare_building_blocks)
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("input.smi", "w", encoding="utf-8") as input_file:
            input_file.write("CCO\n")
        with open("config.yaml", "w", encoding="utf-8") as config_file:
            config_file.write(
                "deprotect: true\n"
                "deprotect_policy: aggressive\n"
                "deprotect_output: append\n"
                "write_inchikey_stock: true\n"
                "write_audit_files: true\n"
                "audit_overwrite: replace\n"
                "num_workers: 1\n"
            )
        result = runner.invoke(
            cli.synplan,
            [
                "building_blocks_standardizing",
                "--config",
                "config.yaml",
                "--input",
                "input.smi",
                "--output",
                "stock.smi",
            ],
        )

    assert result.exit_code == 0, result.output
    config = observed["config"]
    assert config.deprotect_policy == "aggressive"
    assert config.deprotect_output == "append"
    assert config.write_inchikey_stock
    assert config.write_audit_files
    assert config.audit_overwrite == "replace"


def test_building_block_cli_has_only_input_output_and_config():
    command = cli.synplan.commands["building_blocks_standardizing"]
    assert {parameter.name for parameter in command.params} == {
        "input_file",
        "output_file",
        "config_path",
    }
