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
