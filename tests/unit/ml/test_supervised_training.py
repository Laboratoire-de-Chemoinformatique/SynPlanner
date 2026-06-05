"""Tests for supervised policy training orchestration."""

from types import SimpleNamespace

import pytest

import synplan.ml.training.supervised as supervised
from synplan.utils.config import PolicyNetworkConfig


def test_run_policy_training_creates_nested_results_dir(monkeypatch, tmp_path):
    """Training should accept a run directory whose parents do not exist yet."""

    class StopBeforeTraining(RuntimeError):
        pass

    def fake_policy_network(**_kwargs):
        raise StopBeforeTraining

    datamodule = SimpleNamespace(
        train_dataset=SimpleNamespace(dataset=SimpleNamespace(num_classes=3))
    )
    results_path = tmp_path / "results" / "run-1"

    monkeypatch.setattr(supervised, "PolicyNetwork", fake_policy_network)

    with pytest.raises(StopBeforeTraining):
        supervised.run_policy_training(
            datamodule,
            config=PolicyNetworkConfig(num_epoch=1),
            results_path=str(results_path),
        )

    assert results_path.is_dir()


def test_create_logger_supports_litlogger(monkeypatch, tmp_path):
    """LitLogger should use results_path as the default local root directory."""

    class FakeLitLogger:
        def __init__(self, root_dir, **kwargs):
            self.root_dir = root_dir
            self.name = kwargs["name"]

    import pytorch_lightning.loggers as lightning_loggers

    monkeypatch.setattr(lightning_loggers, "LitLogger", FakeLitLogger, raising=False)

    config = PolicyNetworkConfig(logger={"type": "LitLogger", "name": "ranking-policy"})

    logger = supervised._create_logger(
        {**config.logger, "save_logs": False},
        tmp_path,
    )

    assert logger.name == "ranking-policy"
    assert logger.root_dir == str(tmp_path)


def test_run_mhn_network_tuning_rebinds_new_policy_data(monkeypatch, tmp_path):
    observed = {}

    class FakeNetwork:
        architecture = "mhn_ranking"

        def __init__(self):
            self.hparams = {
                "architecture": "mhn_ranking",
                "policy_type": "ranking",
                "vector_dim": 16,
                "batch_size": 2,
                "dropout": 0.1,
                "num_conv_layers": 1,
                "learning_rate": 0.001,
                "num_epoch": 1,
            }

        def bind_training_rules_from_policy_data(
            self, policy_data_path, *, training_labels
        ):
            observed["bound_policy_data_path"] = policy_data_path
            observed["bound_training_labels"] = training_labels

    fake_network = FakeNetwork()

    monkeypatch.setattr(
        supervised.MHNRankingPolicyNetwork,
        "load_from_checkpoint",
        lambda *args, **kwargs: fake_network,
    )

    labels = object()
    datamodule = SimpleNamespace(
        train_dataset=SimpleNamespace(
            dataset=SimpleNamespace(_data=SimpleNamespace(y_rules=labels))
        )
    )

    def fake_create_policy_dataset(**kwargs):
        observed["dataset_kwargs"] = kwargs
        return datamodule

    def fake_fit_policy_network(
        network, datamodule_arg, config, results_path, **kwargs
    ):
        observed["fit_network"] = network
        observed["fit_datamodule"] = datamodule_arg
        observed["fit_config"] = config
        observed["fit_results_path"] = results_path
        observed["fit_kwargs"] = kwargs

    monkeypatch.setattr(supervised, "create_policy_dataset", fake_create_policy_dataset)
    monkeypatch.setattr(supervised, "_fit_policy_network", fake_fit_policy_network)

    supervised.run_mhn_network_tuning(
        policy_network_path="policy_network.ckpt",
        new_policy_data_path="new_reaction_rules_policy_data.tsv",
        results_path=str(tmp_path / "mhn_tuned"),
        config=PolicyNetworkConfig(
            architecture="mhn_ranking",
            num_epoch=3,
            batch_size=4,
            learning_rate=0.0001,
        ),
        num_workers=2,
        cache=False,
        silent=True,
    )

    assert observed["dataset_kwargs"] == {
        "policy_data_path": "new_reaction_rules_policy_data.tsv",
        "results_dir": str(tmp_path / "mhn_tuned"),
        "dataset_type": "ranking",
        "batch_size": 4,
        "num_workers": 2,
        "cache": False,
    }
    assert observed["bound_policy_data_path"] == "new_reaction_rules_policy_data.tsv"
    assert observed["bound_training_labels"] is labels
    assert observed["fit_network"] is fake_network
    assert observed["fit_datamodule"] is datamodule
    assert observed["fit_config"].architecture == "mhn_ranking"
    assert observed["fit_config"].num_epoch == 3
    assert observed["fit_config"].learning_rate == 0.0001
    assert observed["fit_results_path"] == str(tmp_path / "mhn_tuned")
    assert observed["fit_kwargs"]["silent"] is True
    assert fake_network.batch_size == 4
    assert fake_network.lr == 0.0001
