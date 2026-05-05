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
