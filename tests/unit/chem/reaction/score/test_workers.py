"""Worker checkpoints preserve expected failures and expose programming errors."""

from types import SimpleNamespace

import pytest

from scripts import aimnet_workers as workers
from synplan.chem.reaction.score.config import AIMNetConfig, MolecularCalculationError


def adapter(tmp_path):
    service = workers.CPUMolecularWorkers.__new__(workers.CPUMolecularWorkers)
    service.config = AIMNetConfig()
    service.output = tmp_path
    service._identity = {"test": "calculator"}
    service.futures = {}
    return service


def test_successful_worker_result_is_reused_without_submission(tmp_path, monkeypatch):
    service = adapter(tmp_path)
    result = {"energy_eV": -10.0}
    monkeypatch.setattr(
        workers, "_backend", SimpleNamespace(calculate=lambda _: result)
    )
    workers._calculate("O", service._path("O"))
    assert service.calculate("O") == result
    assert service.futures == {}


def test_expected_worker_failure_preserves_conformer_log(tmp_path, monkeypatch):
    service = adapter(tmp_path)
    failure = MolecularCalculationError("optimization", "unconverged", "200 steps")
    failure.conformer_failures = [{"stage": "optimization", "reason": "unconverged"}]

    def calculate(_):
        raise failure

    monkeypatch.setattr(workers, "_backend", SimpleNamespace(calculate=calculate))
    workers._calculate("O", service._path("O"))
    with pytest.raises(MolecularCalculationError) as error:
        service.calculate("O")
    assert error.value.to_dict() == failure.to_dict()
    assert error.value.conformer_failures == failure.conformer_failures


def test_unexpected_worker_exception_remains_visible(tmp_path, monkeypatch):
    def calculate(_):
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(workers, "_backend", SimpleNamespace(calculate=calculate))
    path = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="unexpected bug"):
        workers._calculate("O", path)
    assert not path.exists()
