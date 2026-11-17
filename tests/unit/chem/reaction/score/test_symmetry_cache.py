"""Regression coverage for configuration routing and both checkpoint layers."""

import json
from concurrent.futures import Future

import pytest

from scripts import aimnet_workers as workers
from scripts import assess_thermochemistry as batch
from synplan.chem.reaction.score import AIMNetConfig, AIMNetReactionAssessor, aimnet
from synplan.chem.reaction.score.chemistry import normalize_config

from .test_aimnet import Calculator, route
from .test_batch import input_file


class SymmetryCalculator(Calculator):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def calculate(self, text):
        number = self.config.symmetry_numbers.get(text, 1)
        return {
            **super().calculate(text),
            "symmetry_number": number,
            "gibbs_eV": float(number),
        }


class ImmediatePool:
    """Execute the actual worker functions without processes or optional models."""

    def __init__(self, *, initializer, initargs, **kwargs):
        initializer(*initargs)

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass


def setup_backends(monkeypatch):
    monkeypatch.setattr(aimnet, "AIMNetMolecularCalculator", SymmetryCalculator)
    monkeypatch.setattr(workers, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(
        workers,
        "_initialize",
        lambda config, threads: monkeypatch.setattr(
            workers, "_backend", SymmetryCalculator(AIMNetConfig.model_validate(config))
        ),
    )


def annotation(output):
    return json.loads((output / "reactions/00000.json").read_text())["annotation"]


def test_serial_parallel_override_equivalence_and_reuse(tmp_path, monkeypatch):
    setup_backends(monkeypatch)
    config = AIMNetConfig(symmetry_numbers={"[OH2]": 2})
    source = input_file(tmp_path, ["CC(=O)O.CCO>>CC(=O)OCC"])
    values = []
    for count in (1, 2):
        output = tmp_path / f"run-{count}"
        cache = tmp_path / f"cache-{count}.json"
        batch.run_assessment(source, output, config, cache, workers=count)
        row = annotation(output)
        assert row["calculation"]["config"]["symmetry_numbers"] == {"O": 2}
        assert row["molecules"]["O"]["symmetry_number"] == 2
        values.append(row["variants"])
    assert values[0] == values[1]
    reused = batch.run_assessment(
        source, tmp_path / "reuse", config, tmp_path / "cache-2.json"
    )
    assert reused["statistics"]["molecular_calculations"] == 0
    assert annotation(tmp_path / "reuse")["variants"] == values[0]
    assert config.symmetry_numbers == {"[OH2]": 2}


@pytest.mark.parametrize("recorded", [None, 1, True])
def test_invalid_symmetry_cache_recalculates_only_affected_species(tmp_path, recorded):
    config = normalize_config(AIMNetConfig(symmetry_numbers={"[OH2]": 2}))
    cache = tmp_path / "cache.json"
    rows = [route("O>>O"), route("CCO>>CCO")]
    AIMNetReactionAssessor(config, cache, calculator=SymmetryCalculator(config)).assess(
        rows
    )
    saved = json.loads(cache.read_text())
    water = next(iter(saved["calculations"].values()))["molecules"]["O"]
    water.pop("symmetry_number")
    if recorded is not None:
        water["symmetry_number"] = recorded
    water["gibbs_eV"] = -999
    cache.write_text(json.dumps(saved))
    calculator = SymmetryCalculator(config)
    service = AIMNetReactionAssessor(config, cache, calculator=calculator)
    service.assess(rows)
    assert calculator.calls == ["O"]
    assert service.last_stats["cache_invalidations"] == 1
    assert service.last_stats["cache_hits"] == 1


def test_inconsistent_new_calculator_result_is_not_cached(tmp_path):
    config = AIMNetConfig(symmetry_numbers={"O": 2})
    cache = tmp_path / "cache.json"
    with pytest.raises(ValueError, match="symmetry"):
        AIMNetReactionAssessor(
            config, cache, calculator=SymmetryCalculator(AIMNetConfig())
        ).assess([route("O>>O")])
    assert not cache.exists()


def test_worker_rejects_inconsistent_job_results(tmp_path, monkeypatch):
    setup_backends(monkeypatch)
    service = workers.CPUMolecularWorkers(
        AIMNetConfig(symmetry_numbers={"[OH2]": 2}), tmp_path, 2, 1
    )
    try:
        path = service._path("O")
        path.write_text(json.dumps({"result": {"symmetry_number": 1}}))
        assert service.calculate("O")["symmetry_number"] == 2
    finally:
        service.close()


def test_legacy_and_changed_orchestration_checkpoints_are_refused(
    tmp_path, monkeypatch
):
    setup_backends(monkeypatch)
    config = AIMNetConfig(symmetry_numbers={"O": 2})
    source = input_file(tmp_path, ["O>>O"])
    output = tmp_path / "results"
    batch.run_assessment(source, output, config, tmp_path / "cache.json")
    job = output / "checkpoint.json"
    checkpoint = json.loads(job.read_text())
    assert "scripts/aimnet_workers.py" in checkpoint["code"]
    assert "scripts/assess_thermochemistry.py" in checkpoint["code"]
    before = (output / "routes_aimnet.json").read_bytes()
    digest = batch.file_digest
    monkeypatch.setattr(
        batch,
        "file_digest",
        lambda p: "changed" if str(p).endswith("aimnet_workers.py") else digest(p),
    )
    with pytest.raises(batch.IncompatibleCheckpointError):
        batch.run_assessment(source, output, config, tmp_path / "fresh.json")
    assert (output / "routes_aimnet.json").read_bytes() == before
    checkpoint.pop("version")
    job.write_text(json.dumps(checkpoint))
    with pytest.raises(batch.IncompatibleCheckpointError, match="Legacy"):
        batch.run_assessment(source, output, config, tmp_path / "fresh.json")
    assert (output / "routes_aimnet.json").read_bytes() == before
