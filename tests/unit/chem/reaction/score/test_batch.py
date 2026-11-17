"""Exact-run checkpoint resume and visible aborts without model dependencies."""

import json

import pytest

from scripts import assess_thermochemistry as batch
from synplan.chem.reaction.score import AIMNetConfig, AIMNetReactionAssessor

from .test_aimnet import Calculator, route


def inject(monkeypatch, calculator):
    class Injected(AIMNetReactionAssessor):
        def __init__(self, config, cache_path, **kwargs):
            super().__init__(config, cache_path, calculator=calculator)

    monkeypatch.setattr(batch, "AIMNetReactionAssessor", Injected)


def input_file(tmp_path, texts):
    path = tmp_path / "routes.json"
    path.write_text(
        json.dumps({str(i + 10): route(text).to_json() for i, text in enumerate(texts)})
    )
    return path


def test_resume_complete_run_and_successful_cache_verification(tmp_path, monkeypatch):
    calculator = Calculator()
    inject(monkeypatch, calculator)
    source = input_file(tmp_path, ["CCO>>CCO", "O>>O"])
    output = tmp_path / "results"
    cache = tmp_path / "cache.json"
    result = batch.run_assessment(
        source, output, AIMNetConfig(), cache, verify_cache=True
    )
    assert result["cached_rerun"]["molecular_calculations"] == 0
    assert calculator.calls == ["CCO", "O"]
    assert len(list((output / "reactions").glob("*.json"))) == 2
    resumed = batch.run_assessment(
        source, output, AIMNetConfig(), cache, verify_cache=True
    )
    assert resumed["resumed_reactions"] == 2
    assert calculator.calls == ["CCO", "O"]
    assert list(json.loads((output / "routes_aimnet.json").read_text())) == ["10", "11"]


def test_abort_keeps_results_and_marks_unprocessed_steps(tmp_path, monkeypatch):
    class Broken(Calculator):
        def calculate(self, text):
            if text == "O":
                raise RuntimeError("model failed unexpectedly")
            return super().calculate(text)

    inject(monkeypatch, Broken())
    source = input_file(tmp_path, ["CCO>>CCO", "O>>O", "C>>C"])
    output = tmp_path / "results"
    with pytest.raises(RuntimeError, match="unexpectedly"):
        batch.run_assessment(source, output, AIMNetConfig(), tmp_path / "cache.json")
    data = json.loads((output / "routes_aimnet.json").read_text())
    annotations = [
        row["children"][0]["meta"]["thermochemistry"] for row in data.values()
    ]
    assert annotations[0]["status"] == "ok"
    assert all(a["failure"]["reason"] == "run_aborted" for a in annotations[1:])
    assert "Traceback" in (output / "error.log").read_text()
    assert "Errored — run aborted" in (output / "routes_aimnet.html").read_text()
    assert (
        json.loads((output / "run_summary.json").read_text())["run_status"] == "aborted"
    )


def test_incompatible_run_preserves_existing_reports(tmp_path, monkeypatch):
    inject(monkeypatch, Calculator())
    source = input_file(tmp_path, ["CCO>>CCO"])
    output = tmp_path / "results"
    cache = tmp_path / "cache.json"
    batch.run_assessment(source, output, AIMNetConfig(), cache)
    saved = {
        name: (output / name).read_bytes()
        for name in (
            "routes_aimnet.html",
            "routes_aimnet.json",
            "run_summary.json",
            "checkpoint.json",
        )
    }
    with pytest.raises(batch.IncompatibleCheckpointError):
        batch.run_assessment(source, output, AIMNetConfig(temperature=310), cache)
    assert {name: (output / name).read_bytes() for name in saved} == saved


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("error_type", [ImportError, FileNotFoundError, RuntimeError])
def test_resume_startup_failure_preserves_publication(
    tmp_path, monkeypatch, partial, error_type
):
    class Backend(Calculator):
        broken_identity = False

        def identity(self):
            if self.broken_identity:
                raise error_type("startup unavailable")
            return super().identity()

        def calculate(self, text):
            if partial and text == "O":
                raise RuntimeError("interrupted calculation")
            return super().calculate(text)

    backend = Backend()
    inject(monkeypatch, backend)
    source = input_file(tmp_path, ["CCO>>CCO", "O>>O"])
    output = tmp_path / "results"
    if partial:
        with pytest.raises(RuntimeError):
            batch.run_assessment(
                source, output, AIMNetConfig(), tmp_path / "cache.json"
            )
    else:
        batch.run_assessment(source, output, AIMNetConfig(), tmp_path / "cache.json")
    saved = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    backend.broken_identity = True
    with pytest.raises(error_type, match="startup unavailable"):
        batch.run_assessment(source, output, AIMNetConfig(), tmp_path / "cache.json")
    assert all((output / name).read_bytes() == data for name, data in saved.items())
    failures = [
        json.loads(p.read_text()) for p in output.glob("attempts/*/failure.json")
    ]
    assert any(r["stage"] == "startup" for r in failures)


def test_cleanup_and_export_errors_do_not_mask_original(tmp_path, monkeypatch):
    from scripts import aimnet_workers

    class Broken(Calculator):
        def __init__(self, *args):
            super().__init__()

        def identity(self):
            raise ImportError("original startup error")

        def close(self):
            raise RuntimeError("cleanup error")

    monkeypatch.setattr(aimnet_workers, "CPUMolecularWorkers", Broken)

    def broken_export(*args):
        raise OSError("export error")

    monkeypatch.setattr(batch, "export_results", broken_export)
    source = input_file(tmp_path, ["O>>O"])
    with pytest.raises(ImportError, match="original startup error") as caught:
        batch.run_assessment(
            source, tmp_path / "new", AIMNetConfig(), tmp_path / "cache.json", workers=2
        )
    assert any("cleanup error" in note for note in caught.value.__notes__)
    assert any("export error" in note for note in caught.value.__notes__)
