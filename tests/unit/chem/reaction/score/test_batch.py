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
