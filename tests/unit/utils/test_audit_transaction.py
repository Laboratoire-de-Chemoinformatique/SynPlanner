"""Neutral transactional-output primitives shared by audited workflows."""

import json
from pathlib import Path

import pytest

import synplan.utils.audit as audit_utils
from synplan.utils.audit import (
    OutputBundleTransaction,
    partial_output_paths,
    promote_output_bundle,
    sha256_file,
)


def test_failed_bundle_promotion_removes_stale_summary_marker(
    monkeypatch, tmp_path
) -> None:
    finals = {
        "primary": tmp_path / "stock.smi",
        "fallback": tmp_path / "fallback.tsv",
        "summary.json": tmp_path / "summary.json",
    }
    partials = partial_output_paths(finals)
    for key, path in finals.items():
        path.write_text(f"old-{key}\n", encoding="utf-8")
        partials[key].write_text(f"new-{key}\n", encoding="utf-8")

    real_replace = audit_utils.os.replace

    def fail_on_second_data_file(source: str | Path, target: str | Path) -> None:
        if Path(target) == finals["fallback"]:
            raise OSError("simulated promotion failure")
        real_replace(source, target)

    monkeypatch.setattr(audit_utils.os, "replace", fail_on_second_data_file)

    with pytest.raises(OSError, match="simulated promotion failure"):
        promote_output_bundle(finals, partials)

    assert finals["primary"].read_text(encoding="utf-8") == "new-primary\n"
    assert finals["fallback"].read_text(encoding="utf-8") == "old-fallback\n"
    assert not finals["summary.json"].exists()
    assert partials["summary.json"].exists()


def test_transaction_publishes_validated_bundle_and_summary(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    finals = {
        "primary": tmp_path / "run" / "stock.smi",
        "errors": tmp_path / "run" / "errors.tsv",
        "summary.json": tmp_path / "run" / "summary.json",
    }
    transaction = OutputBundleTransaction(
        finals,
        {"input": source},
        "error",
        create_parents=True,
    )
    handles = transaction.open()
    handles["primary"].write("CCO\n")
    handles["errors"].write("# error\n")
    transaction.close()
    transaction.validate_sources_unchanged()
    transaction.validate_line_counts({"primary": 1, "errors": 1})
    metadata = transaction.artifact_metadata()
    assert metadata["primary"]["path"] == str(finals["primary"].resolve())
    assert metadata["primary"]["rows"] == 1
    transaction.write_summary({"output_files": metadata})
    transaction.promote()

    summary = json.loads(finals["summary.json"].read_text(encoding="utf-8"))
    assert summary["output_files"]["primary"]["sha256"] == sha256_file(
        finals["primary"]
    )
    assert not list((tmp_path / "run").glob("*.partial"))


def test_transaction_without_summary_promotes_all_outputs(tmp_path) -> None:
    output = tmp_path / "stock.smi"
    transaction = OutputBundleTransaction(
        {"primary": output}, {}, "replace", summary_key=None
    )
    transaction.open()["primary"].write("CCO\n")
    transaction.close()
    transaction.validate_line_counts({"primary": 1})
    transaction.promote()

    assert output.read_text(encoding="utf-8") == "CCO\n"
    assert not output.with_name("stock.smi.partial").exists()


def test_transaction_preflight_rejects_collisions_directories_and_existing_files(
    tmp_path,
) -> None:
    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    with pytest.raises(ValueError, match="collides"):
        OutputBundleTransaction(
            {"primary": source}, {"input": source}, "replace", summary_key=None
        )

    directory_output = tmp_path / "directory-output"
    directory_output.mkdir()
    with pytest.raises(IsADirectoryError, match="directories"):
        OutputBundleTransaction(
            {"primary": directory_output}, {}, "replace", summary_key=None
        )

    existing = tmp_path / "existing.smi"
    existing.write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exist"):
        OutputBundleTransaction({"primary": existing}, {}, "error", summary_key=None)
    assert existing.read_text(encoding="utf-8") == "old\n"


def test_replace_removes_stale_partial_but_retains_completed_output(tmp_path) -> None:
    output = tmp_path / "stock.smi"
    partial = output.with_name("stock.smi.partial")
    output.write_text("old\n", encoding="utf-8")
    partial.write_text("stale\n", encoding="utf-8")

    transaction = OutputBundleTransaction(
        {"primary": output}, {}, "replace", summary_key=None
    )

    assert output.read_text(encoding="utf-8") == "old\n"
    assert not partial.exists()
    transaction.open()["primary"].write("new\n")
    transaction.close()
    assert output.read_text(encoding="utf-8") == "old\n"
    transaction.promote()
    assert output.read_text(encoding="utf-8") == "new\n"


def test_source_or_line_count_validation_failure_retains_partials(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "stock.smi"
    transaction = OutputBundleTransaction(
        {"primary": output}, {"input": source}, "replace", summary_key=None
    )
    transaction.open()["primary"].write("CCO\n")
    transaction.close()

    with pytest.raises(RuntimeError, match="line-count mismatch"):
        transaction.validate_line_counts({"primary": 2})
    source.write_text("CCN\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input changed during processing"):
        transaction.validate_sources_unchanged()
    assert transaction.partial_paths["primary"].exists()
    assert not output.exists()
