"""Input framing and transactional sidecars shared by the five Synthon CLIs."""

import json
from types import SimpleNamespace

import pytest
from chython import smiles

import synplan.chem.synthon.cli as synthon_cli
from synplan.chem.synthon.audit import (
    ERROR_HEADER,
    FALLBACK_HEADER,
    AuditError,
    AuditOutcome,
    AuditRun,
    iter_molecule_records,
    iter_pathway_records,
    sha256_file,
)
from synplan.chem.synthon.config import SynthonConfig


def test_smi_metadata_is_tab_only_and_preserved_verbatim(tmp_path) -> None:
    path = tmp_path / "catalogue.smi"
    path.write_text(
        "CCO\tethanol\tvendor-a\nCCN legacy-space-name\n",
        encoding="utf-8",
    )

    good, legacy = iter_molecule_records(path)

    assert good.chemistry == "CCO"
    assert good.metadata == ("ethanol", "vendor-a")
    assert good.raw == "CCO\tethanol\tvendor-a"
    assert good.fallback_record == good.raw
    assert json.loads(good.source_info) == {
        "line": 1,
        "metadata": ["ethanol", "vendor-a"],
    }
    assert good.format_error is None
    assert legacy.chemistry == "CCN legacy-space-name"
    assert "TAB-separated" in legacy.format_error


def test_cxsmiles_extension_and_tab_metadata_are_distinguished(tmp_path) -> None:
    path = tmp_path / "radicals.cxsmiles"
    chemistry = "BrC=1C([CH]C=CC=1)=C |^1:3|"
    path.write_text(f"{chemistry}\tradical-lot\n", encoding="utf-8")

    record = next(iter_molecule_records(path))

    assert record.chemistry == chemistry
    assert record.metadata == ("radical-lot",)
    assert record.format_error is None


def test_headered_tsv_retains_named_metadata(tmp_path) -> None:
    path = tmp_path / "catalogue.tsv"
    path.write_text(
        "supplier\tCXSMILES\tname\nvendor-a\tCCO\tethanol\nvendor-b\tB\n",
        encoding="utf-8",
    )

    complete, short = iter_molecule_records(path)

    assert complete.chemistry == "CCO"
    assert complete.metadata_names == ("supplier", "name")
    assert complete.metadata == ("vendor-a", "ethanol")
    assert json.loads(complete.source_info)["metadata"] == {
        "supplier": "vendor-a",
        "name": "ethanol",
    }
    assert complete.fallback_record == ('CCO\t{"supplier":"vendor-a","name":"ethanol"}')
    assert short.format_error == "expected 3 TSV fields, found 2"


@pytest.mark.parametrize(
    "header",
    ("name\tsupplier", "SMILES\tCXSMILES"),
)
def test_headered_tsv_requires_exactly_one_chemistry_column(tmp_path, header) -> None:
    path = tmp_path / "catalogue.tsv"
    path.write_text(header + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one SMILES or CXSMILES"):
        list(iter_molecule_records(path))


def test_fragmentation_input_is_an_exact_five_column_tsv(tmp_path) -> None:
    path = tmp_path / "pathways.tsv"
    valid = "target\tR1.1_0\tC[CH3_elec].N[NH2_nuc]\t1\t0.5000"
    path.write_text(valid + "\nmalformed\n", encoding="utf-8")

    complete, malformed = iter_pathway_records(path)

    assert complete.kind == "pathway"
    assert complete.fields == tuple(valid.split("\t"))
    assert complete.fallback_record == valid
    assert complete.format_error is None
    assert malformed.input_record == "malformed"
    assert malformed.format_error == "expected 5 fragmentation TSV fields, found 1"


def test_audit_run_publishes_consistent_sidecars_and_summary(tmp_path) -> None:
    source = tmp_path / "blocks.smi"
    source.write_text("CCO\tethanol\nB\tboron\nC1CC\tbroken\n", encoding="utf-8")
    records = list(iter_molecule_records(source))
    output = tmp_path / "classes.tsv"
    config = SynthonConfig(write_audit_files=True, audit_overwrite="replace")

    with AuditRun("bb_classifying", source, output, config) as audit:
        audit.write(
            AuditOutcome(
                records[0],
                "classified",
                output_rows=("CCO\tethanol\tAlcohols_Aliphatic_alcohols",),
            )
        )
        audit.write(AuditOutcome(records[1], "unclassified", detail="no class"))
        audit.write(
            AuditOutcome(
                records[2],
                "processing_error",
                detail="parse failed",
                errors=(AuditError("bb_classifier", "parse_error", "bad ring"),),
            )
        )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "CCO\tethanol\tAlcohols_Aliphatic_alcohols"
    ]
    fallback = (tmp_path / "fallback.tsv").read_text(encoding="utf-8").splitlines()
    assert fallback[0] == FALLBACK_HEADER.rstrip("\n")
    assert [line.split("\t")[2] for line in fallback[1:]] == [
        "unclassified",
        "processing_error",
    ]
    assert (tmp_path / "fallback.smi").read_text(encoding="utf-8") == "B\tboron\n"
    errors = (tmp_path / "errors.tsv").read_text(encoding="utf-8").splitlines()
    assert errors[0] == ERROR_HEADER.rstrip("\n")
    assert errors[1].split("\t")[2:4] == ["bb_classifier", "parse_error"]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["command"] == "bb_classifying"
    assert summary["counts"]["input_records"] == 3
    assert summary["counts"]["status_classified"] == 1
    assert summary["counts"]["status_unclassified"] == 1
    assert summary["counts"]["status_processing_error"] == 1
    for name, metadata in summary["output_files"].items():
        artifact = tmp_path / name
        assert metadata["bytes"] == artifact.stat().st_size
        assert metadata["sha256"] == sha256_file(artifact)
    assert not list(tmp_path.glob("*.partial"))


def test_fallback_tsv_preserves_smiles_backslashes_and_json_metadata(tmp_path) -> None:
    source = tmp_path / "blocks.smi"
    source.write_text("F/C=C\\F\tlot\\42\n", encoding="utf-8")
    record = next(iter_molecule_records(source))

    with AuditRun(
        "bb_classifying",
        source,
        tmp_path / "classes.tsv",
        SynthonConfig(write_audit_files=True, audit_overwrite="replace"),
    ) as audit:
        audit.write(AuditOutcome(record, "unclassified", detail="no class"))

    row = (tmp_path / "fallback.tsv").read_text(encoding="utf-8").splitlines()[1]
    chemistry, source_info, status, detail = row.split("\t")
    assert chemistry == "F/C=C\\F"
    assert json.loads(source_info) == {"line": 1, "metadata": ["lot\\42"]}
    assert (status, detail) == ("unclassified", "no class")


def test_processing_errors_can_never_enter_retryable_fallback(tmp_path) -> None:
    source = tmp_path / "blocks.smi"
    source.write_text("C1CC\n", encoding="utf-8")
    record = next(iter_molecule_records(source))

    with AuditRun(
        "bb_classifying",
        source,
        tmp_path / "classes.tsv",
        SynthonConfig(write_audit_files=True, audit_overwrite="replace"),
    ) as audit:
        audit.write(
            AuditOutcome(
                record,
                "processing_error",
                detail="parse failed",
                retryable=True,
            )
        )

    assert (tmp_path / "fallback.smi").read_text(encoding="utf-8") == ""


def test_audit_preflight_rejects_collisions_before_writing(tmp_path) -> None:
    source = tmp_path / "fallback.smi"
    source.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "classes.tsv"

    with pytest.raises(ValueError, match="collides"):
        AuditRun(
            "bb_classifying",
            source,
            output,
            SynthonConfig(write_audit_files=True),
        )

    assert not output.exists()
    assert not (tmp_path / "summary.json").exists()


def test_audit_error_policy_refuses_existing_artifacts(tmp_path) -> None:
    source = tmp_path / "blocks.smi"
    source.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "classes.tsv"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exist"):
        AuditRun(
            "bb_classifying",
            source,
            output,
            SynthonConfig(write_audit_files=True, audit_overwrite="error"),
        )

    assert output.read_text(encoding="utf-8") == "existing\n"


def test_headered_tsv_rejects_duplicate_provenance_names(tmp_path) -> None:
    source = tmp_path / "catalogue.tsv"
    source.write_text("SMILES\tname\tNAME\nCCO\ta\tb\n", encoding="utf-8")

    with pytest.raises(ValueError, match="header names must be unique"):
        list(iter_molecule_records(source))


def test_normalized_tsv_fallback_is_reusable_as_smi(tmp_path) -> None:
    source = tmp_path / "catalogue.tsv"
    source.write_text("SMILES\tname\nB\tboron\n", encoding="utf-8")
    record = next(iter_molecule_records(source))

    with AuditRun(
        "bb_classifying",
        source,
        tmp_path / "classes.tsv",
        SynthonConfig(write_audit_files=True, audit_overwrite="replace"),
    ) as audit:
        audit.write(AuditOutcome(record, "unclassified"))

    retry = next(iter_molecule_records(tmp_path / "fallback.smi"))
    assert retry.chemistry == "B"
    assert json.loads(retry.metadata[0]) == {"name": "boron"}


def test_reserved_partial_sidecar_name_is_rejected(tmp_path) -> None:
    source = tmp_path / "blocks.smi"
    source.write_text("CCO\n", encoding="utf-8")

    with pytest.raises(ValueError, match="reserved"):
        AuditRun(
            "bb_classifying",
            source,
            tmp_path / "fallback.smi.partial",
            SynthonConfig(write_audit_files=True, audit_overwrite="replace"),
        )


def test_changed_input_retains_partials_and_replace_can_retry(tmp_path) -> None:
    source = tmp_path / "blocks.smi"
    source.write_text("B\n", encoding="utf-8")
    record = next(iter_molecule_records(source))
    output = tmp_path / "classes.tsv"
    config = SynthonConfig(write_audit_files=True, audit_overwrite="replace")

    with (
        pytest.raises(RuntimeError, match="input changed during processing"),
        AuditRun("bb_classifying", source, output, config) as audit,
    ):
        source.write_text("CCO\n", encoding="utf-8")
        audit.write(AuditOutcome(record, "unclassified"))

    assert not output.exists()
    assert not (tmp_path / "summary.json").exists()
    assert (tmp_path / "classes.tsv.partial").exists()
    assert (tmp_path / "run.log.partial").exists()

    retry_record = next(iter_molecule_records(source))
    with AuditRun("bb_classifying", source, output, config) as audit:
        audit.write(AuditOutcome(retry_record, "unclassified"))

    assert (tmp_path / "summary.json").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_all_failed_classifier_calls_are_processing_errors(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "blocks.smi"
    source.write_text("CCO\n", encoding="utf-8")
    record = next(iter_molecule_records(source))

    class BrokenClassifier:
        classes = ()

        @staticmethod
        def classify(_molecule):
            raise RuntimeError("classifier failed")

    monkeypatch.setattr(
        synthon_cli._synthonise_workers,
        "_WORKER",
        SimpleNamespace(classifier=BrokenClassifier()),
    )

    outcome = synthon_cli._classify_audit_record(record)
    assert outcome.status == "processing_error"
    assert [error.error_type for error in outcome.errors] == ["classification_error"]


def test_all_failed_synthon_transformations_are_processing_errors(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "blocks.smi"
    source.write_text("CCO\n", encoding="utf-8")
    record = next(iter_molecule_records(source))

    class Classifier:
        @staticmethod
        def classify(_molecule):
            return ["Alcohols_Aliphatic_alcohols"]

    class BrokenWorker:
        config = SynthonConfig()
        classifier = Classifier()

        @staticmethod
        def synthonise(_molecule, _classes):
            raise RuntimeError("transform failed")

    monkeypatch.setattr(synthon_cli._synthonise_workers, "_WORKER", BrokenWorker())

    outcome = synthon_cli._synthonise_audit_record(record)
    assert outcome.status == "processing_error"
    assert [error.error_type for error in outcome.errors] == ["transformation_error"]


def test_enumeration_keeps_products_emitted_before_an_error(tmp_path) -> None:
    source = tmp_path / "pathways.tsv"
    source.write_text(
        "CC\tR1\tC[CH3_elec].N[NH2_nuc]\t1\t1.0000\n",
        encoding="utf-8",
    )
    record = next(iter_pathway_records(source))

    class Stock:
        @staticmethod
        def slots(synthons, _config):
            return {synthon: [synthon] for synthon in synthons}

    class PartialEnumerator:
        @staticmethod
        def enumerate_analogues(_synthons, _slots):
            yield smiles("CC")
            raise RuntimeError("late failure")

    outcome = synthon_cli._enumerate_outcome(
        record, PartialEnumerator(), Stock(), SynthonConfig()
    )

    assert outcome.status == "enumerated"
    assert len(outcome.output_rows) == 1
    assert [error.error_type for error in outcome.errors] == ["enumeration_error"]


def test_non_strict_enumeration_falls_back_to_pathway_synthons(tmp_path) -> None:
    source = tmp_path / "pathways.tsv"
    source.write_text(
        "CCNN\tR1\tC[CH3_elec].N[NH2_nuc]\t1\t0.0000\n",
        encoding="utf-8",
    )
    record = next(iter_pathway_records(source))
    config = SynthonConfig(
        strict_availability=False,
        find_analogues=True,
        ro2_filtration=True,
        mw_lower=0.0,
        mw_upper=1000.0,
    )

    class EmptyStock:
        seen_config = None

        @classmethod
        def slots(cls, synthons, supplied_config):
            cls.seen_config = supplied_config
            return {synthon: [] for synthon in synthons}

    outcome = synthon_cli._enumerate_outcome(
        record, synthon_cli.Enumerator(config), EmptyStock(), config
    )

    assert EmptyStock.seen_config is config
    assert outcome.status == "enumerated"
    assert outcome.output_rows


def test_synthonisation_reports_max_components_before_classification(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "blocks.smi"
    source.write_text("C.C.C.C.C\n", encoding="utf-8")
    record = next(iter_molecule_records(source))
    worker = SimpleNamespace(config=SynthonConfig(max_components=4))
    monkeypatch.setattr(synthon_cli._synthonise_workers, "_WORKER", worker)

    outcome = synthon_cli._synthonise_audit_record(record)

    assert outcome.status == "max_components"
    assert not outcome.errors


def test_completed_empty_synthon_transformation_is_retryable_no_synthon(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "blocks.smi"
    source.write_text("CCO\n", encoding="utf-8")
    record = next(iter_molecule_records(source))

    class Classifier:
        @staticmethod
        def classify(_molecule):
            return ["Alcohols_Aliphatic_alcohols"]

    worker = SimpleNamespace(
        config=SynthonConfig(),
        classifier=Classifier(),
        synthonise=lambda _molecule, _classes: ({}, False),
    )
    monkeypatch.setattr(synthon_cli._synthonise_workers, "_WORKER", worker)

    outcome = synthon_cli._synthonise_audit_record(record)

    assert outcome.status == "no_synthon"
    assert not outcome.errors


def test_enumeration_reports_no_products_when_all_stock_slots_exist(tmp_path) -> None:
    source = tmp_path / "pathways.tsv"
    source.write_text(
        "CC\tR1\tC[CH3_elec].N[NH2_nuc]\t1\t1.0000\n",
        encoding="utf-8",
    )
    record = next(iter_pathway_records(source))

    class Stock:
        @staticmethod
        def slots(synthons, _config):
            return {synthon: [synthon] for synthon in synthons}

    class EmptyEnumerator:
        @staticmethod
        def enumerate_analogues(_synthons, _slots):
            return iter(())

    outcome = synthon_cli._enumerate_outcome(
        record, EmptyEnumerator(), Stock(), SynthonConfig()
    )

    assert outcome.status == "no_products"
    assert not outcome.output_rows
