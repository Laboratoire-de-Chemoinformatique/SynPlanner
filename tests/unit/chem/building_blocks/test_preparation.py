"""End-to-end preparation output contracts."""

import csv
import json
from pathlib import Path

import pytest

from synplan.chem.building_blocks.catalog import BuildingBlockCatalog
from synplan.chem.building_blocks.config import BuildingBlockPreparationConfig
from synplan.chem.building_blocks.preparation import (
    PreparationRun,
    prepare_building_blocks,
)
from synplan.chem.building_blocks.reports import (
    DUPLICATE_FIELDS,
    IDENTITY_FIELDS,
    STEREO_FIELDS,
)
from synplan.utils.audit import sha256_file


def _lines(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def test_default_smi_is_stereo_preserving_deduplicated_and_has_no_sidecars(
    tmp_path,
) -> None:
    source = tmp_path / "input.smi"
    source.write_text(
        "N[C@@H](C)C(=O)O\tleft\nN[C@H](C)C(=O)O\tright\nN[C@@H](C)C(=O)O\tduplicate\n",
        encoding="utf-8",
    )
    output = tmp_path / "stock.smi"
    result = prepare_building_blocks(
        source, output, BuildingBlockPreparationConfig(num_workers=1)
    )
    rows = _lines(output)
    assert len(rows) == 2
    assert rows[0] != rows[1]
    assert "@" in rows[0] and "@" in rows[1]
    assert result.synthon_input == str(output.resolve())
    assert result.counts["primary_rows"] == 2
    assert not (tmp_path / "fallback.tsv").exists()


@pytest.mark.parametrize("mode", ["replace", "append"])
def test_deprotection_keeps_protected_synthon_feed_invariant(tmp_path, mode) -> None:
    source = tmp_path / "input.smi"
    source.write_text(
        "c1ccccc1NC(=O)OC(C)(C)C\tone\nc1ccccc1N\ttwo\n",
        encoding="utf-8",
    )
    output = tmp_path / f"{mode}.smi"
    config = BuildingBlockPreparationConfig(
        deprotect=True,
        deprotect_output=mode,
        stereo_file=str(tmp_path / f"{mode}.stereo.tsv"),
        num_workers=1,
    )
    result = prepare_building_blocks(source, output, config)
    protected = _lines(result.synthon_input)
    assert len(protected) == 2
    assert any("OC(Nc1ccccc1)=O" in value for value in protected)
    output_rows = _lines(output)
    assert len(output_rows) == (1 if mode == "replace" else 2)
    assert Path(result.duplicates_file).exists()


def test_headered_gzipped_table_retains_named_provenance(tmp_path) -> None:
    import gzip

    source = tmp_path / "input.tsv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        handle.write("ID\tSMILES\tSupplier\n1\tCCO\tA\n")
    output = tmp_path / "stock.smi"
    result = prepare_building_blocks(
        source, output, BuildingBlockPreparationConfig(num_workers=1)
    )
    assert _lines(output) == ["CCO"]
    assert result.counts["successful_input_records"] == 1


def test_audit_outputs_partition_processing_errors(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text("CCO\tgood\nCCO legacy-name\n", encoding="utf-8")
    output = tmp_path / "run" / "stock.smi"
    result = prepare_building_blocks(
        source,
        output,
        BuildingBlockPreparationConfig(
            write_audit_files=True,
            audit_overwrite="error",
            num_workers=1,
        ),
    )
    assert set(result.audit_files) == {
        "fallback.smi",
        "fallback.tsv",
        "errors.tsv",
        "summary.json",
        "run.log",
    }
    assert _lines(result.audit_files["fallback.smi"]) == []
    assert len(_lines(result.audit_files["fallback.tsv"])) == 2
    assert len(_lines(result.audit_files["errors.tsv"])) == 2
    summary = json.loads(Path(result.audit_files["summary.json"]).read_text())
    assert summary["schema_version"] == 2
    assert summary["counts"]["input_records"] == 2
    assert summary["counts"]["successful_input_records"] == 1
    assert summary["counts"]["processing_errors"] == 1
    for artifact in summary["output_files"].values():
        artifact_path = Path(artifact["path"])
        assert artifact["sha256"] == sha256_file(artifact_path)
        assert artifact["rows"] == len(_lines(artifact_path))
    log = _lines(result.audit_files["run.log"])
    assert len(log) == 2
    assert "total=2" in log[0]
    assert "completed input=2" in log[1]
    assert not list(output.parent.glob("*.partial"))


def test_run_log_contains_flushed_periodic_progress(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text("CCO\nCCN\nCCC\n", encoding="utf-8")
    run = PreparationRun(
        source,
        tmp_path / "run" / "building_blocks.smi",
        BuildingBlockPreparationConfig(
            write_audit_files=True,
            num_workers=1,
        ),
    )
    run.progress_every = 2
    run.next_progress = 2

    result = run.run()

    log = _lines(result.audit_files["run.log"])
    assert len(log) == 3
    assert "total=3" in log[0]
    assert "processed=2/3" in log[1]
    assert "rate=" in log[1]
    assert "completed input=3" in log[2]


def test_audit_overwrite_refuses_existing_bundle(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "run" / "stock.smi"
    config = BuildingBlockPreparationConfig(write_audit_files=True, num_workers=1)
    prepare_building_blocks(source, output, config)
    with pytest.raises(FileExistsError):
        prepare_building_blocks(source, output, config)


def test_identity_outputs_contain_full_standard_inchikey(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "stock.smi"
    result = prepare_building_blocks(
        source,
        output,
        BuildingBlockPreparationConfig(write_inchikey_stock=True, num_workers=1),
    )
    assert _lines(result.inchikey_file) == ["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"]
    with Path(result.identity_reference_file).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["standard_inchi"] == "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
    assert rows[0]["standard_inchikey"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def test_deprotect_replace_writes_identity_for_an_unchanged_molecule(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "stock.smi"
    result = prepare_building_blocks(
        source,
        output,
        BuildingBlockPreparationConfig(
            deprotect=True, write_inchikey_stock=True, num_workers=1
        ),
    )
    assert _lines(result.inchikey_file) == ["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"]
    with Path(result.identity_reference_file).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["output_origin"] == "protected"
    assert rows[0]["status"] == "written"
    assert rows[0]["note"] == "no_protective_group"


def test_audit_replace_commits_a_complete_new_bundle(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "run" / "stock.smi"
    prepare_building_blocks(
        source,
        output,
        BuildingBlockPreparationConfig(write_audit_files=True, num_workers=1),
    )
    source.write_text("CCN\n", encoding="utf-8")
    prepare_building_blocks(
        source,
        output,
        BuildingBlockPreparationConfig(
            write_audit_files=True, audit_overwrite="replace", num_workers=1
        ),
    )
    assert _lines(output) == ["CCN"]
    assert not list(output.parent.glob("*.partial"))


def test_audit_rejects_input_artifact_collision_without_touching_input(
    tmp_path,
) -> None:
    output = tmp_path / "run" / "stock.smi"
    source = output.parent / "fallback.tsv"
    source.parent.mkdir()
    source.write_text("SMILES\nCCO\n", encoding="utf-8")
    original = source.read_bytes()
    with pytest.raises(ValueError, match="input path collides"):
        prepare_building_blocks(
            source,
            output,
            BuildingBlockPreparationConfig(write_audit_files=True, num_workers=1),
        )
    assert source.read_bytes() == original
    assert not output.exists()


def test_single_and_multi_worker_outputs_are_order_equivalent(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text(
        "c1ccccc1NC(=O)OC(C)(C)C\tboc\n"
        "N[C@@H](C)C(=O)O\tleft\n"
        "CCO\tplain\n"
        "CCO legacy-name\n",
        encoding="utf-8",
    )

    def run(name: str, workers: int):
        directory = tmp_path / name
        output = directory / "stock.smi"
        return prepare_building_blocks(
            source,
            output,
            BuildingBlockPreparationConfig(
                deprotect=True,
                deprotect_output="append",
                write_inchikey_stock=True,
                stereo_file=str(directory / "stereo.tsv"),
                write_audit_files=True,
                num_workers=workers,
                batch_size=2,
            ),
        )

    one = run("one", 1)
    many = run("many", 2)
    paths = (
        "output_file",
        "protected_output_file",
        "inchikey_file",
        "identity_reference_file",
        "duplicates_file",
        "collisions_file",
        "stereo_file",
    )
    for attribute in paths:
        assert (
            Path(getattr(one, attribute)).read_bytes()
            == Path(getattr(many, attribute)).read_bytes()
        )
    for name in ("fallback.smi", "fallback.tsv", "errors.tsv"):
        assert (
            Path(one.audit_files[name]).read_bytes()
            == Path(many.audit_files[name]).read_bytes()
        )


def test_preparation_writes_self_describing_price_artifact(tmp_path) -> None:
    source = tmp_path / "input.tsv"
    source.write_text(
        "SMILES\tLN_ppg\tSA_ppg\nCCO\t1.5\t0\nCCN\t\t2.5\n",
        encoding="utf-8",
    )
    result = prepare_building_blocks(
        source,
        tmp_path / "building_blocks.smi",
        BuildingBlockPreparationConfig(write_inchikey_stock=True, num_workers=1),
    )

    assert result.price_reference_file is not None
    assert Path(result.price_reference_file).name == "building_blocks_prices.tsv"
    with Path(result.price_reference_file).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert rows == [
        {"source_index": "1", "input_smiles": "CCO", "LN_ppg": "1.5", "SA_ppg": "0"},
        {"source_index": "2", "input_smiles": "CCN", "LN_ppg": "", "SA_ppg": "2.5"},
    ]
    catalog = BuildingBlockCatalog.from_files(
        result.identity_reference_file, result.price_reference_file
    )
    assert catalog.prices_by_source[1]["LN_ppg"] == 1.5
    assert catalog.prices_by_source[2]["SA_ppg"] == 2.5


def test_duplicate_and_stereo_reports_work_without_deprotection(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text(
        "N[C@@H](C)C(=O)O\tfirst\nN[C@@H](C)C(=O)O\tduplicate\n",
        encoding="utf-8",
    )
    output = tmp_path / "stock.smi"
    result = prepare_building_blocks(
        source,
        output,
        BuildingBlockPreparationConfig(
            duplicates_file=str(tmp_path / "duplicates.tsv"),
            stereo_file=str(tmp_path / "stereo.tsv"),
            num_workers=1,
        ),
    )
    duplicate_rows = _lines(result.duplicates_file)
    stereo_rows = _lines(result.stereo_file)
    assert duplicate_rows[0] == "\t".join(DUPLICATE_FIELDS)
    assert stereo_rows[0] == "\t".join(STEREO_FIELDS)
    assert "first_source_info" not in DUPLICATE_FIELDS
    assert "duplicate_source_info" not in DUPLICATE_FIELDS
    assert "source_info" not in IDENTITY_FIELDS
    assert "source_info" not in STEREO_FIELDS
    assert len(duplicate_rows) == 2
    assert len(stereo_rows) == 3
    assert result.counts["duplicate_rows"] == 1
    assert result.counts["stereo_rows"] == 2


def test_full_pipeline_preset_writes_every_structured_report(tmp_path) -> None:
    preset_path = (
        Path(__file__).resolve().parents[4]
        / "configs"
        / "building_blocks_full_pipeline.yaml"
    )
    config = BuildingBlockPreparationConfig.from_yaml(str(preset_path)).model_copy(
        update={"num_workers": 1}
    )
    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    result = prepare_building_blocks(
        source, tmp_path / "run" / "building_blocks.smi", config
    )
    report_paths = (
        result.protected_output_file,
        result.inchikey_file,
        result.identity_reference_file,
        result.duplicates_file,
        result.collisions_file,
        result.stereo_file,
        *result.audit_files.values(),
    )
    assert all(path is not None and Path(path).is_file() for path in report_paths)
    assert Path(result.output_file).name == "building_blocks.smi"
    assert Path(result.protected_output_file).name == "building_blocks_protected.smi"
    assert Path(result.inchikey_file).name == "building_blocks.inchikey"
    assert Path(result.identity_reference_file).name == "building_blocks_identity.tsv"
    assert Path(result.duplicates_file).name == "building_blocks_duplicates.tsv"
    assert Path(result.collisions_file).name == "building_blocks_collisions.tsv"
    assert Path(result.stereo_file).name == "building_blocks_stereo.tsv"
    assert _lines(result.duplicates_file)[0] == "\t".join(DUPLICATE_FIELDS)
    assert _lines(result.stereo_file)[0] == "\t".join(STEREO_FIELDS)
    summary = json.loads(Path(result.audit_files["summary.json"]).read_text())
    from rdkit import rdBase
    from rdkit.Chem import rdinchi

    from synplan.chem.building_blocks.rules import protective_rules_path

    assert summary["protective_rules"]["sha256"] == sha256_file(protective_rules_path())
    assert summary["engines"] == {
        "rdkit": rdBase.rdkitVersion,
        "inchi": rdinchi.GetInchiVersion(),
    }


def test_failed_validation_retains_staged_partials(monkeypatch, tmp_path) -> None:
    from synplan.chem.building_blocks import preparation as preparation_module

    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "run" / "stock.smi"

    def fail_validation(self, counts) -> None:
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(
        preparation_module.OutputBundleTransaction,
        "validate_line_counts",
        fail_validation,
    )
    with pytest.raises(RuntimeError, match="injected validation failure"):
        prepare_building_blocks(
            source,
            output,
            BuildingBlockPreparationConfig(
                write_audit_files=True,
                num_workers=1,
            ),
        )
    partials = {path.name for path in output.parent.glob("*.partial")}
    assert "stock.smi.partial" in partials
    assert "fallback.tsv.partial" in partials
    assert "errors.tsv.partial" in partials
    assert not output.exists()
    assert not (output.parent / "summary.json").exists()


def test_complete_cxsmiles_with_multiple_tab_metadata_fields_is_accepted(
    tmp_path,
) -> None:
    source = tmp_path / "input.cxsmiles"
    source.write_text(
        "BrC=1C([CH]C=CC=1)=C |^1:3|\tcompound-1\tsupplier-a\n",
        encoding="utf-8",
    )
    output = tmp_path / "stock.smi"
    result = prepare_building_blocks(
        source,
        output,
        BuildingBlockPreparationConfig(num_workers=1),
    )
    assert result.counts["input_records"] == 1
    assert result.counts["successful_input_records"] == 1
    assert len(_lines(output)) == 1


def test_sdf_stereo_report_omits_source_metadata(tmp_path) -> None:
    from chython import smiles
    from chython.files.SDFrw import SDFWrite

    source = tmp_path / "input.sdf"
    molecule = smiles("N[C@@H](C)C(=O)O", ignore=True)
    molecule.meta["Supplier"] = "A"
    writer = SDFWrite(str(source))
    try:
        writer.write(molecule)
    finally:
        writer.close()

    result = prepare_building_blocks(
        source,
        tmp_path / "stock.smi",
        BuildingBlockPreparationConfig(
            stereo_file=str(tmp_path / "stereo.tsv"),
            num_workers=1,
        ),
    )
    with Path(result.stereo_file).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert "source_info" not in rows[0]
    assert rows[0]["source_index"] == "1"
    assert rows[0]["canonical_smiles"] == _lines(result.output_file)[0]


def test_sdf_audit_partitions_valid_and_malformed_framed_records(tmp_path) -> None:
    from chython import smiles
    from chython.files.SDFrw import SDFWrite

    source = tmp_path / "input.sdf"
    writer = SDFWrite(str(source))
    try:
        writer.write(smiles("CCO", ignore=True))
    finally:
        writer.close()
    with source.open("a", encoding="utf-8") as handle:
        handle.write("broken SDF record\n$$$$\n")

    result = prepare_building_blocks(
        source,
        tmp_path / "run" / "stock.smi",
        BuildingBlockPreparationConfig(write_audit_files=True, num_workers=1),
    )
    assert result.counts["input_records"] == 2
    assert result.counts["successful_input_records"] == 1
    assert result.counts["processing_errors"] == 1
    assert len(_lines(result.output_file)) == 1
    assert len(_lines(result.audit_files["fallback.tsv"])) == 2
    assert len(_lines(result.audit_files["errors.tsv"])) == 2
    assert "processing_error" in _lines(result.audit_files["fallback.tsv"])[1]
    assert "invalid SDF record" in _lines(result.audit_files["errors.tsv"])[1]
    summary = json.loads(Path(result.audit_files["summary.json"]).read_text())
    assert summary["counts"]["input_records"] == 2
    assert (
        summary["counts"]["successful_input_records"]
        + summary["counts"]["processing_errors"]
        == summary["counts"]["input_records"]
    )


def test_deprotection_fails_if_taxonomy_changes_during_run(
    monkeypatch, tmp_path
) -> None:
    from synplan.chem.building_blocks import preparation as preparation_module

    taxonomy = tmp_path / "protective_rules.tsv"
    taxonomy.write_text("original taxonomy\n", encoding="utf-8")
    monkeypatch.setattr(preparation_module, "protective_rules_path", lambda: taxonomy)
    process_records = preparation_module._processed_records

    def mutate_taxonomy(records, config):
        for processed in process_records(records, config):
            taxonomy.write_text("mutated taxonomy\n", encoding="utf-8")
            yield processed

    monkeypatch.setattr(
        preparation_module,
        "_processed_records",
        mutate_taxonomy,
    )
    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "run" / "stock.smi"
    with pytest.raises(RuntimeError, match="taxonomy changed during preparation"):
        prepare_building_blocks(
            source,
            output,
            BuildingBlockPreparationConfig(
                deprotect=True,
                write_audit_files=True,
                num_workers=1,
            ),
        )
    assert not output.exists()
    assert not (output.parent / "summary.json").exists()
    assert (output.parent / "stock.smi.partial").exists()


def test_collision_report_retains_duplicate_source_relations(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text(
        "NCC(=O)O\tneutral-first\n"
        "NCC(=O)O\tneutral-duplicate\n"
        "[NH3+]CC(=O)[O-]\tzwitterion\n",
        encoding="utf-8",
    )
    result = prepare_building_blocks(
        source,
        tmp_path / "stock.smi",
        BuildingBlockPreparationConfig(write_inchikey_stock=True, num_workers=1),
    )
    assert len(_lines(result.output_file)) == 2
    assert len(_lines(result.inchikey_file)) == 1
    with Path(result.collisions_file).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 2
    source_relations = sorted(json.loads(row["source_indexes"]) for row in rows)
    assert source_relations == [[1, 2], [3]]


def test_failed_replacement_promotion_invalidates_old_summary_marker(
    monkeypatch, tmp_path
) -> None:
    from synplan.utils import audit as audit_module

    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "run" / "stock.smi"
    prepare_building_blocks(
        source,
        output,
        BuildingBlockPreparationConfig(write_audit_files=True, num_workers=1),
    )
    summary = output.parent / "summary.json"
    assert summary.exists()
    source.write_text("CCN\n", encoding="utf-8")

    replace = audit_module.os.replace
    calls = 0

    def fail_mid_promotion(source_path, destination_path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected promotion failure")
        replace(source_path, destination_path)

    monkeypatch.setattr(audit_module.os, "replace", fail_mid_promotion)
    with pytest.raises(OSError, match="injected promotion failure"):
        prepare_building_blocks(
            source,
            output,
            BuildingBlockPreparationConfig(
                write_audit_files=True,
                audit_overwrite="replace",
                num_workers=1,
            ),
        )
    assert calls == 2
    assert not summary.exists()


def test_identity_reference_retains_processing_errors_without_audit(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text("CCO\nnot-a-smiles\n", encoding="utf-8")
    identity = tmp_path / "identity.tsv"

    result = prepare_building_blocks(
        source,
        tmp_path / "stock.smi",
        BuildingBlockPreparationConfig(
            write_inchikey_stock=True,
            identity_reference_file=str(identity),
            num_workers=1,
        ),
    )

    with identity.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert result.counts["input_records"] == 2
    assert result.counts["processing_errors"] == 1
    assert len(rows) == 2
    failed = next(row for row in rows if row["status"] == "processing_error")
    assert failed["input_smiles"] == "not-a-smiles"
    assert failed["standard_inchikey"] == ""
    assert "standardization" in failed["note"]
    assert not (tmp_path / "fallback.tsv").exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_sdf_preparation_preserves_tetrahedral_and_double_bond_stereo(
    tmp_path, workers
) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    source = tmp_path / "input.sdf"
    writer = Chem.SDWriter(str(source))
    try:
        for index, value in enumerate(
            (
                "F[C@H](Cl)Br",
                "F[C@@H](Cl)Br",
                "F/C=C/F",
                "F/C=C\\F",
            ),
            1,
        ):
            molecule = Chem.MolFromSmiles(value)
            AllChem.Compute2DCoords(molecule)
            molecule.SetProp("vendor", f"vendor-{index}")
            writer.write(molecule)
    finally:
        writer.close()

    identity = tmp_path / "identity.tsv"
    result = prepare_building_blocks(
        source,
        tmp_path / "stock.smi",
        BuildingBlockPreparationConfig(
            write_inchikey_stock=True,
            identity_reference_file=str(identity),
            num_workers=workers,
        ),
    )

    assert result.counts["input_records"] == 4
    assert result.counts["primary_rows"] == 4
    prepared = _lines(result.output_file)
    assert len(prepared) == 4

    assert len(set(prepared)) == 4
    with identity.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len({row["standard_inchikey"] for row in rows}) == 4
    assert all("source_info" not in row for row in rows)


def test_legacy_utils_standardization_delegates_to_canonical_wrapper(tmp_path) -> None:
    from synplan.chem.utils import standardize_building_blocks

    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    output = tmp_path / "stock.smi"
    assert standardize_building_blocks(str(source), str(output)) == str(
        output.resolve()
    )
    assert _lines(output) == ["CCO"]


def test_preparation_run_owns_and_executes_workflow_state(tmp_path) -> None:
    source = tmp_path / "input.smi"
    source.write_text("CCO\nCCO\n", encoding="utf-8")
    run = PreparationRun(
        source,
        tmp_path / "stock.smi",
        BuildingBlockPreparationConfig(num_workers=1),
    )

    result = run.run()

    assert _lines(result.output_file) == ["CCO"]
    assert run.counts["input_records"] == 2
    assert run.counts["primary_rows"] == 1
    assert result.counts == dict(sorted(run.counts.items()))


def test_plain_smiles_is_parsed_once_when_writing_identity(
    monkeypatch, tmp_path
) -> None:
    from synplan.chem.building_blocks import preparation as preparation_module

    source = tmp_path / "input.smi"
    source.write_text("OCC\n", encoding="utf-8")
    parser = preparation_module.smiles_parser
    calls: list[str] = []

    def counting_parser(value, **kwargs):
        calls.append(value)
        return parser(value, **kwargs)

    monkeypatch.setattr(preparation_module, "smiles_parser", counting_parser)
    result = prepare_building_blocks(
        source,
        tmp_path / "stock.smi",
        BuildingBlockPreparationConfig(write_inchikey_stock=True, num_workers=1),
    )

    assert calls == ["OCC"]
    assert _lines(result.output_file) == ["CCO"]
    assert _lines(result.inchikey_file) == ["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"]


def test_sdf_molecule_is_reused_without_smiles_reparsing(monkeypatch, tmp_path) -> None:
    from rdkit import Chem

    from synplan.chem.building_blocks import preparation as preparation_module

    source = tmp_path / "input.sdf"
    writer = Chem.SDWriter(str(source))
    try:
        writer.write(Chem.MolFromSmiles("F/C=C/F"))
    finally:
        writer.close()

    def reject_reparse(*_args, **_kwargs):
        raise AssertionError("SDF molecule must not be reparsed as SMILES")

    monkeypatch.setattr(preparation_module, "smiles_parser", reject_reparse)
    result = prepare_building_blocks(
        source,
        tmp_path / "stock.smi",
        BuildingBlockPreparationConfig(write_inchikey_stock=True, num_workers=1),
    )

    assert result.counts["processing_errors"] == 0
    assert result.counts["identity_rows"] == 1
    assert len(_lines(result.inchikey_file)) == 1


def test_worker_copies_a_preparsed_source_molecule() -> None:
    from chython import smiles

    from synplan.chem.building_blocks import preparation as preparation_module
    from synplan.utils.files import ChemicalRecord

    molecule = smiles("F[C@H](Cl)Br", ignore=True)
    original = str(molecule)
    record = ChemicalRecord(
        sequence=1,
        line_number=1,
        chemistry=original,
        raw=original,
        molecule=molecule,
    )

    processed = preparation_module._process_record(
        (record, False, BuildingBlockPreparationConfig().deprotect_policy)
    )

    assert not processed.failed
    assert processed.protected_molecule is not molecule
    assert str(molecule) == original
    assert "@" in str(processed.protected_molecule)


def test_deprotected_identity_records_exact_replay_provenance(tmp_path) -> None:
    from chython import smiles
    from chython.containers import ReactionContainer

    from synplan.chem.building_blocks.rules import protective_rules_path

    source = tmp_path / "input.smi"
    source.write_text(
        "CC(C)(C)OC(=O)NCCOCc1ccccc1\n",
        encoding="utf-8",
    )
    result = prepare_building_blocks(
        source,
        tmp_path / "stock.smi",
        BuildingBlockPreparationConfig(
            deprotect=True,
            deprotect_policy="conservative",
            write_inchikey_stock=True,
            num_workers=1,
        ),
    )
    with Path(result.identity_reference_file).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    deprotected = next(row for row in rows if row["output_origin"] == "deprotected")

    assert deprotected["standardized_input_smiles"]
    assert deprotected["deprotection_policy"] == "conservative"
    assert deprotected["protective_rules_sha256"] == sha256_file(
        protective_rules_path()
    )
    events = json.loads(deprotected["deprotection_events"])
    assert [event["rule_name"] for event in events] == ["amine_boc"]
    assert events[0]["pass_index"] == 0
    assert events[0]["query_mapping"]

    reaction = smiles(deprotected["mapped_deprotection"])
    assert isinstance(reaction, ReactionContainer)
    assert str(reaction.reactants[0]) == deprotected["standardized_input_smiles"]
    assert str(reaction.products[0]) == deprotected["canonical_smiles"]

    catalog = BuildingBlockCatalog.from_files(result.identity_reference_file)
    records = catalog.protected_alternative_records(deprotected["canonical_smiles"])
    assert records[0]["mapped_deprotection"] == deprotected["mapped_deprotection"]
