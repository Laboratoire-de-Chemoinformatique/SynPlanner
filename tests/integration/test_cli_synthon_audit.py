"""Audited public behavior of all five Synthon CLI commands."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import synplan.interfaces.cli as cli

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "synthon"
SIDECARS = ("fallback.smi", "fallback.tsv", "errors.tsv", "summary.json", "run.log")
FALLBACK_HEADER = "# input_record\tsource_info\tstatus\tdetail"
ERROR_HEADER = "# input_record\tsource_info\tstage\terror_type\terror_message"
CENOBAMATE = "NC(=O)OC(CN1N=CN=N1)C1=CC=CC=C1Cl"


def _config(
    path: Path,
    *,
    overwrite: str = "replace",
    workers: int = 1,
) -> Path:
    path.write_text(
        "\n".join(
            (
                "write_audit_files: true",
                f"audit_overwrite: {overwrite}",
                f"num_workers: {workers}",
                "mw_lower: 0.0",
                "mw_upper: 10000.0",
                "max_products: 100",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _run(*args: str):
    result = CliRunner().invoke(cli.synplan, list(args))
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result


def _sidecars(directory: Path) -> dict[str, Path]:
    paths = {name: directory / name for name in SIDECARS}
    assert all(path.is_file() for path in paths.values())
    assert not list(directory.glob("*.partial"))
    assert paths["fallback.tsv"].read_text(encoding="utf-8").splitlines()[0] == (
        FALLBACK_HEADER
    )
    assert paths["errors.tsv"].read_text(encoding="utf-8").splitlines()[0] == (
        ERROR_HEADER
    )
    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    assert summary["schema_version"]
    return paths


def _fallback_statuses(path: Path) -> list[str]:
    return [
        line.split("\t")[2]
        for line in path.read_text(encoding="utf-8").splitlines()[1:]
    ]


def test_audited_classification_accepts_headered_tsv(tmp_path) -> None:
    case = tmp_path / "classify"
    case.mkdir()
    source = case / "catalogue.tsv"
    source.write_text(
        "SMILES\tname\tsupplier\n"
        "CCO\tethanol\tvendor-a\n"
        "B\tboron\tvendor-b\n"
        "C1CC\tbroken\tvendor-c\n",
        encoding="utf-8",
    )
    output = case / "classes.tsv"

    _run(
        "bb_classifying",
        "--input",
        str(source),
        "--output",
        str(output),
        "--config",
        str(_config(case / "config.yaml")),
    )

    assert output.read_text(encoding="utf-8").split("\t", 2)[:2] == [
        "CCO",
        "ethanol",
    ]
    paths = _sidecars(case)
    assert _fallback_statuses(paths["fallback.tsv"]) == [
        "unclassified",
        "processing_error",
    ]
    assert paths["fallback.smi"].read_text(encoding="utf-8") == (
        'B\t{"name":"boron","supplier":"vendor-b"}\n'
    )
    assert len(paths["errors.tsv"].read_text().splitlines()) == 2


def test_audited_synthonisation_preserves_retryable_smi_record(tmp_path) -> None:
    case = tmp_path / "synthonise"
    case.mkdir()
    source = case / "catalogue.smi"
    source.write_text("CCO\tethanol\nB\tboron\nC1CC\tbroken\n", encoding="utf-8")
    output = case / "stock.smi"

    _run(
        "bb_synthonizing",
        "--input",
        str(source),
        "--output",
        str(output),
        "--config",
        str(_config(case / "config.yaml")),
    )

    assert output.read_text(encoding="utf-8").splitlines()
    paths = _sidecars(case)
    assert _fallback_statuses(paths["fallback.tsv"]) == [
        "unclassified",
        "processing_error",
    ]
    assert paths["fallback.smi"].read_text(encoding="utf-8") == "B\tboron\n"


def test_audited_fragmentation_records_no_pathways(tmp_path) -> None:
    case = tmp_path / "fragment"
    case.mkdir()
    source = case / "targets.smi"
    source.write_text(
        f"{CENOBAMATE}\treadme-target\nCCO\tethanol\nC1CC\tbroken\n",
        encoding="utf-8",
    )
    output = case / "pathways.tsv"

    _run(
        "synthon_fragment",
        "--input",
        str(source),
        "--output",
        str(output),
        "--config",
        str(_config(case / "config.yaml")),
    )

    assert output.read_text(encoding="utf-8").splitlines()
    paths = _sidecars(case)
    assert _fallback_statuses(paths["fallback.tsv"]) == [
        "no_pathways",
        "processing_error",
    ]
    assert paths["fallback.smi"].read_text(encoding="utf-8") == "CCO\tethanol\n"


def test_audited_enumeration_preserves_full_pathway_fallback(tmp_path) -> None:
    prep = tmp_path / "enumerate-prep"
    prep.mkdir()
    blocks = prep / "blocks.smi"
    blocks.write_text((FIXTURES / "BBs.cxsmiles").read_text(), encoding="utf-8")
    stock = prep / "stock.smi"
    _run("bb_synthonizing", "--input", str(blocks), "--output", str(stock))
    target = prep / "target.smi"
    target.write_text("CCCCCC(C)OC(=O)CC\ttarget\n", encoding="utf-8")
    pathways = prep / "pathways.tsv"
    _run(
        "synthon_fragment",
        "--input",
        str(target),
        "--output",
        str(pathways),
        "--stock",
        str(stock),
    )

    case = tmp_path / "enumerate"
    case.mkdir()
    missing = "CCO\tR0\tC[CH3_elec].N[NH2_elec]\t1\t0.0000"
    source = case / "pathways.tsv"
    source.write_text(
        pathways.read_text(encoding="utf-8").splitlines()[0]
        + "\n"
        + missing
        + "\nmalformed\n",
        encoding="utf-8",
    )
    output = case / "library.smi"

    _run(
        "synthon_enumerate",
        "--input",
        str(source),
        "--output",
        str(output),
        "--stock",
        str(stock),
        "--config",
        str(_config(case / "config.yaml")),
    )

    assert output.read_text(encoding="utf-8").splitlines()
    paths = _sidecars(case)
    assert _fallback_statuses(paths["fallback.tsv"]) == [
        "missing_stock_slots",
        "processing_error",
    ]
    assert paths["fallback.smi"].read_text(encoding="utf-8") == missing + "\n"


def test_audited_scaffolds_have_no_retryable_failure_status(tmp_path) -> None:
    case = tmp_path / "scaffold"
    case.mkdir()
    source = case / "blocks.smi"
    source.write_text("CCO\tethanol\nC1CC\tbroken\n", encoding="utf-8")
    output = case / "scaffolds.tsv"

    _run(
        "bb_scaffolds",
        "--input",
        str(source),
        "--output",
        str(output),
        "--config",
        str(_config(case / "config.yaml")),
    )

    assert output.read_text(encoding="utf-8") == "CCO\tlinearMolecule\n"
    paths = _sidecars(case)
    assert _fallback_statuses(paths["fallback.tsv"]) == ["processing_error"]
    assert paths["fallback.smi"].read_text(encoding="utf-8") == ""


def test_audit_sidecars_are_not_created_by_default(tmp_path) -> None:
    case = tmp_path / "disabled"
    case.mkdir()
    source = case / "blocks.smi"
    source.write_text("CCO\tethanol\n", encoding="utf-8")

    _run(
        "bb_classifying",
        "--input",
        str(source),
        "--output",
        str(case / "classes.tsv"),
    )

    assert not any((case / name).exists() for name in SIDECARS)


def test_legacy_space_metadata_is_rejected_but_cxsmiles_is_not(tmp_path) -> None:
    case = tmp_path / "framing"
    case.mkdir()
    cxsmiles = "BrC=1C([CH]C=CC=1)=C |^1:3|"
    source = case / "blocks.cxsmiles"
    source.write_text(
        f"CCN legacy-space-name\n{cxsmiles}\tradical-lot\n",
        encoding="utf-8",
    )

    _run(
        "bb_classifying",
        "--input",
        str(source),
        "--output",
        str(case / "classes.tsv"),
        "--config",
        str(_config(case / "config.yaml")),
    )

    paths = _sidecars(case)
    assert _fallback_statuses(paths["fallback.tsv"]) == [
        "processing_error",
        "unclassified",
    ]
    assert paths["fallback.smi"].read_text(encoding="utf-8") == (
        f"{cxsmiles}\tradical-lot\n"
    )
    errors = paths["errors.tsv"].read_text(encoding="utf-8")
    assert "input_format_error" in errors
    assert "radical-lot" not in errors


def test_cli_overwrite_error_preserves_bundle_and_replace_updates_it(tmp_path) -> None:
    case = tmp_path / "overwrite"
    case.mkdir()
    source = case / "blocks.smi"
    source.write_text("CCO\tethanol\n", encoding="utf-8")
    output = case / "classes.tsv"
    replace = _config(case / "replace.yaml")
    error = _config(case / "error.yaml", overwrite="error")
    args = ("bb_classifying", "--input", str(source), "--output", str(output))

    _run(*args, "--config", str(replace))
    guarded = (output, *(case / name for name in SIDECARS))
    original = {path.name: path.read_bytes() for path in guarded}

    result = CliRunner().invoke(cli.synplan, [*args, "--config", str(error)])
    assert result.exit_code != 0
    assert isinstance(result.exception, FileExistsError)
    assert {path.name: path.read_bytes() for path in guarded} == original
    assert not list(case.glob("*.partial"))

    source.write_text("CCCCO\tbutanol\n", encoding="utf-8")
    _run(*args, "--config", str(replace))
    assert output.read_text(encoding="utf-8").startswith("CCCCO\tbutanol\t")
    assert (case / "summary.json").read_bytes() != original["summary.json"]
    _sidecars(case)


@pytest.mark.parametrize(
    ("command", "output_name"),
    (
        ("bb_classifying", "classes.tsv"),
        ("bb_synthonizing", "stock.smi"),
    ),
)
def test_audited_parallel_output_is_ordered_and_keeps_duplicate_rows(
    tmp_path, command, output_name
) -> None:
    source = tmp_path / "duplicates.smi"
    source.write_text(
        "CCO\tfirst\nCCCCO\tsecond\nCCO\tduplicate\nB\tunclassified\n",
        encoding="utf-8",
    )
    cases = []
    for workers in (1, 2):
        case = tmp_path / f"workers-{workers}-{command}"
        case.mkdir()
        output = case / output_name
        _run(
            command,
            "--input",
            str(source),
            "--output",
            str(output),
            "--config",
            str(_config(case / "config.yaml", workers=workers)),
        )
        paths = _sidecars(case)
        cases.append((output, paths))

    one_output, one_sidecars = cases[0]
    two_output, two_sidecars = cases[1]
    assert one_output.read_bytes() == two_output.read_bytes()
    for name in ("fallback.smi", "fallback.tsv", "errors.tsv"):
        assert one_sidecars[name].read_bytes() == two_sidecars[name].read_bytes()
    if command == "bb_classifying":
        rows = one_output.read_text(encoding="utf-8").splitlines()
        assert [row.split("\t")[1] for row in rows if row.startswith("CCO\t")] == [
            "first",
            "duplicate",
        ]
    else:
        rows = [line.split("\t") for line in one_output.read_text().splitlines()]
        assert sum(row[1] == "CCO" for row in rows) == 4


def test_keep_pg_override_preserves_audit_configuration(tmp_path) -> None:
    case = tmp_path / "keep-pg"
    case.mkdir()
    source = case / "blocks.smi"
    source.write_text("CCO\tethanol\n", encoding="utf-8")

    _run(
        "bb_synthonizing",
        "--input",
        str(source),
        "--output",
        str(case / "stock.smi"),
        "--config",
        str(_config(case / "config.yaml")),
        "--keep-pg",
    )

    summary = json.loads((case / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["keep_protecting_groups"] is True
    assert summary["config"]["write_audit_files"] is True
    assert summary["config"]["audit_overwrite"] == "replace"
    _sidecars(case)


def test_audit_preflight_rejects_reserved_input_and_stock_collisions(tmp_path) -> None:
    reserved = tmp_path / "reserved"
    reserved.mkdir()
    source = reserved / "blocks.smi"
    source.write_text("CCO\n", encoding="utf-8")
    config = _config(reserved / "config.yaml")
    result = CliRunner().invoke(
        cli.synplan,
        [
            "bb_classifying",
            "--input",
            str(source),
            "--output",
            str(reserved / "fallback.tsv"),
            "--config",
            str(config),
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert not (reserved / "summary.json").exists()

    input_collision = tmp_path / "input-collision"
    input_collision.mkdir()
    colliding_input = input_collision / "fallback.smi"
    colliding_input.write_text("CCO\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli.synplan,
        [
            "bb_classifying",
            "--input",
            str(colliding_input),
            "--output",
            str(input_collision / "classes.tsv"),
            "--config",
            str(_config(input_collision / "config.yaml")),
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert not (input_collision / "classes.tsv").exists()

    stock_collision = tmp_path / "stock-collision"
    stock_collision.mkdir()
    stock = stock_collision / "fallback.smi"
    stock.write_text(
        "CC[OH_nuc]\tCCO\tAlcohols_Aliphatic_alcohols\t0\n",
        encoding="utf-8",
    )
    target = stock_collision / "targets.smi"
    target.write_text("CCO\n", encoding="utf-8")
    result = CliRunner().invoke(
        cli.synplan,
        [
            "synthon_fragment",
            "--input",
            str(target),
            "--output",
            str(stock_collision / "pathways.tsv"),
            "--stock",
            str(stock),
            "--config",
            str(_config(stock_collision / "config.yaml")),
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert not (stock_collision / "pathways.tsv").exists()
