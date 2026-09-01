import json
import logging
from importlib import import_module

import pytest
from click.testing import CliRunner
from frozendict import frozendict

from synplan.chem.building_blocks import (
    BuildingBlock,
    load_building_block_indexes,
)
from synplan.chem.building_blocks import io as catalogue_io
from synplan.chem.utils import standardize_building_blocks
from synplan.interfaces.cli import synplan

CATALOGUE = """SMILES\tLN_ppg\tSA_ppg
F[C@H](Cl)Br\t10\t0
F[C@@H](Cl)Br\t5\t7
F/C=C/F\t8\t0
F/C=C\\F\t9\t0
CCO\t4\t0
OCC\t2\t3
"""


def test_building_block_is_publicly_reexported_from_core():
    from synplan.chem.building_blocks.core import BuildingBlock as CoreBuildingBlock

    assert BuildingBlock is CoreBuildingBlock
    with pytest.raises(ModuleNotFoundError):
        import_module("synplan.chem.building_blocks.model")


def test_standardize_json_preserves_stereo_and_merges_vendor_prices(tmp_path):
    source = tmp_path / "blocks.tsv"
    output = tmp_path / "blocks.json"
    source.write_text(CATALOGUE)

    assert standardize_building_blocks(str(source), str(output)) == str(output)

    raw = json.loads(output.read_text())
    assert len(raw) == 5
    stereo_records = [record for record in raw.values() if record["has_stereo"]]
    assert len(stereo_records) == 4
    assert sum("@" in record["smiles"] for record in stereo_records) == 2
    assert sum(
        "/" in record["smiles"] or "\\" in record["smiles"]
        for record in stereo_records
    ) == 2
    ethanol = next(record for record in raw.values() if record["smiles"] == "CCO")
    assert ethanol == {
        "smiles": "CCO",
        "vendors": {"LN": 2.0, "SA": 3.0},
        "has_stereo": False,
    }

    by_key, candidates = load_building_block_indexes(output)
    assert isinstance(by_key, frozendict)
    assert isinstance(candidates, frozendict)
    assert all(isinstance(block, BuildingBlock) for block in by_key.values())
    assert all(block in candidates[block.inchikey[:14]] for block in by_key.values())


def test_standardize_json_publishes_valid_rows_and_reports_every_bad_row(
    tmp_path, caplog
):
    source = tmp_path / "blocks.tsv"
    output = tmp_path / "blocks.json"
    output.write_text('{"old":true}\n')
    source.write_text(
        "SMILES\tLN_ppg\n"
        "CCO\t2\n"
        "CCN\tnot-a-price\n"
        "\t3\n"
        "CCCC\t-1\n"
        "CCCl\tinf\n"
        "C1CC\t2\n"
        "CCC\n"
    )

    with caplog.at_level(logging.WARNING):
        assert standardize_building_blocks(str(source), str(output)) == str(output)

    raw = json.loads(output.read_text())
    assert len(raw) == 1
    assert next(iter(raw.values()))["smiles"] == "CCO"
    report = (tmp_path / "blocks.json.errors.tsv").read_text()
    assert "not numeric" in report
    assert "SMILES is empty" in report
    assert "finite and non-negative" in report
    assert "row does not match the header column count" in report
    assert len(report.splitlines()) == 7
    assert "dropping 6 invalid row(s)" in caplog.text


def test_all_invalid_rows_preserve_existing_json(tmp_path):
    source = tmp_path / "blocks.tsv"
    output = tmp_path / "blocks.json"
    output.write_text('{"old":true}\n')
    source.write_text("SMILES\tLN_ppg\nCCO\tnot-a-price\n\t3\nCCN\t-1\n")

    with pytest.raises(ValueError, match="no valid rows; 3 invalid row"):
        standardize_building_blocks(str(source), str(output))

    assert output.read_text() == '{"old":true}\n'
    assert len((tmp_path / "blocks.json.errors.tsv").read_text().splitlines()) == 4


def test_partial_publication_preserves_json_if_atomic_replace_fails(
    tmp_path, monkeypatch
):
    source = tmp_path / "blocks.tsv"
    output = tmp_path / "blocks.json"
    source.write_text("SMILES\tLN_ppg\nCCO\t2\nCCN\tbad-price\n")
    output.write_text('{"old":true}\n')
    real_replace = catalogue_io.os.replace

    def fail_json_replace(source_path, destination_path):
        if destination_path == output:
            raise OSError("simulated atomic publication failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(catalogue_io.os, "replace", fail_json_replace)

    with pytest.raises(OSError, match="simulated atomic publication failure"):
        standardize_building_blocks(str(source), str(output))

    assert output.read_text() == '{"old":true}\n'
    assert not tuple(tmp_path.glob(".blocks.json.*.tmp"))


def test_clean_run_removes_a_stale_error_report(tmp_path):
    source = tmp_path / "blocks.tsv"
    output = tmp_path / "blocks.json"
    error_path = tmp_path / "blocks.json.errors.tsv"
    source.write_text("SMILES\tLN_ppg\nCCO\t2\n")
    error_path.write_text("stale\n")

    standardize_building_blocks(str(source), str(output))

    assert output.exists()
    assert not error_path.exists()


def test_existing_cli_and_python_function_produce_identical_json(tmp_path):
    source = tmp_path / "blocks.tsv"
    direct = tmp_path / "direct.json"
    cli = tmp_path / "cli.json"
    source.write_text(CATALOGUE)
    standardize_building_blocks(str(source), str(direct))

    result = CliRunner().invoke(
        synplan,
        [
            "building_blocks_standardizing",
            "--input",
            str(source),
            "--output",
            str(cli),
        ],
    )

    assert result.exit_code == 0, result.output
    assert cli.read_bytes() == direct.read_bytes()
