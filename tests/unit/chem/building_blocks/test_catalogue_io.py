import gzip
import json
import logging
from importlib import import_module

import pytest
from click.testing import CliRunner

from synplan.chem.building_blocks import (
    BuildingBlock,
    SQLiteBuildingBlockCatalogue,
    load_building_block_catalogue,
    match_building_blocks,
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


def test_standardize_json_preserves_stereo_and_merges_vendor_prices(
    tmp_path, monkeypatch
):
    source = tmp_path / "blocks.tsv"
    output = tmp_path / "blocks.json"
    source.write_text(CATALOGUE)

    assert standardize_building_blocks(str(source), str(output)) == str(output)

    raw = json.loads(output.read_text())
    assert len(raw) == 5
    stereo_records = [record for record in raw.values() if record["has_stereo"]]
    assert len(stereo_records) == 4
    assert sum("@" in record["smiles"] for record in stereo_records) == 2
    assert (
        sum(
            "/" in record["smiles"] or "\\" in record["smiles"]
            for record in stereo_records
        )
        == 2
    )
    ethanol = next(record for record in raw.values() if record["smiles"] == "CCO")
    assert ethanol == {
        "smiles": "CCO",
        "vendors": {"LN": 2.0, "SA": 3.0},
        "has_stereo": False,
    }

    catalogue = load_building_block_catalogue(output)
    assert isinstance(catalogue, SQLiteBuildingBlockCatalogue)
    assert catalogue.record_count == len(raw)
    assert sum(map(len, catalogue.values())) == len(raw)
    assert all(isinstance(bucket, tuple) for bucket in catalogue.values())
    assert all(
        isinstance(block, BuildingBlock)
        for bucket in catalogue.values()
        for block in bucket
    )
    loaded = {
        block.inchikey: block for bucket in catalogue.values() for block in bucket
    }
    assert set(loaded) == set(raw)
    for key, record in raw.items():
        assert loaded[key].smiles == record["smiles"]
        assert dict(loaded[key].vendors) == record["vendors"]
        assert loaded[key].has_stereo is record["has_stereo"]
    expected_prefixes = list(dict.fromkeys(key[:14] for key in raw))
    assert list(catalogue) == expected_prefixes
    assert [block.inchikey for bucket in catalogue.values() for block in bucket] == [
        key for prefix in expected_prefixes for key in raw if key[:14] == prefix
    ]
    assert load_building_block_catalogue(output) is catalogue
    monkeypatch.chdir(tmp_path)
    assert load_building_block_catalogue("blocks.json") is catalogue
    assert load_building_block_catalogue(output.resolve()) is catalogue

    first_key = next(iter(raw))
    assert len(catalogue[first_key[:14]]) == 2
    assert match_building_blocks(catalogue, first_key) is catalogue[first_key[:14]]
    assert match_building_blocks(catalogue, "AAAAAAAAAAAAAA-UHFFFAOYSA-N") == ()

    with pytest.raises(TypeError):
        catalogue["AAAAAAAAAAAAAA"] = ()
    with pytest.raises(TypeError):
        catalogue[first_key[:14]][0].vendors["new"] = 1.0


@pytest.mark.parametrize("compressed", [False, True])
def test_loader_rejects_duplicate_full_keys_within_a_bucket(tmp_path, compressed):
    output = tmp_path / ("duplicate.json.gz" if compressed else "duplicate.json")
    record = '{"smiles":"CCO","vendors":{"LN":2.0},"has_stereo":false}'
    data = (
        '{"LFQSCWFLJHTTHZ-UHFFFAOYSA-N":'
        f"{record},"
        '"LFQSCWFLJHTTHZ-UHFFFAOYSA-N":'
        f"{record}}}\n"
    )
    if compressed:
        output.write_bytes(gzip.compress(data.encode()))
    else:
        output.write_text(data)

    with pytest.raises(ValueError, match="duplicate InChIKey"):
        load_building_block_catalogue(output)


def test_prepared_gzip_catalogue_loads_without_repeating_chemistry(
    tmp_path, monkeypatch
):
    from synplan.utils.loading import load_building_blocks

    source, plain, compressed = (
        tmp_path / name for name in ("raw.tsv", "bb.json", "bb.json.gz")
    )
    source.write_text(CATALOGUE)
    standardize_building_blocks(source, plain)
    compressed.write_bytes(gzip.compress(plain.read_bytes()))

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("Prepared catalogues must not repeat chemistry or InChI generation")

    for name in (
        "parse_smiles_preserving_stereo",
        "safe_canonicalization",
        "molecule_to_inchikey",
    ):
        monkeypatch.setattr(catalogue_io, name, unexpected_preparation)
    monkeypatch.setattr(
        "synplan.utils.loading.standardize_smiles_batch", unexpected_preparation
    )
    expected = load_building_block_catalogue(plain)
    actual = load_building_blocks(compressed)
    assert actual == expected
    assert load_building_blocks(plain) is expected
    assert load_building_block_catalogue(compressed) is actual
    assert load_building_blocks(compressed) is actual
    assert sum(map(len, actual.values())) == 5
    assert sum(block.has_stereo for bucket in actual.values() for block in bucket) == 4


def test_truncated_gzip_catalogue_is_not_cached(tmp_path):
    path = tmp_path / "broken.json.gz"
    data = b'{"LFQSCWFLJHTTHZ-UHFFFAOYSA-N":{"smiles":"CCO","vendors":{"SA":2},"has_stereo":false}}'
    compressed = gzip.compress(data)
    path.write_bytes(compressed[:-5])
    with pytest.raises(ValueError, match="Could not read building-block catalogue"):
        load_building_block_catalogue(path)
    path.write_bytes(compressed)
    assert len(load_building_block_catalogue(path)) == 1


def test_parallel_preparation_matches_serial_across_batches(tmp_path):
    source = tmp_path / "raw.tsv.gz"
    plain = tmp_path / "serial.json"
    parallel = tmp_path / "parallel.json.gz"
    header, *rows = CATALOGUE.splitlines(keepends=True)
    rows = rows * 170
    rows.insert(10, "CCO\tbad-price\t0\n")
    rows.insert(1010, "C1CC\t1\t0\n")
    rows.append("OCC\t1\t2\n")
    source.write_bytes(gzip.compress((header + "".join(rows)).encode()))
    standardize_building_blocks(source, plain)

    result = CliRunner().invoke(
        synplan,
        [
            "building_blocks_standardizing",
            "--input",
            str(source),
            "--output",
            str(parallel),
            "--num-workers",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert gzip.decompress(parallel.read_bytes()) == plain.read_bytes()
    assert (
        parallel.with_suffix(".gz.errors.tsv").read_bytes()
        == plain.with_suffix(".json.errors.tsv").read_bytes()
    )
    assert load_building_block_catalogue(parallel) == load_building_block_catalogue(
        plain
    )


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


@pytest.mark.parametrize("compressed", [False, True])
def test_partial_publication_preserves_json_if_atomic_replace_fails(
    tmp_path, monkeypatch, compressed
):
    source = tmp_path / "blocks.tsv"
    output = tmp_path / ("blocks.json.gz" if compressed else "blocks.json")
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
