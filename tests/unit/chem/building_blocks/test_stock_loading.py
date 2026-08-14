import gzip

import pytest
from chython import smiles as smiles_parser

from synplan.chem.building_blocks import (
    BuildingBlockStockLoadConfig,
    detect_building_blocks_format,
)
from synplan.utils.loading import load_building_block

WATER_INCHI = "InChI=1S/H2O/h1H2"
WATER_KEY = "XLYOFNOQVPJJNP-UHFFFAOYSA-N"


def test_load_smiles_stock_preserves_stereo_and_tab_metadata(tmp_path):
    path = tmp_path / "stock.smi"
    path.write_text("F[C@H](Cl)Br\tleft\nF[C@@H](Cl)Br\tright\n", encoding="utf-8")

    stock = load_building_block(path)

    assert stock.identity_format == "smiles"
    assert len(stock) == 2
    assert any("@" in key for key in stock)


def test_typed_smiles_stock_always_canonicalizes_keys(tmp_path):
    path = tmp_path / "stock.smi"
    path.write_text("OCC\n", encoding="utf-8")

    stock = load_building_block(path)
    assert stock.keys == frozenset({"CCO"})
    assert stock.contains_molecule(smiles_parser("CCO"))


def test_typed_loader_observes_replaced_stock_file(tmp_path):
    path = tmp_path / "stock.smi"
    path.write_text("CCO\n", encoding="utf-8")
    first = load_building_block(path)

    path.write_text("CCN\n", encoding="utf-8")
    second = load_building_block(path)

    assert first.keys == frozenset({"CCO"})
    assert second.keys == frozenset({"CCN"})
    assert first is not second


def test_typed_loader_reads_once_and_parses_each_smiles_once(monkeypatch, tmp_path):
    from synplan.chem.building_blocks import stock as stock_module

    path = tmp_path / "stock.smi"
    path.write_text("OCC\nCCN\n", encoding="utf-8")
    record_reads = 0
    smiles_parses = 0
    original_records = stock_module.iter_chemical_records
    original_parser = stock_module.smiles_parser

    def counted_records(*args, **kwargs):
        nonlocal record_reads
        record_reads += 1
        yield from original_records(*args, **kwargs)

    def counted_parser(*args, **kwargs):
        nonlocal smiles_parses
        smiles_parses += 1
        return original_parser(*args, **kwargs)

    monkeypatch.setattr(stock_module, "iter_chemical_records", counted_records)
    monkeypatch.setattr(stock_module, "smiles_parser", counted_parser)

    stock = stock_module.load_building_block_stock(path)

    assert stock.keys == frozenset({"CCO", "CCN"})
    assert record_reads == 1
    assert smiles_parses == 2


def test_load_smiles_stock_rejects_space_metadata(tmp_path):
    path = tmp_path / "stock.smi"
    path.write_text("CCO ethanol\n", encoding="utf-8")

    with pytest.raises(ValueError, match="TAB-separated"):
        load_building_block(path)


def test_plain_raw_inchi_stock_is_rejected(tmp_path):
    path = tmp_path / "stock.inchi"
    path.write_text(f"{WATER_INCHI}\n", encoding="utf-8")

    for operation in (detect_building_blocks_format, load_building_block):
        with pytest.raises(ValueError, match="raw InChI stock input is unsupported"):
            operation(path)


def test_load_plain_inchikey_validates_complete_standard_key(tmp_path):
    path = tmp_path / "stock.inchikey"
    path.write_text(f"{WATER_KEY}\n", encoding="utf-8")

    stock = load_building_block(path)

    assert stock.identity_format == "inchikey"
    assert stock.keys == frozenset({WATER_KEY})


def test_plain_inchikey_fast_path_skips_generic_record_reader(monkeypatch, tmp_path):
    from synplan.chem.building_blocks import stock as stock_module

    path = tmp_path / "stock.inchikey"
    path.write_text(f"# prepared stock\n{WATER_KEY}\n{WATER_KEY}\n", encoding="utf-8")

    def unexpected_generic_reader(*args, **kwargs):
        raise AssertionError("plain InChIKey stock used the generic record reader")

    monkeypatch.setattr(
        stock_module, "iter_chemical_records", unexpected_generic_reader
    )

    stock = load_building_block(path)

    assert stock.identity_format == "inchikey"
    assert stock.keys == frozenset({WATER_KEY})


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ("NOT-A-KEY", "invalid Standard InChIKey"),
        (f"{WATER_KEY} metadata", "cannot contain whitespace"),
        (WATER_INCHI, "raw InChI stock input is unsupported"),
    ],
)
def test_plain_inchikey_fast_path_rejects_invalid_records(tmp_path, record, message):
    path = tmp_path / "stock.inchikey"
    path.write_text(record + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_building_block(path)


@pytest.mark.parametrize("suffix,delimiter", [("csv", ","), ("tsv", "\t")])
def test_load_headered_table_detects_reordered_case_insensitive_inchikey_column(
    tmp_path, suffix, delimiter
):
    path = tmp_path / f"stock.{suffix}"
    path.write_text(
        delimiter.join(["ID", "iNcHiKeY"])
        + "\n"
        + delimiter.join(["1", WATER_KEY])
        + "\n",
        encoding="utf-8",
    )

    stock = load_building_block(path)

    assert stock.keys == frozenset({WATER_KEY})
    assert stock.identity_format == "inchikey"


def test_load_gzip_table(tmp_path):
    path = tmp_path / "stock.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(f"InChIKey,ID\n{WATER_KEY},1\n")

    assert load_building_block(path).keys == frozenset({WATER_KEY})


def test_stock_load_config_selects_custom_table_column_and_delimiter(tmp_path):
    path = tmp_path / "stock.csv"
    path.write_text("ID;Structure\nethanol;OCC\n", encoding="utf-8")

    stock = load_building_block(
        path,
        config=BuildingBlockStockLoadConfig(
            chemistry_column="structure",
            delimiter=";",
        ),
    )

    assert stock.keys == frozenset({"CCO"})


def test_table_rejects_ambiguous_identity_columns(tmp_path):
    path = tmp_path / "stock.tsv"
    path.write_text(f"SMILES\tInChIKey\nO\t{WATER_KEY}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        load_building_block(path)


def test_plain_stock_rejects_mixed_identity_formats(tmp_path):
    path = tmp_path / "stock.smi"
    path.write_text(f"O\n{WATER_KEY}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mixed"):
        load_building_block(path)


def test_loader_config_rejects_unexpected_identity_format(tmp_path):
    path = tmp_path / "stock.inchikey"
    path.write_text(f"{WATER_KEY}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"requested 'smiles'.*detected 'inchikey'"):
        load_building_block(
            path,
            config=BuildingBlockStockLoadConfig(identity_format="smiles"),
        )


def test_empty_and_malformed_stocks_are_rejected(tmp_path):
    empty = tmp_path / "empty.smi"
    empty.write_text("\n", encoding="utf-8")
    malformed = tmp_path / "bad.inchikey"
    malformed.write_text("NOT-A-KEY\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_building_block(empty)
    with pytest.raises(ValueError):
        load_building_block(malformed)


@pytest.mark.parametrize("terminated", [True, False])
def test_sdf_detect_and_load_reject_every_malformed_record(tmp_path, terminated):
    from chython.files.SDFrw import SDFWrite

    path = tmp_path / "stock.sdf"
    writer = SDFWrite(str(path))
    try:
        writer.write(smiles_parser("CCO"))
    finally:
        writer.close()
    suffix = "broken SDF record\n$$$$\n" if terminated else "broken SDF record\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(suffix)

    for operation in (detect_building_blocks_format, load_building_block):
        with pytest.raises(ValueError, match="SDF record 2"):
            operation(path)


def test_sdf_stock_preserves_tetrahedral_and_double_bond_stereo(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    path = tmp_path / "stock.sdf"
    writer = Chem.SDWriter(str(path))
    try:
        for value in (
            "F[C@H](Cl)Br",
            "F[C@@H](Cl)Br",
            "F/C=C/F",
            "F/C=C\\F",
        ):
            molecule = Chem.MolFromSmiles(value)
            AllChem.Compute2DCoords(molecule)
            writer.write(molecule)
    finally:
        writer.close()

    stock = load_building_block(path)

    assert len(stock) == 4
    assert stock.contains_molecule(smiles_parser("F[C@H](Cl)Br"))
    assert stock.contains_molecule(smiles_parser("F[C@@H](Cl)Br"))
    assert stock.contains_molecule(smiles_parser("F/C=C/F"))
    assert stock.contains_molecule(smiles_parser("F/C=C\\F"))
