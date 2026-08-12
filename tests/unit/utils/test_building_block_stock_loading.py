import gzip

import pytest

from synplan.chem.building_blocks import inchi_to_inchi_key
from synplan.utils.loading import (
    detect_building_blocks_format,
    load_building_block_stock,
)

WATER_INCHI = "InChI=1S/H2O/h1H2"
WATER_KEY = "XLYOFNOQVPJJNP-UHFFFAOYSA-N"


def test_load_smiles_stock_preserves_stereo_and_tab_metadata(tmp_path):
    path = tmp_path / "stock.smi"
    path.write_text("F[C@H](Cl)Br\tleft\nF[C@@H](Cl)Br\tright\n", encoding="utf-8")

    stock = load_building_block_stock(path)

    assert stock.identity_format == "smiles"
    assert len(stock) == 2
    assert any("@" in key for key in stock)


def test_load_smiles_stock_rejects_space_metadata(tmp_path):
    path = tmp_path / "stock.smi"
    path.write_text("CCO ethanol\n", encoding="utf-8")

    with pytest.raises(ValueError, match="TAB-separated"):
        load_building_block_stock(path)


def test_load_plain_inchi_normalizes_to_inchikey(tmp_path):
    path = tmp_path / "stock.inchi"
    path.write_text(f"{WATER_INCHI}\n", encoding="utf-8")

    stock = load_building_block_stock(path)

    assert detect_building_blocks_format(path) == "inchi"
    assert stock.identity_format == "inchikey"
    assert stock.keys == frozenset({WATER_KEY})


def test_load_plain_inchikey_validates_complete_standard_key(tmp_path):
    path = tmp_path / "stock.inchikey"
    path.write_text(f"{WATER_KEY}\n", encoding="utf-8")

    stock = load_building_block_stock(path)

    assert stock.identity_format == "inchikey"
    assert stock.keys == frozenset({WATER_KEY})


@pytest.mark.parametrize("suffix,delimiter", [("csv", ","), ("tsv", "\t")])
def test_load_headered_table_detects_reordered_case_insensitive_inchi_column(
    tmp_path, suffix, delimiter
):
    path = tmp_path / f"stock.{suffix}"
    path.write_text(
        delimiter.join(["ID", "iNcHi"])
        + "\n"
        + delimiter.join(["1", WATER_INCHI])
        + "\n",
        encoding="utf-8",
    )

    stock = load_building_block_stock(path)

    assert stock.keys == frozenset({inchi_to_inchi_key(WATER_INCHI)})
    assert stock.identity_format == "inchikey"


def test_load_gzip_table(tmp_path):
    path = tmp_path / "stock.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(f"InChIKey,ID\n{WATER_KEY},1\n")

    assert load_building_block_stock(path).keys == frozenset({WATER_KEY})


def test_table_rejects_ambiguous_identity_columns(tmp_path):
    path = tmp_path / "stock.tsv"
    path.write_text(f"SMILES\tInChI\nO\t{WATER_INCHI}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        load_building_block_stock(path)


def test_plain_stock_rejects_mixed_identity_formats(tmp_path):
    path = tmp_path / "stock.smi"
    path.write_text(f"O\n{WATER_INCHI}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mixed"):
        load_building_block_stock(path)


def test_identifier_stock_rejects_explicit_standardization(tmp_path):
    path = tmp_path / "stock.inchi"
    path.write_text(f"{WATER_INCHI}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be standardized"):
        load_building_block_stock(path, standardize=True)


def test_loader_alias_rejects_conflicting_formats(tmp_path):
    path = tmp_path / "stock.inchi"
    path.write_text(f"{WATER_INCHI}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="different values"):
        load_building_block_stock(
            path,
            building_blocks_format="inchi",
            input_format="smiles",
        )


def test_empty_and_malformed_stocks_are_rejected(tmp_path):
    empty = tmp_path / "empty.smi"
    empty.write_text("\n", encoding="utf-8")
    malformed = tmp_path / "bad.inchikey"
    malformed.write_text("NOT-A-KEY\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_building_block_stock(empty)
    with pytest.raises(ValueError):
        load_building_block_stock(malformed)
