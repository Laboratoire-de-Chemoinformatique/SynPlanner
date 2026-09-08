"""Six unchanged vendor SMILES with source lines; offers are illustrative.

Source TSV SHA-256: a12e6443f68a113a069cce1525fe5407d166c43e60a40b7c196def0f90fbf40d.
Unreadable prepared output must be rejected; accepted records remain purchasable.
"""

import csv
import json
from pathlib import Path

import pytest
from frozendict import frozendict

from synplan.chem.building_blocks import (
    load_building_blocks,
    standardize_building_blocks,
)
from synplan.chem.precursor import Precursor
from synplan.chem.stereo import parse_smiles_preserving_stereo


@pytest.mark.parametrize("extension", [".sqlite", ".json.gz"])
def test_real_vendor_records_remain_purchasable_or_are_reported(tmp_path, extension):
    rows = json.loads(
        (
            Path(__file__).parents[3] / "data/building_blocks/vendor_self_match.json"
        ).read_text()
    )
    source = tmp_path / "vendor.tsv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]["raw"]), delimiter="\t")
        writer.writeheader()
        writer.writerows(row["raw"] for row in rows)
    output = tmp_path / ("stock" + extension)
    standardize_building_blocks(source, output)
    stock = load_building_blocks(output)
    try:
        assert stock.record_count == 4
        report = Path(str(output) + ".errors.tsv").read_text()
        assert len(report.splitlines()) == 2
        assert "5\tdirectional bond has no assigned stereo element" in report
        for row in rows:
            if row["line"] not in {305831, 122415, 478784, 371213}:
                continue  # the other rows supply an earlier stereoisomer and a second offer
            if row["line"] == 371213:
                assert row["key"][:14] not in stock
                continue
            query = Precursor(parse_smiles_preserving_stereo(row["raw"]["SMILES"]))
            assert query.inchi_key == row["key"]
            assert query.is_purchasable(stock), query.stock_diagnostics
            assert query.selected_stock["inchikey"] == row["key"]
            if row["line"] == 305831:
                assert (
                    query.stock_diagnostics[0]["reason"]
                    == "stock_assessment_incomplete"
                )
                bucket = stock[row["key"][:14]]
                for records in (bucket, tuple(reversed(bucket))):
                    assert query.is_purchasable(frozendict({row["key"][:14]: records}))
            if row["line"] == 122415:
                assert query.selected_stock["vendors"] == {
                    "example_A": 10.0,
                    "example_B": 20.0,
                }
    finally:
        stock.close()
