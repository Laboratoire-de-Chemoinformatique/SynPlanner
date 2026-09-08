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
    molecule_to_inchikey,
    standardize_building_blocks,
)
from synplan.chem.building_blocks.stereo import _record_molecule, compatible_records
from synplan.chem.precursor import Precursor
from synplan.chem.stereo import parse_smiles_preserving_stereo


@pytest.fixture(params=[".sqlite", ".json", ".json.gz"])
def prepared_catalogue(tmp_path, request):
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
    output = tmp_path / ("stock" + request.param)
    standardize_building_blocks(source, output)
    stock = load_building_blocks(output)
    try:
        yield rows, output, stock
    finally:
        stock.close()


def test_every_published_record_is_readable_and_matches_itself(prepared_catalogue):
    """Check every accepted representation, without molecule-specific expectations."""
    _, _, stock = prepared_catalogue
    assert stock.record_count
    for prefix, bucket in stock.items():
        for record in bucket:
            query = parse_smiles_preserving_stereo(record.smiles)
            assert molecule_to_inchikey(query) == record.inchikey
            # Exact stock must remain reachable even with virtually no mapping budget.
            for catalogue in (stock, frozendict({prefix: tuple(reversed(bucket))})):
                _record_molecule.cache_clear()
                for _ in range(2):  # cold and warm chemistry caches
                    assert record in compatible_records(
                        query, catalogue, max_mapping_work=1
                    ), record.inchikey


def test_real_vendor_records_remain_purchasable_or_are_reported(prepared_catalogue):
    rows, output, stock = prepared_catalogue
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
            assert query.stock_diagnostics[0]["reason"] == "stock_assessment_incomplete"
            bucket = stock[row["key"][:14]]
            for records in (bucket, tuple(reversed(bucket))):
                assert query.is_purchasable(frozendict({row["key"][:14]: records}))
        if row["line"] == 122415:
            assert query.selected_stock["vendors"] == {
                "example_A": 10.0,
                "example_B": 20.0,
            }
