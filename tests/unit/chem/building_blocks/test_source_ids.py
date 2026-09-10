"""Supplier IDs and stereo declarations survive stock selection and export."""

import csv
import gzip
import json
import re
import sqlite3
from html import unescape

import pytest
from chython import inchi_key, smiles

from synplan.chem.building_blocks import load_building_blocks
from synplan.chem.building_blocks.database import build_catalogue
from synplan.chem.building_blocks.io import standardize_building_block_catalogue
from synplan.chem.building_blocks.stereo import compatible_records, selected_record
from synplan.chem.precursor import Precursor
from synplan.chem.reaction.reactor import Reaction
from synplan.chem.reaction.routes.route import Route, Step
from synplan.chem.reaction.routes.stereo import match_stereo_stock
from synplan.chem.stereo import has_stereo
from synplan.utils.visualisation import routes_report_html


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def test_legacy_sqlite_still_loads_and_exports(tmp_path):
    molecule = smiles("C[C@H](O)C(=O)O")
    key = inchi_key(molecule)
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE blocks (inchikey, smiles, vendors, has_stereo)"
        )
        connection.execute(
            "INSERT INTO blocks VALUES (?,?,?,?)", (key, str(molecule), '{"MP":5}', 1)
        )
        connection.execute("CREATE TABLE metadata (json)")
        connection.execute(
            "INSERT INTO metadata VALUES (?)",
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "records": 1,
                        "buckets": 1,
                        "cache_id": "legacy",
                    }
                ),
            ),
        )
    stock = load_building_blocks(path)
    assert Precursor(molecule).is_building_block(stock, min_mol_size=0)
    assert stock[key[:14]][0].sources == ({"vendor": "MP", "ppg": "5.0"},)
    exported = tmp_path / "legacy.json.gz"
    standardize_building_block_catalogue(path, exported)
    assert list(load_building_blocks(exported).records()) == list(stock.records())


@pytest.mark.parametrize(
    "source, url",
    [
        (
            {"vendor": "MC", "id": "MCULE-1234567890"},
            "https://mcule.com/MCULE-1234567890/",
        ),
        (
            {"vendor": "MP", "id": "Molport-001-789-854"},
            "https://www.molport.com/shop/compound/Molport-001-789-854",
        ),
    ],
)
def test_material_variants_keep_source_ids_and_do_not_sell_a_racemate_as_an_enantiomer(
    tmp_path,
    source,
    url,
):
    molecule = smiles("C[C@H](O)C(=O)O")
    key = inchi_key(molecule)
    base = {"smiles": str(molecule), "vendors": {}, "has_stereo": True}
    records = [
        dict(base, stereo_type=kind, sources=[dict(source, stereo_type=kind)])
        for kind in ("racemic", "relative", "unknown", "absolute")
    ]
    path = tmp_path / "stock.json"
    path.write_text(json.dumps({key: records}))
    stock = load_building_blocks(path)
    assert stock.record_count == 4
    assert len(stock[key[:14]]) == 4
    exported = tmp_path / "roundtrip.json.gz"
    standardize_building_block_catalogue(stock.path, exported)
    assert list(load_building_blocks(exported).records()) == list(stock.records())
    candidates = compatible_records(molecule, stock)
    assert len(candidates) == 1 and candidates[0].stereo_type == "absolute"
    precursor = Precursor(molecule)
    assert precursor.is_building_block(stock)
    selected = precursor.selected_stock
    assert selected["sources"] == records[-1]["sources"]
    assert selected["price"] is None
    with pytest.raises(TypeError):
        candidates[0].sources[0]["id"] = "changed"
    target = smiles("CCOC(=O)C(O)C")
    route = Route((Step(Reaction([precursor.molecule], [target]), target),))
    detached = Route.from_json(json.loads(json.dumps(route.to_json())))
    assert detached.leaves()[0].meta["selected_stock"] == selected
    stock.close()
    stock.path.unlink()
    html = routes_report_html([detached], None, prices=False)
    payloads = re.findall(r'<g class="sp-price" data-offers="([^"]*)">', html)
    payload = json.loads(unescape(payloads[0]))
    assert payload["rows"][0] == [
        source["vendor"],
        {"text": source["id"], "href": url},
    ]
    assert "source IDs" in html
    assert "Price per g of target" not in html


def test_unpriced_sources_merge_and_compressed_sqlite_opens_without_chemistry(
    tmp_path, monkeypatch
):
    sources = [
        {"vendor": "MC", "id": "MCULE-8453099153"},
        {"vendor": "MP", "id": "Molport-000-000-001"},
    ]
    raw = tmp_path / "sources.tsv"
    with raw.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("SMILES", "sources"))
        for source in sources:
            writer.writerow(
                (
                    "C1CCCC1",
                    json.dumps(
                        [
                            dict(
                                source,
                                url="https://example.org/?utm_source=test",
                                lead_time="10",
                                availability="8",
                            )
                        ]
                    ),
                )
            )
    stock = load_building_blocks(raw)
    (record,) = stock[inchi_key(smiles("C1CCCC1"))[:14]]
    assert selected_record(record)["sources"] == sources
    assert record.price is None
    path = tmp_path / "stock.sqlite.gz"
    path.write_bytes(gzip.compress(stock.path.read_bytes()))
    from synplan.chem.building_blocks import io

    monkeypatch.setattr(
        io,
        "_prepare_catalogue_batches",
        lambda *a, **kw: pytest.fail("Repeated chemistry"),
    )
    loaded = load_building_blocks(path)
    assert loaded == stock
    assert load_building_blocks(path) is loaded


def test_molport_legacy_cx_flag_keeps_source_id_without_promoting_stereo(tmp_path):
    raw_smiles = "Cl.N[C@H]1C[C@@H](C(O)=O)c2ccccc12 |r|"
    source = {"vendor": "MP", "id": "Molport-000-000-296", "smiles": raw_smiles}
    raw = tmp_path / "sources.tsv"
    with raw.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("SMILES", "sources"))
        writer.writerow((raw_smiles, json.dumps([source])))
    stock = load_building_blocks(raw)
    (record,) = next(iter(stock.values()))
    assert record.sources[0] == {"vendor": "MP", "id": "Molport-000-000-296"}
    assert record.stereo_type == "unknown"
    assert not compatible_records(smiles(raw_smiles.split(" |")[0]), stock)
    query = smiles("Cl.NC1CC(C(O)=O)c2ccccc12")
    assert Precursor(query).is_building_block(stock, min_mol_size=0)
    selected, metadata = match_stereo_stock(query, (), stock, 256)
    assert not has_stereo(
        selected
    )  # Unconfirmed wedges never become inherited geometry.
    assert metadata["sources"] == [{k: v for k, v in source.items() if k != "smiles"}]
    assert metadata["stereo_type"] == "unknown"


@pytest.mark.parametrize("source", [{"vendor": "MC"}, {"vendor": "MC", "id": 123}])
def test_invalid_source_metadata_rejects_prepared_file(tmp_path, source):
    molecule = smiles("CCO")
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                inchi_key(molecule): {
                    "smiles": str(molecule),
                    "has_stereo": False,
                    "vendors": {},
                    "sources": [source],
                }
            }
        )
    )
    with pytest.raises(ValueError, match="vendor and id"):
        build_catalogue(path, tmp_path / "bad.sqlite")
    assert not (tmp_path / "bad.sqlite").exists()
