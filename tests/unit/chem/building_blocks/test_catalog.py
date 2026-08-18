from __future__ import annotations

import re

import pytest
from chython import smiles

from synplan.chem.building_blocks import BuildingBlockCatalog
from synplan.chem.building_blocks.stock import coerce_building_block_stock
from synplan.chem.reaction.routes.postprocess import estimate_route_cost
from synplan.chem.utils import safe_canonicalization


def _files(tmp_path):
    identity = tmp_path / "identity.tsv"
    identity.write_text(
        "source_index\tinput_smiles\tcanonical_smiles\tstandard_inchikey\t"
        "output_origin\tstatus\n"
        "1\tCCOC(=O)N\tN\tQGZKDVFQNNGYKY-UHFFFAOYSA-N\tdeprotected\twritten\n"
    )
    prices = tmp_path / "prices.tsv"
    prices.write_text("source_index\tinput_smiles\tVendor_ppg\n1\tCCOC(=O)N\t2.5\n")
    return identity, prices


def test_catalog_joins_provenance_prices_and_builds_stock(tmp_path):
    identity, prices = _files(tmp_path)

    catalog = BuildingBlockCatalog.from_files(identity, prices)
    stock = catalog.stock()
    provenance = catalog.provenance_for_molecule(smiles("N"))

    assert len(catalog.records) == 1
    assert catalog.prices_by_source[1] == {"Vendor_ppg": 2.5}
    assert stock.contains_molecule(smiles("N"))
    assert catalog.contains_key("N")
    assert catalog.contains_molecule(smiles("N"))
    assert coerce_building_block_stock(catalog) is catalog
    assert provenance[0]["source_index"] == 1
    assert provenance[0]["input_smiles"] == "CCOC(=O)N"
    assert provenance[0]["output_origin"] == "deprotected"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("1\tCCOC(=O)N\t2.5\n1\tCCOC(=O)N\t3\n", "duplicate source_index 1"),
        ("2\tCCOC(=O)N\t2.5\n", "missing source_index values [1]"),
        ("1\tCCN\t2.5\n", "does not match identity input_smiles"),
    ],
)
def test_catalog_rejects_invalid_explicit_price_join(tmp_path, rows, message):
    identity, prices = _files(tmp_path)
    prices.write_text(
        "source_index\tinput_smiles\tVendor_ppg\n" + rows,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        BuildingBlockCatalog.from_files(identity, prices)


def test_catalog_requires_self_describing_price_keys(tmp_path):
    identity, prices = _files(tmp_path)
    prices.write_text("SMILES\tVendor_ppg\nCCOC(=O)N\t2.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing price columns"):
        BuildingBlockCatalog.from_files(identity, prices)


def test_catalog_costs_restored_input_without_rescanning_price_file(tmp_path):
    identity, prices = _files(tmp_path)
    catalog = BuildingBlockCatalog.from_files(identity, prices)
    prices.unlink()
    route = {"type": "mol", "smiles": "CCOC(=O)N", "in_stock": True}

    estimate = estimate_route_cost(route, catalog)

    assert estimate.complete is True
    assert estimate.building_blocks[0].price_per_gram == 2.5


def test_catalog_rejects_partial_exact_deprotection_provenance(tmp_path):
    identity = tmp_path / "identity.tsv"
    identity.write_text(
        "source_index\tinput_smiles\tcanonical_smiles\tstandard_inchikey\t"
        "output_origin\tstatus\tstandardized_input_smiles\n"
        "1\tCCOC(=O)N\tN\tQGZKDVFQNNGYKY-UHFFFAOYSA-N\t"
        "deprotected\twritten\tCCOC(=O)N\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="incomplete exact deprotection provenance",
    ):
        BuildingBlockCatalog.from_files(identity)


def test_catalog_construction_does_not_parse_every_canonical_smiles(
    tmp_path, monkeypatch
):
    import synplan.chem.building_blocks.catalog as catalog_module

    identity, prices = _files(tmp_path)

    def unexpected_parse(_value):
        raise AssertionError("catalog construction must not parse canonical SMILES")

    monkeypatch.setattr(catalog_module, "smiles_parser", unexpected_parse)

    catalog = BuildingBlockCatalog.from_files(identity, prices)

    assert catalog.contains_key("N")


def test_catalog_discovers_and_lazily_uses_precomputed_stereo_file(
    tmp_path, monkeypatch
):
    import synplan.chem.building_blocks.catalog as catalog_module

    stored = smiles("N[C@H](C)C(=O)Cl")
    query = smiles("N[C@@H](C)C(=O)Cl")
    stereo_free = stored.copy()
    stereo_free.clean_stereo()
    stored_smiles = str(safe_canonicalization(stored, clean_stereo=False))
    stereo_free_smiles = str(safe_canonicalization(stereo_free, clean_stereo=False))
    identity = tmp_path / "blocks_identity.tsv"
    identity.write_text(
        "source_index\tinput_smiles\tcanonical_smiles\tstandard_inchikey\t"
        "output_origin\tstatus\n"
        f"1\t{stored_smiles}\t{stored_smiles}\t\tprotected\twritten\n",
        encoding="utf-8",
    )
    stereo = tmp_path / "blocks_stereo.tsv"
    stereo.write_text(
        "source_index\tinput_smiles\tcanonical_smiles\tstereo_free_smiles\t"
        "stereo_present\toutput_origin\tstatus\tnote\n"
        f"1\t{stored_smiles}\t{stored_smiles}\t{stereo_free_smiles}\t"
        "True\tprotected\twritten\t\n",
        encoding="utf-8",
    )
    catalog = BuildingBlockCatalog.from_files(identity)

    def unexpected_parse(_value):
        raise AssertionError("precomputed stereo lookup must not parse catalog rows")

    monkeypatch.setattr(catalog_module, "smiles_parser", unexpected_parse)

    exact, candidates = catalog.validate_stereo_for_molecule(query)

    assert catalog.stereo_file == stereo
    assert exact is False
    assert candidates == (stored_smiles,)


def test_exact_stereo_match_does_not_read_discovered_sidecar(tmp_path):
    molecule = smiles("N[C@H](C)C(=O)Cl")
    canonical = str(safe_canonicalization(molecule, clean_stereo=False))
    identity = tmp_path / "blocks_identity.tsv"
    identity.write_text(
        "source_index\tinput_smiles\tcanonical_smiles\tstandard_inchikey\t"
        "output_origin\tstatus\n"
        f"1\t{canonical}\t{canonical}\t\tprotected\twritten\n",
        encoding="utf-8",
    )
    stereo = tmp_path / "blocks_stereo.tsv"
    stereo.write_text(
        "source_index\tcanonical_smiles\tstereo_free_smiles\toutput_origin\tstatus\n",
        encoding="utf-8",
    )
    catalog = BuildingBlockCatalog.from_files(identity)
    stereo.unlink()

    exact, candidates = catalog.validate_stereo_for_molecule(molecule)

    assert exact is True
    assert candidates == (canonical,)
