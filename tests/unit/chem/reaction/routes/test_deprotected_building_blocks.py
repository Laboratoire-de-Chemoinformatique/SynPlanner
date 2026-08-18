"""Exact and legacy provenance behavior for protected-BB route expansion."""

import csv
from pathlib import Path

from chython import smiles
from chython.containers import ReactionContainer

from synplan.chem.building_blocks import BuildingBlockCatalog
from synplan.chem.building_blocks.config import BuildingBlockPreparationConfig
from synplan.chem.building_blocks.preparation import prepare_building_blocks
from synplan.chem.building_blocks.reports import IdentityReportRow
from synplan.chem.reaction.routes.postprocess import (
    expand_deprotected_building_blocks,
)


def _prepared_conservative_record(tmp_path):
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
    record = next(row for row in rows if row["output_origin"] == "deprotected")
    return BuildingBlockCatalog.from_files(result.identity_reference_file), record


def test_exact_replay_does_not_rerun_current_taxonomy(monkeypatch, tmp_path) -> None:
    import synplan.chem.reaction.routes.postprocess.deprotected_building_blocks as module

    catalog, record = _prepared_conservative_record(tmp_path)
    route = {
        "type": "mol",
        "smiles": record["canonical_smiles"],
        "in_stock": True,
        "bb": {"records": [record]},
    }

    def reject_inference(*_args, **_kwargs):
        raise AssertionError("exact provenance must not rerun deprotection")

    monkeypatch.setattr(module, "deprotect_molecule", reject_inference)
    restored = expand_deprotected_building_blocks(route, catalog)[0]
    reaction_node = restored["children"][0]
    reaction = smiles(reaction_node["smiles"])

    assert isinstance(reaction, ReactionContainer)
    assert reaction_node["meta"]["preprocessing_provenance"] == "exact"
    assert reaction_node["meta"]["deprotection_policy"] == "conservative"
    assert reaction_node["children"][0]["smiles"] == record["input_smiles"]
    product_maps = set(reaction.products[0].atoms_numbers)
    introduced_maps = set(reaction.reactants[0].atoms_numbers) - product_maps
    assert introduced_maps
    assert min(introduced_maps) > max(product_maps)


def test_legacy_identity_rows_keep_marked_inference_fallback() -> None:
    catalog = BuildingBlockCatalog(
        (
            IdentityReportRow(
                source_index=1,
                input_smiles="c1ccccc1NC(=O)OC(C)(C)C",
                canonical_smiles="c1cc(ccc1)N",
                standard_inchi="",
                standard_inchikey="",
                inchi_return_code="",
                inchi_warnings="",
                output_origin="deprotected",
                status="written",
            ),
        )
    )
    route = {"type": "mol", "smiles": "c1ccccc1N", "in_stock": True}

    restored = expand_deprotected_building_blocks(route, catalog)[0]

    assert (
        restored["children"][0]["meta"]["preprocessing_provenance"]
        == "legacy_inference"
    )
