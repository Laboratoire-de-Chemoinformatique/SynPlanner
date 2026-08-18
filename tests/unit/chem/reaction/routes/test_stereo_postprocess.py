from copy import deepcopy

import pytest
from chython import smiles
from chython.containers import MoleculeContainer, ReactionContainer

from synplan.chem.building_blocks.catalog import BuildingBlockCatalog
from synplan.chem.building_blocks.reports import IdentityReportRow
from synplan.chem.reaction.routes import restore_route_stereo
from synplan.chem.reaction.routes.postprocess import RouteStereoError


def _route():
    return {
        "type": "mol",
        "smiles": "[NH2:1][CH:2]([CH3:3])[C:4](=[O:5])[OH:6]",
        "children": [
            {
                "type": "reaction",
                "smiles": (
                    "[NH2:1][CH:2]([CH3:3])[C:4](=[O:5])[Cl:6]>>"
                    "[NH2:1][CH:2]([CH3:3])[C:4](=[O:5])[OH:6]"
                ),
                "meta": {"rule_id": 7},
                "children": [
                    {
                        "type": "mol",
                        "smiles": "[NH2:1][CH:2]([CH3:3])[C:4](=[O:5])[Cl:6]",
                        "in_stock": True,
                    }
                ],
            }
        ],
    }


def _has_atom_stereo(value: str) -> bool:
    parsed = smiles(value)
    molecules = (
        (*parsed.reactants, *parsed.products)
        if isinstance(parsed, ReactionContainer)
        else (parsed,)
    )
    return any(
        atom.stereo is not None
        for molecule in molecules
        if isinstance(molecule, MoleculeContainer)
        for _number, atom in molecule.atoms()
    )


def test_restores_target_and_propagates_stereo_to_reaction_and_bb():
    route = _route()
    original = deepcopy(route)

    restored = restore_route_stereo(route, "N[C@@H](C)C(=O)O")

    reaction = restored["children"][0]
    leaf = reaction["children"][0]
    assert _has_atom_stereo(restored["smiles"])
    assert _has_atom_stereo(reaction["smiles"])
    assert _has_atom_stereo(leaf["smiles"])
    assert reaction["meta"] == {"rule_id": 7}
    assert leaf["in_stock"] is True
    assert route == original


def test_target_stereo_replaces_conflicting_route_label():
    route = _route()
    route["smiles"] = "N[C@H](C)C(=O)O"

    restored = restore_route_stereo(route, "N[C@@H](C)C(=O)O")

    assert smiles(restored["smiles"]) == smiles("N[C@@H](C)C(=O)O")


def test_rejects_target_with_different_connectivity():
    with pytest.raises(RouteStereoError, match="not isomorphic"):
        restore_route_stereo(_route(), "NCC(=O)O")


def test_rejects_reaction_child_count_mismatch():
    route = _route()
    route["children"][0]["children"] = []

    with pytest.raises(RouteStereoError, match="1 reactants but 0 molecule children"):
        restore_route_stereo(route, "N[C@@H](C)C(=O)O")


def test_propagates_cis_trans_stereo_to_reaction_and_bb():
    route = {
        "type": "mol",
        "smiles": "[F:1][CH:2]=[CH:3][CH2:4][CH2:5][OH:6]",
        "children": [
            {
                "type": "reaction",
                "smiles": (
                    "[F:1][CH:2]=[CH:3][CH2:4][CH2:5][Cl:6]>>"
                    "[F:1][CH:2]=[CH:3][CH2:4][CH2:5][OH:6]"
                ),
                "children": [
                    {
                        "type": "mol",
                        "smiles": "[F:1][CH:2]=[CH:3][CH2:4][CH2:5][Cl:6]",
                    }
                ],
            }
        ],
    }

    restored = restore_route_stereo(route, "F/C=C/CCO")

    reaction = restored["children"][0]
    leaf = reaction["children"][0]
    assert "/" in restored["smiles"] or "\\" in restored["smiles"]
    assert "/" in reaction["smiles"] or "\\" in reaction["smiles"]
    assert "/" in leaf["smiles"] or "\\" in leaf["smiles"]


def _catalog(smiles_value: str) -> BuildingBlockCatalog:
    canonical = str(smiles(smiles_value))
    return BuildingBlockCatalog(
        (
            IdentityReportRow(
                source_index=1,
                input_smiles=canonical,
                canonical_smiles=canonical,
                standard_inchi="",
                standard_inchikey="",
                inchi_return_code="",
                inchi_warnings="",
                output_origin="protected",
                status="written",
            ),
        )
    )


def test_catalog_marks_matching_stereo_on_building_block_leaf():
    restored = restore_route_stereo(
        _route(),
        "N[C@@H](C)C(=O)O",
        catalog=_catalog("N[C@@H](C)C(=O)Cl"),
    )

    leaf = restored["children"][0]["children"][0]
    assert restored["stereo_mismatch"] is False
    assert leaf["bb"]["stereo_mismatch"] is False
    assert leaf["bb"]["stereo_validation"]["status"] == "matched"


def test_catalog_flags_opposite_building_block_stereo():
    catalog_smiles = str(smiles("N[C@H](C)C(=O)Cl"))

    restored = restore_route_stereo(
        _route(),
        "N[C@@H](C)C(=O)O",
        catalog=_catalog(catalog_smiles),
    )

    leaf = restored["children"][0]["children"][0]
    validation = leaf["bb"]["stereo_validation"]
    assert restored["stereo_mismatch"] is True
    assert leaf["bb"]["stereo_mismatch"] is True
    assert validation["status"] == "mismatch"
    assert validation["catalog_smiles"] == [catalog_smiles]


def test_catalog_flags_leaf_missing_from_library():
    restored = restore_route_stereo(
        _route(), "N[C@@H](C)C(=O)O", catalog=_catalog("CCO")
    )

    leaf = restored["children"][0]["children"][0]
    assert restored["stereo_mismatch"] is True
    assert leaf["bb"]["stereo_validation"]["status"] == "not_found"
