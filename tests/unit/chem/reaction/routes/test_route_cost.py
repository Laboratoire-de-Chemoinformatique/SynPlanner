import json
import pickle

import pytest
from chython import smiles
from frozendict import frozendict

from synplan.chem.building_blocks import BuildingBlock, molecule_to_inchikey
from synplan.chem.reaction.reactor import Reaction
from synplan.chem.reaction.routes.route import Route, Step

R_LACTIC = "C[C@H](O)C(=O)O"
S_LACTIC = "C[C@@H](O)C(=O)O"


def _block(smiles_value: str, **vendors: float) -> BuildingBlock:
    molecule = smiles(smiles_value, ignore_stereo=False)
    return BuildingBlock(
        smiles=str(molecule),
        inchikey=molecule_to_inchikey(molecule),
        vendors=frozendict(vendors),
        has_stereo=any(atom.stereo is not None for _, atom in molecule.atoms()),
    )


def _index(*blocks: BuildingBlock):
    groups = {}
    for block in blocks:
        groups.setdefault(block.inchikey[:14], []).append(block)
    return frozendict((key, tuple(value)) for key, value in groups.items())


def _route(target_smiles: str, *leaf_smiles: str) -> Route:
    target = smiles(target_smiles, ignore_stereo=False)
    leaves = [smiles(value, ignore_stereo=False) for value in leaf_smiles]
    reaction = Reaction(leaves, [target])
    return Route((Step(reaction, target),))


@pytest.mark.parametrize("target_smiles", ["C[C@H](F)Cl", "CC(F)Cl"])
def test_costing_is_connectivity_only_and_selects_the_cheapest_offer(target_smiles):
    r_block = _block(R_LACTIC, expensive=5.0)
    s_block = _block(S_LACTIC, inexpensive=1.0)
    route = _route(target_smiles, R_LACTIC)

    result = route.calculate_cost(_index(r_block, s_block))

    assert result["complete"]
    assert result["leaves"][0]["selected_inchikey"] == s_block.inchikey
    assert result["leaves"][0]["vendor"] == "inexpensive"
    assert result["cost_per_mol"] == pytest.approx(
        route.leaves()[0].molecular_mass * 1.0
    )


def test_repeated_leaves_are_grouped_as_equivalents():
    block = _block(R_LACTIC, vendor=2.0)
    route = _route("CCO", R_LACTIC, R_LACTIC)

    result = route.calculate_cost(_index(block))

    assert len(result["leaves"]) == 1
    assert result["leaves"][0]["equivalents"] == 2
    assert result["cost_per_mol"] == pytest.approx(
        2 * route.leaves()[0].molecular_mass * 2.0
    )


def test_missing_and_unpriced_leaves_make_the_total_incomplete():
    unpriced = _block(R_LACTIC)
    route = _route("CCO", R_LACTIC, "CCCCCCC")

    result = route.calculate_cost(_index(unpriced))

    assert not result["complete"]
    assert result["cost_per_mol"] is None
    assert result["cost_per_gram"] is None
    assert result["priced_cost_per_mol"] == 0.0
    assert result["missing_leaves"] == [str(smiles("CCCCCCC"))]
    assert result["unpriced_leaves"] == [str(smiles(R_LACTIC))]
    assert {row["status"] for row in result["leaves"]} == {"missing", "unpriced"}


def test_costing_does_not_attach_state_and_result_is_json_compatible():
    block = _block(R_LACTIC, vendor=2.0)
    route = _route("CCO", R_LACTIC)
    before = pickle.dumps(route)

    result = route.calculate_cost(_index(block))

    assert pickle.dumps(route) == before
    assert not hasattr(route, "cost")
    assert pickle.loads(before) == route
    json.dumps(result)
