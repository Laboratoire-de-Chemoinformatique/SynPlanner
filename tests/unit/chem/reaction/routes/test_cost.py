from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from chython import smiles

import synplan.chem.reaction.routes.postprocess.cost as cost_module
from synplan.chem.reaction.routes.postprocess.cost import (
    BuildingBlockCost,
    RouteCostError,
    RouteCostEstimate,
    estimate_route_cost,
    estimate_route_costs,
)
from synplan.chem.utils import safe_canonicalization


def _molecule_node(
    structure: str,
    *children: dict[str, object],
    in_stock: bool = False,
) -> dict[str, object]:
    node: dict[str, object] = {
        "type": "mol",
        "smiles": structure,
        "in_stock": in_stock,
    }
    if children:
        node["children"] = list(children)
    return node


def _reaction_node(*children: dict[str, object]) -> dict[str, object]:
    return {
        "type": "reaction",
        "smiles": "C.O>>CO",
        "children": list(children),
    }


def _catalogue(
    tmp_path: Path,
    header: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    name: str = "building_blocks.tsv",
) -> Path:
    path = tmp_path / name
    lines = ["\t".join(header), *("\t".join(row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _mass(structure: str) -> float:
    molecule = safe_canonicalization(smiles(structure), clean_stereo=False)
    return molecule.molecular_mass


def test_nested_route_groups_repeated_leaves_and_calculates_costs(tmp_path):
    route = _molecule_node(
        "CCO",
        _reaction_node(
            _molecule_node("C"),
            _molecule_node(
                "CO",
                _reaction_node(_molecule_node("C"), _molecule_node("O")),
            ),
        ),
    )
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg", "SA_ppg"),
        [("C", "2", "1.5"), ("O", "0", "3")],
    )

    estimate = estimate_route_cost(route, catalogue)

    carbon_mass = _mass("C")
    oxygen_mass = _mass("O")
    target_mass = _mass("CCO")
    expected_cost = 2 * carbon_mass * 1.5 + oxygen_mass * 3

    assert isinstance(estimate, RouteCostEstimate)
    assert estimate.complete is True
    assert estimate.target_smiles == "CCO"
    assert estimate.target_molecular_weight == pytest.approx(target_mass)
    assert estimate.cost_per_mol == pytest.approx(expected_cost)
    assert estimate.priced_cost_per_mol == pytest.approx(expected_cost)
    assert estimate.cost_per_gram == pytest.approx(expected_cost / target_mass)
    assert estimate.priced_cost_per_gram == pytest.approx(expected_cost / target_mass)
    assert estimate.price_columns == ("LN_ppg", "SA_ppg")
    assert estimate.cost_units == "raw_catalogue_units"
    assert estimate.missing_smiles == ()
    assert estimate.unpriced_smiles == ()

    assert len(estimate.building_blocks) == 2
    carbon, oxygen = estimate.building_blocks
    assert isinstance(carbon, BuildingBlockCost)
    assert carbon.smiles == "C"
    assert carbon.equivalents == 2
    assert carbon.molecular_weight == pytest.approx(carbon_mass)
    assert carbon.vendor == "SA"
    assert carbon.price_column == "SA_ppg"
    assert carbon.price_per_gram == pytest.approx(1.5)
    assert carbon.cost_per_mol == pytest.approx(2 * carbon_mass * 1.5)
    assert carbon.cost_per_gram == pytest.approx(2 * carbon_mass * 1.5 / target_mass)
    assert carbon.status == "priced"
    assert oxygen.smiles == "O"
    assert oxygen.equivalents == 1
    assert oxygen.price_column == "SA_ppg"
    assert oxygen.status == "priced"


def test_minimum_price_ties_follow_tsv_header_order(tmp_path):
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "ZZ_ppg", "AA_ppg", "BB_ppg"),
        [("N", "2", "1", "1")],
    )

    item = estimate_route_cost(_molecule_node("N"), catalogue).building_blocks[0]

    assert item.vendor == "AA"
    assert item.price_column == "AA_ppg"
    assert item.price_per_gram == pytest.approx(1)


def test_blank_and_zero_prices_make_a_catalogue_match_unpriced(tmp_path):
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg", "SA_ppg"),
        [("Cl", "0", "")],
    )

    estimate = estimate_route_cost(_molecule_node("Cl"), catalogue)

    assert estimate.complete is False
    assert estimate.cost_per_mol is None
    assert estimate.cost_per_gram is None
    assert estimate.priced_cost_per_mol == pytest.approx(0)
    assert estimate.priced_cost_per_gram == pytest.approx(0)
    assert estimate.missing_smiles == ()
    assert estimate.unpriced_smiles == ("Cl",)
    item = estimate.building_blocks[0]
    assert item.status == "unpriced"
    assert item.vendor is None
    assert item.price_column is None
    assert item.price_per_gram is None
    assert item.cost_per_mol is None
    assert item.cost_per_gram is None


def test_missing_leaf_is_not_rescued_by_in_stock_flag(tmp_path):
    route = _molecule_node(
        "CN",
        _reaction_node(_molecule_node("C"), _molecule_node("N", in_stock=True)),
    )
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg"),
        [("C", "2")],
    )

    estimate = estimate_route_cost(route, catalogue)

    assert estimate.complete is False
    assert estimate.cost_per_mol is None
    assert estimate.cost_per_gram is None
    assert estimate.priced_cost_per_mol == pytest.approx(_mass("C") * 2)
    assert estimate.priced_cost_per_gram == pytest.approx(_mass("C") * 2 / _mass("CN"))
    assert estimate.missing_smiles == ("N",)
    assert estimate.unpriced_smiles == ()
    assert [item.status for item in estimate.building_blocks] == [
        "priced",
        "missing",
    ]


@pytest.mark.parametrize("with_empty_children", [False, True])
def test_root_only_route_prices_the_purchased_target(tmp_path, with_empty_children):
    route = _molecule_node("CCO")
    if with_empty_children:
        route["children"] = []
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg"),
        [("CCO", "7.5")],
    )

    estimate = estimate_route_cost(route, catalogue)

    assert estimate.complete is True
    assert len(estimate.building_blocks) == 1
    assert estimate.building_blocks[0].equivalents == 1
    assert estimate.cost_per_mol == pytest.approx(_mass("CCO") * 7.5)
    assert estimate.cost_per_gram == pytest.approx(7.5)


def test_duplicate_catalogue_rows_use_the_lowest_positive_price(tmp_path):
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg", "SA_ppg"),
        [("CCO", "8", "0"), ("CCO", "5", "6")],
    )

    item = estimate_route_cost(_molecule_node("CCO"), catalogue).building_blocks[0]

    assert item.vendor == "LN"
    assert item.price_column == "LN_ppg"
    assert item.price_per_gram == pytest.approx(5)


def test_canonical_leaf_alias_matches_prepared_catalogue_smiles(tmp_path):
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg"),
        [("CCO", "4")],
    )

    estimate = estimate_route_cost(_molecule_node("C(C)O"), catalogue)

    assert estimate.complete is True
    assert estimate.target_smiles == "CCO"
    assert estimate.building_blocks[0].smiles == "CCO"


@pytest.mark.parametrize(
    "structure,canonical",
    [
        ("C[C@H](O)Cl", "O[C@H](Cl)C"),
        ("CC.[Na+]", "CC.[Na+]"),
    ],
)
def test_stereo_and_complete_salts_are_preserved_in_cost_identity(
    tmp_path, structure, canonical
):
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg"),
        [(structure, "2")],
    )

    estimate = estimate_route_cost(_molecule_node(structure), catalogue)

    assert estimate.complete is True
    assert estimate.target_smiles == canonical
    assert estimate.target_molecular_weight == pytest.approx(_mass(structure))
    assert estimate.building_blocks[0].smiles == canonical
    assert estimate.building_blocks[0].molecular_weight == pytest.approx(
        _mass(structure)
    )


@pytest.mark.parametrize(
    "route",
    [
        {},
        {"type": "reaction", "smiles": "C>>C", "children": []},
        {"type": "mol", "children": []},
        {
            "type": "mol",
            "smiles": "C",
            "children": [
                {
                    "type": "reaction",
                    "smiles": "C>>C",
                    "children": [{"type": "reaction", "children": []}],
                }
            ],
        },
        {"type": "mol", "smiles": "C", "children": "not-a-list"},
    ],
)
def test_malformed_route_tree_is_rejected(tmp_path, route):
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg"),
        [("C", "1")],
    )

    with pytest.raises(RouteCostError):
        estimate_route_cost(route, catalogue)


@pytest.mark.parametrize(
    "route",
    [
        _molecule_node("not a molecule"),
        _molecule_node(
            "C",
            _reaction_node(_molecule_node("not a molecule")),
        ),
    ],
)
def test_invalid_target_or_leaf_smiles_is_rejected(tmp_path, route):
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg"),
        [("C", "1")],
    )

    with pytest.raises(RouteCostError):
        estimate_route_cost(route, catalogue)


@pytest.mark.parametrize(
    "header,rows",
    [
        (("name", "LN_ppg"), [("methane", "1")]),
        (("SMILES", "smiles", "LN_ppg"), [("C", "C", "1")]),
        (("SMILES", "vendor"), [("C", "LN")]),
        (("SMILES", "LN_ppg"), [("C", "1", "extra")]),
    ],
)
def test_invalid_catalogue_schema_or_row_shape_is_rejected(tmp_path, header, rows):
    catalogue = _catalogue(tmp_path, header, rows)

    with pytest.raises(RouteCostError):
        estimate_route_cost(_molecule_node("C"), catalogue)


@pytest.mark.parametrize("invalid_price", ["-1", "not-a-price", "nan", "inf"])
def test_invalid_price_has_line_and_column_diagnostics(tmp_path, invalid_price):
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg"),
        [("C", "1"), ("N", invalid_price)],
    )

    with pytest.raises(RouteCostError) as captured:
        estimate_route_cost(_molecule_node("C"), catalogue)

    message = str(captured.value)
    assert "LN_ppg" in message
    assert "3" in message


def test_empty_catalogue_is_rejected(tmp_path):
    catalogue = tmp_path / "building_blocks.tsv"
    catalogue.write_text("", encoding="utf-8")

    with pytest.raises(RouteCostError):
        estimate_route_cost(_molecule_node("C"), catalogue)


def test_results_are_frozen_value_objects(tmp_path):
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg"),
        [("C", "1")],
    )
    estimate = estimate_route_cost(_molecule_node("C"), catalogue)

    with pytest.raises(FrozenInstanceError):
        estimate.complete = False
    with pytest.raises(FrozenInstanceError):
        estimate.building_blocks[0].equivalents = 2


def test_route_pool_scans_catalogue_once_and_preserves_route_order(
    tmp_path, monkeypatch
):
    routes = {
        "priced": _molecule_node("C"),
        "partly-unpriced": _molecule_node(
            "CN",
            _reaction_node(_molecule_node("C"), _molecule_node("N")),
        ),
        "missing": _molecule_node("O"),
    }
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg"),
        [("C", "2"), ("N", "0")],
    )
    scans = 0
    original = cost_module.iter_chemical_records

    def counted_records(*args, **kwargs):
        nonlocal scans
        scans += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cost_module, "iter_chemical_records", counted_records)

    estimates = estimate_route_costs(routes, catalogue)

    assert list(estimates) == list(routes)
    assert scans == 1
    assert estimates["priced"].complete is True
    assert estimates["partly-unpriced"].complete is False
    assert estimates["partly-unpriced"].unpriced_smiles == ("N",)
    assert estimates["missing"].complete is False
    assert estimates["missing"].missing_smiles == ("O",)


def test_batch_and_single_route_results_are_equal(tmp_path):
    route = _molecule_node(
        "CO",
        _reaction_node(_molecule_node("C"), _molecule_node("O")),
    )
    catalogue = _catalogue(
        tmp_path,
        ("SMILES", "LN_ppg"),
        [("C", "1"), ("O", "2")],
    )

    assert (
        estimate_route_cost(route, catalogue)
        == estimate_route_costs({"route": route}, catalogue)["route"]
    )


def test_empty_route_pool_does_not_open_catalogue(tmp_path):
    missing_catalogue = tmp_path / "does-not-exist.tsv"

    assert estimate_route_costs({}, missing_catalogue) == {}


def test_cost_api_is_available_from_lazy_routes_facade():
    from synplan.chem.reaction import routes

    assert routes.BuildingBlockCost is BuildingBlockCost
    assert routes.RouteCostError is RouteCostError
    assert routes.RouteCostEstimate is RouteCostEstimate
    assert routes.estimate_route_cost is estimate_route_cost
    assert routes.estimate_route_costs is estimate_route_costs
    assert "BuildingBlockCost" in routes.__all__
    assert "RouteCostError" in routes.__all__
    assert "RouteCostEstimate" in routes.__all__
    assert "estimate_route_cost" in routes.__all__
    assert "estimate_route_costs" in routes.__all__
