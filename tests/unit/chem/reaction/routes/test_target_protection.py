from copy import deepcopy

import pytest
from chython import smiles
from chython.containers import MoleculeContainer, ReactionContainer

from synplan.chem.building_blocks import (
    BuildingBlockCatalog,
    deprotect_molecule_with_provenance,
)
from synplan.chem.building_blocks.deprotection import (
    DeprotectionSequenceLimitError,
    deprotect_molecule,
)
from synplan.chem.reaction.routes import restore_protected_target
from synplan.chem.reaction.routes.postprocess import (
    RoutePostprocessConfig,
    TargetProtectionError,
    postprocess_routes,
)
from synplan.chem.reaction.routes.tree_ops import (
    iter_molecule_leaves,
    iter_reactions_postorder,
    max_route_atom_map,
)

_TWO_GROUP_TARGET = "CC(C)(C)OC(=O)NCCO[Si](C)(C)C"
_THREE_GROUP_TARGET = "CC(C)(C)OC(=O)NCC(O[Si](C)(C)C)C1OCCO1"


def _mapped_identity_route(
    root_smiles: str,
    *,
    first_atom_map: int = 11,
) -> tuple[dict[str, object], set[int]]:
    molecule = smiles(root_smiles)
    assert isinstance(molecule, MoleculeContainer)
    remapping = {
        atom_number: first_atom_map + index
        for index, atom_number in enumerate(molecule)
    }
    molecule.remap(remapping)
    reaction = ReactionContainer(
        reactants=[molecule.copy()],
        products=[molecule.copy()],
    )
    route = {
        "type": "mol",
        "smiles": str(molecule),
        "in_stock": False,
        "children": [
            {
                "type": "reaction",
                "smiles": format(reaction, "m"),
                "children": [
                    {
                        "type": "mol",
                        "smiles": str(molecule),
                        "in_stock": True,
                    }
                ],
            }
        ],
    }
    return route, set(remapping.values())


def _protection_reactions(route):
    return [
        reaction
        for reaction in iter_reactions_postorder(route)
        if (reaction.get("meta") or {}).get("reaction_class") == "protection"
    ]


def test_default_restoration_returns_every_unique_protection_order() -> None:
    endpoint = deprotect_molecule(smiles(_TWO_GROUP_TARGET))
    route, original_atom_maps = _mapped_identity_route(str(endpoint))
    original = deepcopy(route)

    variants = restore_protected_target(route, _TWO_GROUP_TARGET)

    assert len(variants) == 2
    assert {
        tuple(variant["target_protection_rule_sequence"]) for variant in variants
    } == {
        ("amine_boc", "hydroxyl_tms"),
        ("hydroxyl_tms", "amine_boc"),
    }
    for variant_index, variant in enumerate(variants):
        assert str(smiles(variant["smiles"])) == str(smiles(_TWO_GROUP_TARGET))
        assert variant["target_protection_sequence_mode"] == "enumerate"
        assert variant["target_protection_variant_index"] == variant_index
        assert variant["target_protection_steps"] == 2

        protection_reactions = _protection_reactions(variant)
        assert [
            reaction["meta"]["protection_order"] for reaction in protection_reactions
        ] == [1, 2]
        assert [
            reaction["meta"]["protective_rule"] for reaction in protection_reactions
        ] == variant["target_protection_rule_sequence"]
        assert all(
            reaction["meta"]["bookkeeping"] is True for reaction in protection_reactions
        )

        final_reaction = smiles(variant["children"][0]["smiles"])
        assert isinstance(final_reaction, ReactionContainer)
        final_atom_maps = set(final_reaction.products[0].atoms_numbers)
        introduced_atom_maps = final_atom_maps - original_atom_maps
        assert introduced_atom_maps
        assert min(introduced_atom_maps) > max_route_atom_map(route)
        assert final_atom_maps & original_atom_maps == original_atom_maps
    assert route == original


def test_three_groups_return_six_restored_routes() -> None:
    endpoint = deprotect_molecule(smiles(_THREE_GROUP_TARGET))
    route, _ = _mapped_identity_route(str(endpoint))

    variants = restore_protected_target(route, _THREE_GROUP_TARGET)

    assert len(variants) == 6
    assert (
        len({tuple(variant["target_protection_rule_sequence"]) for variant in variants})
        == 6
    )
    assert all(variant["target_protection_steps"] == 3 for variant in variants)


def test_deterministic_mode_returns_one_stable_route() -> None:
    endpoint = deprotect_molecule(smiles(_TWO_GROUP_TARGET))
    route, _ = _mapped_identity_route(str(endpoint))

    first = restore_protected_target(
        route,
        _TWO_GROUP_TARGET,
        sequence_mode="deterministic",
    )
    second = restore_protected_target(
        route,
        _TWO_GROUP_TARGET,
        sequence_mode="deterministic",
    )

    assert len(first) == len(second) == 1
    assert first == second
    assert first[0]["target_protection_sequence_mode"] == "deterministic"
    assert first[0]["target_protection_variant_index"] == 0


def test_no_protecting_groups_returns_one_unchanged_copy() -> None:
    route, _ = _mapped_identity_route("NCCO")

    variants = restore_protected_target(route, "NCCO")

    assert variants == [route]
    assert variants[0] is not route


def test_target_sequence_limit_is_enforced_before_route_materialization() -> None:
    endpoint = deprotect_molecule(smiles(_TWO_GROUP_TARGET))
    route, _ = _mapped_identity_route(str(endpoint))

    with pytest.raises(
        DeprotectionSequenceLimitError,
        match="configured limit of 1 variants",
    ):
        restore_protected_target(route, _TWO_GROUP_TARGET, max_variants=1)


def test_target_restoration_rejects_an_incompatible_route_root() -> None:
    route, _ = _mapped_identity_route("CC")

    with pytest.raises(TargetProtectionError, match="route root"):
        restore_protected_target(route, _TWO_GROUP_TARGET)


def test_pipeline_auto_detects_stereo_only_target_processing() -> None:
    target = "N[C@@H](C)C(=O)O"
    planning_target = smiles(target)
    assert isinstance(planning_target, MoleculeContainer)
    planning_target.clean_stereo()
    route, _ = _mapped_identity_route(str(planning_target))

    result = postprocess_routes(
        {"route": route},
        BuildingBlockCatalog(()),
        target_smiles=target,
        config=RoutePostprocessConfig(
            expand_deprotected=False,
            calculate_cost=False,
        ),
    )

    assert result.ok
    assert len(result.variants) == 1
    restored = result.variants[0].route
    assert restored["target_postprocessing_scenario"] == "stereo"
    assert not _protection_reactions(restored)
    root = smiles(restored["smiles"])
    assert isinstance(root, MoleculeContainer)
    assert any(atom.stereo is not None for _, atom in root.atoms())


def test_pipeline_auto_detects_protection_only_target_processing() -> None:
    protected_target = "CC(C)(C)OC(=O)N[C@@H](C)CO[Si](C)(C)C"
    endpoint = deprotect_molecule(smiles(protected_target))
    route, _ = _mapped_identity_route(str(endpoint))

    result = postprocess_routes(
        {"route": route},
        BuildingBlockCatalog(()),
        target_smiles=protected_target,
        config=RoutePostprocessConfig(
            expand_deprotected=False,
            calculate_cost=False,
        ),
    )

    assert result.ok
    assert len(result.variants) == 2
    for item in result.variants:
        assert item.route["target_postprocessing_scenario"] == "protection"
        assert len(_protection_reactions(item.route)) == 2
        root = smiles(item.route["smiles"])
        assert isinstance(root, MoleculeContainer)
        assert any(atom.stereo is not None for _, atom in root.atoms())


def test_pipeline_propagates_protected_target_stereo_after_all_expansion() -> None:
    protected_target = "CC(C)(C)OC(=O)N[C@@H](C)CO[Si](C)(C)C"
    endpoint = deprotect_molecule(smiles(protected_target))
    endpoint.clean_stereo()
    route, _ = _mapped_identity_route(str(endpoint))

    result = postprocess_routes(
        {"route": route},
        BuildingBlockCatalog(()),
        target_smiles=protected_target,
        config=RoutePostprocessConfig(calculate_cost=False),
    )

    assert len(result.variants) == 2
    assert result.ok
    for item in result.variants:
        assert item.route["target_postprocessing_scenario"] == "stereo_and_protection"
        root = smiles(item.route["smiles"])
        assert isinstance(root, MoleculeContainer)
        assert any(atom.stereo is not None for _, atom in root.atoms())
        leaves = list(iter_molecule_leaves(item.route))
        assert len(leaves) == 1
        leaf = smiles(leaves[0][1]["smiles"])
        assert isinstance(leaf, MoleculeContainer)
        assert any(atom.stereo is not None for _, atom in leaf.atoms())


def test_pipeline_uses_exact_target_preprocessing_provenance() -> None:
    endpoint, record = deprotect_molecule_with_provenance(smiles(_TWO_GROUP_TARGET))
    assert record is not None
    route, _ = _mapped_identity_route(str(endpoint))

    result = postprocess_routes(
        {"route": route},
        BuildingBlockCatalog(()),
        target_smiles=_TWO_GROUP_TARGET,
        preprocessing_provenance=record,
        config=RoutePostprocessConfig(calculate_cost=False),
    )

    assert result.ok
    assert len(result.variants) == 2
    assert all(
        item.route["target_preprocessing_provenance"] == "exact"
        for item in result.variants
    )
    assert all(
        reaction["meta"]["preprocessing_provenance"] == "exact"
        for item in result.variants
        for reaction in _protection_reactions(item.route)
    )


def test_target_exact_provenance_rejects_taxonomy_drift() -> None:
    endpoint, record = deprotect_molecule_with_provenance(smiles(_TWO_GROUP_TARGET))
    assert record is not None
    route, _ = _mapped_identity_route(str(endpoint))
    drifted = {**record, "protective_rules_sha256": "0" * 64}

    with pytest.raises(
        TargetProtectionError,
        match="taxonomy differs from target preprocessing",
    ):
        restore_protected_target(
            route,
            _TWO_GROUP_TARGET,
            preprocessing_provenance=drifted,
        )


def test_target_exact_provenance_rejects_policy_mismatch() -> None:
    endpoint, record = deprotect_molecule_with_provenance(smiles(_TWO_GROUP_TARGET))
    assert record is not None
    route, _ = _mapped_identity_route(str(endpoint))
    wrong_policy = {**record, "deprotection_policy": "aggressive"}

    with pytest.raises(TargetProtectionError, match="does not match requested policy"):
        restore_protected_target(
            route,
            _TWO_GROUP_TARGET,
            preprocessing_provenance=wrong_policy,
        )
