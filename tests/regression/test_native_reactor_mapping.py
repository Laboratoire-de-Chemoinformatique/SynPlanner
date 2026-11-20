"""Native reactor atom identity and policy-compatibility regressions."""

import pytest
from chython import smarts, smiles

from synplan.chem.reaction import (
    AtomMappingState,
    CanonicalRetroReactor,
    apply_reaction_rule,
)
from synplan.chem.reaction.reactor import (
    ATOM_MAPPING_STATE_META,
    ATOM_ORIGINS_META,
    LEGACY_ATOM_MAPS_META,
    initialize_atom_origins,
)


def _reactor(rule_text: str, *, native: bool) -> CanonicalRetroReactor:
    rule = smarts(rule_text)
    return CanonicalRetroReactor(
        patterns=tuple(rule.reactants),
        products=tuple(rule.products),
        delete_atoms=False,
        reconcile_mapping=native,
    )


def test_fresh_atom_is_allocated_above_both_lineage_ceilings():
    molecule = smiles("CC(C)O")
    root = initialize_atom_origins(molecule)
    state = AtomMappingState(root.map_ceiling + 20, root.origin_ceiling + 30)
    reactor = _reactor(
        "[C;D3:1]-[O;D1:2]>>[C;D3:1]-[O;D2:2]-[C;D1:3]",
        native=True,
    )

    reaction = next(iter(reactor(molecule, atom_mapping_state=state)))
    product = reaction.products[0]

    assert product.has_atom(state.map_ceiling + 1)
    assert product.meta[ATOM_ORIGINS_META][state.map_ceiling + 1] == (
        state.origin_ceiling + 1
    )
    assert product.meta[ATOM_MAPPING_STATE_META] == (
        state.map_ceiling + 1,
        state.origin_ceiling + 1,
    )


def test_element_substitution_gets_a_new_identity_but_ring_atoms_do_not():
    molecule = smiles("c1ccccc1Cl")
    state = initialize_atom_origins(molecule)
    original_origins = dict(molecule.meta[ATOM_ORIGINS_META])
    reactor = _reactor("[c:1]-[Cl:2]>>[c:1]-[O:2]", native=True)

    product = next(iter(reactor(molecule, atom_mapping_state=state))).products[0]
    oxygen_map = next(n for n, atom in product.atoms() if atom.atomic_number == 8)

    assert oxygen_map == state.map_ceiling + 1
    assert product.meta[ATOM_ORIGINS_META][oxygen_map] == state.origin_ceiling + 1
    for atom_map in set(product) & set(molecule):
        assert product.meta[ATOM_ORIGINS_META][atom_map] == original_origins[atom_map]
    assert product.meta[LEGACY_ATOM_MAPS_META][oxygen_map] == next(
        n for n, atom in molecule.atoms() if atom.atomic_number == 17
    )


def test_split_products_receive_independent_complete_origin_dicts():
    molecule = smiles("CC(=O)N(C)C")
    state = initialize_atom_origins(molecule)
    reactor = _reactor(
        "[N;D3:1]-[C;D3:2]=[O:3]>>[N;D2:1].[C;D3:2]=[O:3]",
        native=True,
    )

    products = next(iter(reactor(molecule, atom_mapping_state=state))).products

    assert len(products) == 2
    assert all(set(p.meta[ATOM_ORIGINS_META]) == set(p) for p in products)
    assert (
        products[0].meta[ATOM_ORIGINS_META] is not products[1].meta[ATOM_ORIGINS_META]
    )


def test_split_fragment_reuses_neither_sibling_map_nor_origin_without_state():
    molecule = smiles("CCCO")
    state = initialize_atom_origins(molecule)
    split_rule = _reactor("[C:1]-[O:2]>>[C:1].[O:2]", native=True)
    products = next(iter(split_rule(molecule, atom_mapping_state=state))).products
    carbon = next(product for product in products if len(product) == 3)
    sibling = next(product for product in products if len(product) == 1)
    add_rule = _reactor(
        "[C;D1:1]>>[C;D2:1]-[O;D1:2]",
        native=True,
    )

    product = next(iter(add_rule(carbon))).products[0]

    assert product.has_atom(state.map_ceiling + 1)
    assert product.meta[ATOM_ORIGINS_META][state.map_ceiling + 1] == (
        state.origin_ceiling + 1
    )
    assert set(product).isdisjoint(sibling)
    assert set(product.meta[ATOM_ORIGINS_META].values()).isdisjoint(
        sibling.meta[ATOM_ORIGINS_META].values()
    )
    with pytest.raises(ValueError, match="below the input provenance"):
        list(
            add_rule(
                carbon,
                atom_mapping_state=AtomMappingState(
                    state.map_ceiling - 1,
                    state.origin_ceiling - 1,
                ),
            )
        )


def test_multiple_product_only_atoms_allocate_consecutive_ids():
    molecule = smiles("Nc1ccccc1")
    state = initialize_atom_origins(molecule)
    reactor = _reactor("[c:1]-[N:2]>>[c:1]-[N+:2](-[O-:3])=[O:4]", native=True)

    product = next(iter(reactor(molecule, atom_mapping_state=state))).products[0]

    assert {state.map_ceiling + 1, state.map_ceiling + 2}.issubset(set(product))
    assert {
        product.meta[ATOM_ORIGINS_META][state.map_ceiling + 1],
        product.meta[ATOM_ORIGINS_META][state.map_ceiling + 2],
    } == {state.origin_ceiling + 1, state.origin_ceiling + 2}


def test_isotope_loss_allocates_a_new_identity_while_charge_change_does_not():
    isotope_molecule = smiles("[13CH3]C")
    isotope_state = initialize_atom_origins(isotope_molecule)
    isotope_rule = _reactor("[C:1]-[C:2]>>[C:1]-[C:2]", native=True)
    isotope_products = list(
        isotope_rule(isotope_molecule, atom_mapping_state=isotope_state)
    )
    changed = next(
        product
        for reaction in isotope_products
        for product in reaction.products
        if all(atom.isotope is None for _, atom in product.atoms())
    )
    assert changed.has_atom(isotope_state.map_ceiling + 1)

    charged_molecule = smiles("CN")
    charged_state = initialize_atom_origins(charged_molecule)
    charge_rule = _reactor("[N:1]>>[N+:1]", native=True)
    charged = next(
        iter(charge_rule(charged_molecule, atom_mapping_state=charged_state))
    ).products[0]
    assert charged.meta[ATOM_MAPPING_STATE_META] == (
        charged_state.map_ceiling,
        charged_state.origin_ceiling,
    )


def test_identity_preserving_rule_reuses_immutable_lineage_tables(monkeypatch):
    molecule = smiles("CN")
    state = initialize_atom_origins(molecule)
    origins = molecule.meta[ATOM_ORIGINS_META]
    legacy_maps = {next(iter(molecule)): 100}
    molecule.meta[LEGACY_ATOM_MAPS_META] = legacy_maps
    charge_rule = _reactor("[N:1]>>[N+:1]", native=True)
    monkeypatch.setattr(
        charge_rule,
        "_allocate_native_mapping",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("identity-preserving rule entered allocator")
        ),
    )

    product = next(
        apply_reaction_rule(
            molecule,
            charge_rule,
            atom_mapping_state=state,
        )
    )[0]

    assert set(product) == set(molecule)
    assert product.meta[ATOM_ORIGINS_META] is origins
    assert product.meta[LEGACY_ATOM_MAPS_META] is legacy_maps


def test_multirule_carries_candidate_specific_mapping_state():
    molecule = smiles("OCCO")
    state = initialize_atom_origins(molecule)
    reactor = _reactor("[C:1]-[O;D1:2]>>[C:1]-[O;D2:2]-[C;D1:3]", native=True)

    outcomes = list(
        apply_reaction_rule(
            molecule,
            reactor,
            multirule=True,
            rm_dup=True,
            top_reactions_num=5,
            atom_mapping_state=state,
        )
    )
    ceilings = {
        product.meta[ATOM_MAPPING_STATE_META]
        for outcome in outcomes
        for product in outcome
    }

    assert (state.map_ceiling + 1, state.origin_ceiling + 1) in ceilings
    assert (state.map_ceiling + 2, state.origin_ceiling + 2) in ceilings


def test_baseline_mode_does_not_add_private_mapping_metadata():
    molecule = smiles("c1ccccc1Cl")
    product = next(
        iter(_reactor("[c:1]-[Cl:2]>>[c:1]-[O:2]", native=False)(molecule))
    ).products[0]

    assert ATOM_ORIGINS_META not in product.meta
    assert LEGACY_ATOM_MAPS_META not in product.meta
    assert ATOM_MAPPING_STATE_META not in product.meta
