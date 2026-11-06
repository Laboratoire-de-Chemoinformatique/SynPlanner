from __future__ import annotations

from chython import smiles

from synplan.chem.target_bonds import (
    TargetAtomProvenance,
    TargetBondConstraints,
    removed_target_bonds,
    selected_bonds_svg,
    target_bond_keys,
)


def test_provenance_inheritance_excludes_introduced_number_collisions():
    provenance = TargetAtomProvenance.from_mapping({1: 1, 2: 2})
    product = smiles("[CH3:1][CH2:2][CH2:3][CH3:4]")

    inherited = provenance.inherit(product)

    assert inherited.as_dict() == {1: 1, 2: 2}
    assert 3 not in inherited.as_dict()
    assert 4 not in inherited.as_dict()


def test_fragmentation_translates_only_surviving_target_adjacencies():
    target = smiles("[CH3:1][CH2:2][CH2:3][CH3:4]")
    provenance = TargetAtomProvenance.for_target(target)
    products = (
        smiles("[CH3:1][CH3:2]"),
        smiles("[CH3:3][CH3:4]"),
    )
    states = tuple((product, provenance.inherit(product)) for product in products)

    assert target_bond_keys(states) == {(1, 2), (3, 4)}
    assert removed_target_bonds((target, provenance), states) == {(2, 3)}


def test_bond_order_and_endpoint_element_changes_preserve_adjacency():
    alkene = smiles("[CH2:1]=[CH2:2]")
    alkene_provenance = TargetAtomProvenance.for_target(alkene)
    alkane = smiles("[CH3:1][CH3:2]")

    carbon_chlorine = smiles("[CH3:1][Cl:2]")
    substitution_provenance = TargetAtomProvenance.for_target(carbon_chlorine)
    carbon_oxygen = smiles("[CH3:1][OH:2]")

    assert (
        removed_target_bonds(
            (alkene, alkene_provenance),
            ((alkane, alkene_provenance.inherit(alkane)),),
        )
        == set()
    )
    assert (
        removed_target_bonds(
            (carbon_chlorine, substitution_provenance),
            ((carbon_oxygen, substitution_provenance.inherit(carbon_oxygen)),),
        )
        == set()
    )


def test_endpoint_deletion_removes_target_adjacency():
    target = smiles("[CH3:1][CH3:2]")
    provenance = TargetAtomProvenance.for_target(target)
    product = smiles("[CH4:1]")

    assert removed_target_bonds(
        (target, provenance), ((product, provenance.inherit(product)),)
    ) == {(1, 2)}


def test_constraint_value_is_immutable_and_returns_defensive_mapping():
    target = smiles("[CH3:1][CH2:2][CH3:3]")
    constraints = TargetBondConstraints.from_state(target, {(2, 1): 1, (2, 3): 2})

    snapshot = constraints.as_dict()
    snapshot[(1, 2)] = 0

    assert constraints.required == frozenset({(1, 2)})
    assert constraints.frozen == frozenset({(2, 3)})
    assert constraints.as_dict() == {(1, 2): 1, (2, 3): 2}


def test_selected_bonds_svg_highlights_normalized_nonzero_constraints():
    target = smiles("[CH3:1][CH2:2][CH3:3]")
    target.clean2d()

    svg = selected_bonds_svg(target, {(2, 1): 1, (2, 3): 2})

    assert svg.count('stroke="red"') == 1
    assert svg.count('stroke="blue"') == 1
    assert svg.endswith("</svg>")


def test_selected_bonds_svg_omits_neutral_constraints():
    target = smiles("[CH3:1][CH2:2][CH3:3]")
    target.clean2d()

    svg = selected_bonds_svg(target, {(1, 2): 0})

    assert 'stroke="red"' not in svg
    assert 'stroke="blue"' not in svg
