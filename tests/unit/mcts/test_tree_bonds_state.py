from __future__ import annotations

from unittest.mock import Mock

import pytest
from chython import smiles

from synplan.chem.reaction.reactor import ReactionApplication
from synplan.chem.target_bonds import TargetAtomProvenance
from synplan.mcts import tree as tree_module
from synplan.mcts.config import TreeConfig
from synplan.mcts.node import Node
from synplan.mcts.record import read_search_record, write_search_record
from synplan.mcts.tree import Tree


class FakePolicy:
    def __init__(self, rule):
        self.rule = rule

    def predict_reaction_rules(self, precursor, reaction_rules):
        yield 1.0, self.rule, 0


def test_tree_forwards_immutable_constraints_and_root_provenance(monkeypatch):
    rule = object()
    bonds_state = {(3, 2): 2}
    observed = []

    def fake_iter_reaction_applications(**kwargs):
        observed.append(kwargs)
        return iter(())

    monkeypatch.setattr(
        tree_module,
        "iter_reaction_applications",
        fake_iter_reaction_applications,
    )

    config = TreeConfig(
        max_iterations=1,
        max_time=1,
        max_depth=1,
        min_mol_size=0,
        silent=True,
        enable_pruning=False,
    )
    tree = Tree(
        target=smiles("CCCC"),
        config=config,
        reaction_rules=[rule],
        building_blocks=set(),
        expansion_function=FakePolicy(rule),
        evaluation_function=object(),
        bonds_state=bonds_state,
    )

    with pytest.raises(StopIteration):
        tree._expand_node(1)

    assert tree.bonds_state == {(2, 3): 2}
    assert len(observed) == 1
    assert observed[0]["molecule"] == tree.nodes[1].curr_precursor.molecule
    assert observed[0]["reaction_rule"] is rule
    assert observed[0]["provenance"] == (
        tree.nodes[1].curr_precursor.target_atom_provenance
    )
    assert observed[0]["constraints"] is tree.bond_constraints


def _build_tree(bonds_state=None, *, min_mol_size=0, enable_pruning=False):
    rule = object()
    config = TreeConfig(
        algorithm="breadth_first",
        max_iterations=5,
        max_time=1,
        max_depth=4,
        min_mol_size=min_mol_size,
        silent=True,
        enable_pruning=enable_pruning,
    )
    return Tree(
        target=smiles("[CH3:1][CH2:2][CH2:3][CH3:4]"),
        config=config,
        reaction_rules=[rule],
        building_blocks=set(),
        expansion_function=FakePolicy(rule),
        evaluation_function=object(),
        bonds_state=bonds_state,
    )


def _candidate():
    return tree_module._RuleCandidate(
        probability=1.0,
        rule=object(),
        rule_id=0,
        rule_source="policy",
        policy_rank=1,
    )


def _context(tree, node_id):
    node = tree.nodes[node_id]
    return tree_module._ExpansionContext(
        node_id=node_id,
        parent=node,
        previous_precursors=node.curr_precursor.prev_precursors,
        seen_products=set(),
    )


def _reaction_application(tree, node_id, reaction_smiles):
    products = tuple(smiles(reaction_smiles).products)
    parent_provenance = tree.nodes[node_id].curr_precursor.target_atom_provenance
    return ReactionApplication(
        products=products,
        provenances=tuple(parent_provenance.inherit(product) for product in products),
    )


@pytest.mark.parametrize(
    ("bonds_state", "message"),
    [
        ([], "must be a mapping"),
        ({(1,): 1}, "two-item tuple"),
        ({(1, 1): 1}, "different atoms"),
        ({(1, "2"): 1}, "must be integers"),
        ({(1, 2): True}, "must be integers"),
        ({(1, 99): 0}, "not present in the target"),
        ({(1, 2): 3}, "must be 0, 1, or 2"),
        ({(1, 99): 1}, "not present in the target"),
        ({(1, 2): 1, (2, 1): 2}, "conflicting states"),
    ],
)
def test_tree_rejects_invalid_bonds_state(bonds_state, message):
    with pytest.raises(ValueError, match=message):
        _build_tree(bonds_state)


def test_tree_normalizes_states_and_initializes_required_bonds():
    tree = _build_tree({(2, 1): 1, (3, 2): 2, (4, 3): 0})

    assert tree.bonds_state == {(1, 2): 1, (2, 3): 2, (3, 4): 0}
    assert tree.required_break_bonds == frozenset({(1, 2)})
    assert tree.nodes[1].remaining_required_bonds == frozenset({(1, 2)})


def test_bonds_state_is_a_defensive_snapshot():
    tree = _build_tree({(1, 2): 1})

    snapshot = tree.bonds_state
    snapshot[(1, 2)] = 2

    assert tree.bonds_state == {(1, 2): 1}
    assert tree.required_break_bonds == frozenset({(1, 2)})
    assert tree.nodes[1].remaining_required_bonds == frozenset({(1, 2)})


def test_tree_without_constraints_preserves_empty_required_state():
    tree = _build_tree()

    assert tree.bonds_state == {}
    assert tree.required_break_bonds == frozenset()
    assert tree.nodes[1].remaining_required_bonds == frozenset()


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize(
    "stereo_obligations", [(), ({"reason": "requires_stereo_forming_step"},)]
)
@pytest.mark.parametrize("remaining_required_bonds", [frozenset(), frozenset({(1, 2)})])
def test_node_is_solved_requires_all_obligations_complete(
    terminal, stereo_obligations, remaining_required_bonds
):
    node = Node(
        precursors_to_expand=() if terminal else (object(),),
        new_precursors=(),
        stereo_obligations=stereo_obligations,
        remaining_required_bonds=remaining_required_bonds,
    )

    assert node.is_terminal() is terminal
    assert node.is_solved() is (
        terminal and not stereo_obligations and not remaining_required_bonds
    )


@pytest.mark.parametrize("bonds_state", [None, {}, {(1, 2): 0}, {(1, 2): 2}])
def test_tree_skips_required_bond_scan_without_requirements(monkeypatch, bonds_state):
    scan = Mock(wraps=tree_module.removed_target_bonds)
    monkeypatch.setattr(tree_module, "removed_target_bonds", scan)
    tree = _build_tree(bonds_state)
    products = _reaction_application(
        tree, 1, "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH3:1][CH2:2][CH3:3].[CH4:4]"
    )

    assert tree._add_child_if_new(_context(tree, 1), products, _candidate())
    assert not tree.nodes[2].remaining_required_bonds
    scan.assert_not_called()


def test_tree_skips_required_bond_scan_after_requirements_are_satisfied(monkeypatch):
    scan = Mock(wraps=tree_module.removed_target_bonds)
    monkeypatch.setattr(tree_module, "removed_target_bonds", scan)
    tree = _build_tree({(3, 4): 1}, min_mol_size=1)
    first_products = _reaction_application(
        tree, 1, "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH3:1][CH2:2][CH3:3].[CH4:4]"
    )

    assert tree._add_child_if_new(_context(tree, 1), first_products, _candidate())
    assert not tree.nodes[2].remaining_required_bonds
    assert tree.nodes[1].remaining_required_bonds == frozenset({(3, 4)})
    scan.assert_called_once()

    next_products = _reaction_application(
        tree, 2, "[CH3:1][CH2:2][CH3:3]>>[CH3:1][CH3:2].[CH4:3]"
    )
    assert tree._add_child_if_new(_context(tree, 2), next_products, _candidate())
    assert not tree.nodes[3].remaining_required_bonds
    scan.assert_called_once()


def test_terminal_route_is_rejected_until_required_bond_is_broken():
    tree = _build_tree({(1, 2): 1}, min_mol_size=10)
    products = _reaction_application(
        tree, 1, "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH3:1][CH2:2][CH3:3].[CH4:4]"
    )

    added = tree._add_child_if_new(_context(tree, 1), products, _candidate())

    assert added is False
    assert tree.children[1] == set()
    assert tree.curr_tree_size == 2
    assert set(tree.nodes) == {1}
    assert tree.proposal_nodes == []
    assert tree.winning_nodes == []


def test_terminal_route_is_accepted_after_required_bond_is_broken():
    tree = _build_tree({(1, 2): 1}, min_mol_size=10)
    products = _reaction_application(
        tree, 1, "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH4:1].[CH3:2][CH2:3][CH3:4]"
    )

    added = tree._add_child_if_new(_context(tree, 1), products, _candidate())

    assert added is True
    child = tree.nodes[2]
    assert child.is_solved() is True
    assert child.remaining_required_bonds == frozenset()


def test_required_bonds_can_be_broken_across_multiple_steps(monkeypatch):
    scan = Mock(wraps=tree_module.removed_target_bonds)
    monkeypatch.setattr(tree_module, "removed_target_bonds", scan)
    tree = _build_tree({(1, 2): 1, (2, 3): 1}, min_mol_size=1)

    unrelated_products = _reaction_application(
        tree, 1, "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH3:1][CH2:2][CH3:3].[CH4:4]"
    )
    assert tree._add_child_if_new(_context(tree, 1), unrelated_products, _candidate())
    assert tree.nodes[2].remaining_required_bonds == frozenset({(1, 2), (2, 3)})
    assert scan.call_count == 1

    first_required_products = _reaction_application(
        tree, 2, "[CH3:1][CH2:2][CH3:3]>>[CH4:1].[CH3:2][CH3:3]"
    )
    assert tree._add_child_if_new(
        _context(tree, 2), first_required_products, _candidate()
    )
    assert tree.nodes[3].remaining_required_bonds == frozenset({(2, 3)})
    assert tree.nodes[2].remaining_required_bonds == frozenset({(1, 2), (2, 3)})
    assert scan.call_count == 2

    second_required_products = _reaction_application(
        tree, 3, "[CH3:2][CH3:3]>>[CH4:2].[CH4:3]"
    )
    assert tree._add_child_if_new(
        _context(tree, 3), second_required_products, _candidate()
    )
    assert tree.nodes[4].is_solved() is True
    assert tree.nodes[4].remaining_required_bonds == frozenset()
    assert tree.nodes[3].remaining_required_bonds == frozenset({(2, 3)})
    assert tree.nodes[1].remaining_required_bonds == frozenset({(1, 2), (2, 3)})
    assert scan.call_count == 3


def test_pruning_key_includes_remaining_required_bonds():
    tree = _build_tree({(1, 2): 1}, min_mol_size=0, enable_pruning=True)
    products = _reaction_application(
        tree, 1, "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH3:1][CH2:2][CH3:3].[CH4:4]"
    )

    assert tree._add_child_if_new(_context(tree, 1), products, _candidate())

    keys = tree.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks
    assert len(keys) == 1
    ((_precursors, remaining, _obligations, _history),) = keys
    assert remaining == frozenset({(1, 2)})


def test_state_zero_preserves_structure_only_pruning_key():
    tree = _build_tree({(1, 2): 0}, min_mol_size=0, enable_pruning=True)
    application = _reaction_application(
        tree,
        1,
        "[CH3:1][CH2:2][CH2:3][CH3:4]>>[CH3:1][CH2:2][CH3:3].[CH4:4]",
    )

    assert tree._add_child_if_new(_context(tree, 1), application, _candidate())

    (pruning_key,) = (
        tree.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks
    )
    assert tree.bond_constraints.active is False
    node = tree.nodes[2]
    assert pruning_key == (
        node.precursors_to_expand,
        node.stereo_obligations.key,
        node.stereo_history,
    )


def test_active_candidate_dedup_and_pruning_distinguish_provenance():
    tree = _build_tree({(1, 2): 2}, min_mol_size=0, enable_pruning=True)
    product = smiles("[CH3:1][CH2:2][CH3:3]")
    first = ReactionApplication(
        products=(product,),
        provenances=(
            tree.nodes[1].curr_precursor.target_atom_provenance.inherit(product),
        ),
    )
    second = ReactionApplication(
        products=(product.copy(),),
        provenances=(TargetAtomProvenance.from_mapping({1: 1, 2: 2, 3: 4}),),
    )
    context = _context(tree, 1)

    assert tree._add_child_if_new(context, first, _candidate())
    assert tree._add_child_if_new(context, second, _candidate())
    assert len(tree.children[1]) == 2
    assert (
        len(tree.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks)
        == 2
    )


def test_active_cycle_detection_distinguishes_provenance():
    constrained = _build_tree({(1, 2): 2}, min_mol_size=0)
    root = constrained.nodes[1].curr_precursor.molecule
    different_provenance = ReactionApplication(
        products=(root.copy(),),
        provenances=(TargetAtomProvenance.from_mapping({1: 1, 2: 2, 3: 4}),),
    )
    assert constrained._add_child_if_new(
        _context(constrained, 1), different_provenance, _candidate()
    )

    unconstrained = _build_tree(min_mol_size=0)
    structure_only = ReactionApplication(
        products=(unconstrained.nodes[1].curr_precursor.molecule.copy(),),
        provenances=(unconstrained.nodes[1].curr_precursor.target_atom_provenance,),
    )
    assert not unconstrained._add_child_if_new(
        _context(unconstrained, 1), structure_only, _candidate()
    )


def test_search_record_preserves_target_bonds_and_equal_molecule_provenance(tmp_path):
    tree = _build_tree({(1, 2): 1})
    molecule = smiles("[CH3:1][CH3:2]")
    application = ReactionApplication(
        products=(molecule, molecule.copy()),
        provenances=(
            TargetAtomProvenance.from_mapping({1: 1, 2: 2}),
            TargetAtomProvenance.from_mapping({1: 3, 2: 4}),
        ),
    )
    assert tree._add_child_if_new(_context(tree, 1), application, _candidate())
    node = tree.nodes[2]
    node.precursors_to_expand = tuple(reversed(node.new_precursors))
    node.curr_precursor, *rest = node.precursors_to_expand
    node.next_precursor = tuple(rest)

    path = write_search_record(tree, tmp_path / "search.json")
    record = read_search_record(path)
    assert record.bonds_state == tree.bonds_state
    assert record.nodes[2].remaining_required_bonds == frozenset({(1, 2)})
    assert record.nodes[2].curr_precursor is record.nodes[2].new_precursors[1]
    assert [p.target_atom_provenance for p in record.nodes[2].new_precursors] == list(
        application.provenances
    )
    rewritten = read_search_record(
        write_search_record(record, tmp_path / "copy.json.gz")
    )
    assert rewritten.bonds_state == record.bonds_state
    assert (
        rewritten.nodes[2].curr_precursor.target_atom_provenance
        == application.provenances[1]
    )
