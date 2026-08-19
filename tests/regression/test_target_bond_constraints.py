"""Regressions for target-bond identity across real reactor applications."""

from __future__ import annotations

import pytest
from chython import smarts, smiles

from synplan.chem.reaction import CanonicalRetroReactor, apply_reaction_rule
from synplan.chem.reaction.reactor import (
    iter_reaction_applications,
)
from synplan.chem.target_bonds import (
    TargetAtomProvenance,
    TargetBondConstraints,
    removed_target_bonds,
    target_bond_keys,
)
from synplan.mcts import tree as tree_module
from synplan.mcts.tree import Tree
from synplan.utils.config import TreeConfig


def _reactor(rule_smarts: str) -> CanonicalRetroReactor:
    rule = smarts(rule_smarts)
    return CanonicalRetroReactor(
        patterns=tuple(rule.reactants),
        products=tuple(rule.products),
        delete_atoms=False,
    )


class _Policy:
    def __init__(self, rules=()):
        self.rules = tuple(rules)

    def predict_reaction_rules(self, _precursor, _reaction_rules):
        for rule_id, rule in enumerate(self.rules):
            yield 1.0, rule, rule_id


class _Evaluator:
    def evaluate_node(self, **_kwargs):
        return 0.0


def _config(*, min_mol_size: int = 0, max_depth: int = 4) -> TreeConfig:
    return TreeConfig(
        algorithm="breadth_first",
        max_iterations=5,
        max_tree_size=50,
        max_time=5,
        max_depth=max_depth,
        min_mol_size=min_mol_size,
        silent=True,
        enable_pruning=False,
        search_strategy="expansion_first",
    )


def _tree(target, rules, bonds_state, *, building_blocks=(), min_mol_size=0):
    return Tree(
        target=target,
        config=_config(min_mol_size=min_mol_size),
        reaction_rules=rules,
        building_blocks=set(building_blocks),
        expansion_function=_Policy(rules),
        evaluation_function=_Evaluator(),
        bonds_state=bonds_state,
    )


def _applications(tree: Tree, node_id: int, rule: CanonicalRetroReactor):
    precursor = tree.nodes[node_id].curr_precursor
    return iter_reaction_applications(
        molecule=precursor.molecule,
        reaction_rule=rule,
        provenance=precursor.target_atom_provenance,
        constraints=tree.bond_constraints,
    )


def _context(tree: Tree, node_id: int) -> tree_module._ExpansionContext:
    node = tree.nodes[node_id]
    return tree_module._ExpansionContext(
        node_id=node_id,
        parent=node,
        previous_precursors=node.curr_precursor.prev_precursors,
        seen_products=set(),
    )


def _candidate(rule, rule_id: int) -> tree_module._RuleCandidate:
    return tree_module._RuleCandidate(
        probability=1.0,
        rule=rule,
        rule_id=rule_id,
        rule_source="policy",
        policy_rank=1,
    )


def test_frozen_bond_keeps_real_element_substitution_candidate():
    target = smiles("Cc1ccc(Cl)cc1")
    substitution = _reactor("[c:1]:[c:2]-[Cl;D1:3]>>[c:1]:[c:2]-[O;h1:3]")

    baseline = list(apply_reaction_rule(target, substitution))
    assert len(baseline) == 1
    with pytest.raises(ValueError, match="elements should be of the same type"):
        next(iter(substitution(target))).compose()

    provenance = TargetAtomProvenance.for_target(target)
    constraints = TargetBondConstraints.from_state(target, {(1, 2): 2})
    constrained = list(
        iter_reaction_applications(target, substitution, provenance, constraints)
    )

    assert len(constrained) == 1
    assert (1, 2) not in removed_target_bonds(
        (target, provenance), constrained[0].states
    )


def test_real_cleavage_of_frozen_target_bond_is_rejected():
    target = smiles("[CH3:1][CH2:2][OH:3]")
    cleavage = _reactor("[C;D2:1]-[O;D1:2]>>[C;D1:1].[O;D0:2]")
    provenance = TargetAtomProvenance.for_target(target)

    unconstrained = list(
        iter_reaction_applications(
            target,
            cleavage,
            provenance,
            TargetBondConstraints(),
        )
    )
    assert len(unconstrained) == 1
    assert (2, 3) in removed_target_bonds((target, provenance), unconstrained[0].states)

    frozen = TargetBondConstraints.from_state(target, {(2, 3): 2})
    assert list(iter_reaction_applications(target, cleavage, provenance, frozen)) == []


@pytest.mark.parametrize(("state", "terminal_added"), [(1, False), (2, True)])
def test_reused_atom_numbers_do_not_alias_target_atoms(state, terminal_added):
    target = smiles("[CH3:1][CH2:2][CH2:3][OH:4]")
    selected = (3, 4)
    split_rule = _reactor("[C;D2:1]-[C;D2:2]>>[C;D1:1].[C;D1:2]")
    add_rule = _reactor("[C;D1:1]-[C;D1:2]>>[C;D1:1]-[C;D2:2]-[C:3]-[C:4]")
    break_rule = _reactor("[C;D2:1]-[C;D1:2]>>[C;D1:1].[C;D0:2]")
    tree = _tree(
        target,
        [split_rule, add_rule, break_rule],
        {selected: state},
        building_blocks={"CO", "CCC", "C"},
    )

    step1 = next(
        application
        for application in _applications(tree, 1, split_rule)
        if sorted(map(str, application.products)) == ["CC", "CO"]
    )
    assert tree._add_child_if_new(_context(tree, 1), step1, _candidate(split_rule, 0))

    stock_product = next(
        precursor
        for precursor in tree.nodes[2].new_precursors
        if str(precursor) == "CO"
    )
    assert selected in target_bond_keys(
        ((stock_product.molecule, stock_product.target_atom_provenance),)
    )

    step2 = next(
        application
        for application in _applications(tree, 2, add_rule)
        if any(
            set(product.atoms_numbers) == {1, 2, 3, 4}
            for product in application.products
        )
    )
    assert step2.provenances == (TargetAtomProvenance(frozenset({(1, 1), (2, 2)})),)
    assert tree._add_child_if_new(_context(tree, 2), step2, _candidate(add_rule, 1))

    step3 = next(
        application
        for application in _applications(tree, 3, break_rule)
        if sorted(map(str, application.products)) == ["C", "CCC"]
    )
    assert (
        tree._add_child_if_new(_context(tree, 3), step3, _candidate(break_rule, 2))
        is terminal_added
    )

    if state == 1:
        assert tree.nodes[3].remaining_required_bonds == frozenset({selected})
        assert all(not node.is_solved() for node in tree.nodes.values())
    else:
        assert tree.nodes[4].is_solved()


def test_public_tree_run_accepts_required_and_rejects_frozen_true_cleavage():
    target = smiles("[CH3:1][CH2:2][CH2:3][OH:4]")
    selected = (3, 4)
    cleavage = _reactor("[C;D2:1]-[O;D1:2]>>[C;D1:1].[O;D0:2]")

    required_tree = _tree(
        target,
        [cleavage],
        {selected: 1},
        min_mol_size=10,
    ).run()
    assert len(required_tree.winning_nodes) == 1
    winning_node = required_tree.nodes[required_tree.winning_nodes[0]]
    assert winning_node.remaining_required_bonds == frozenset()
    winning_states = tuple(
        (precursor.molecule, precursor.target_atom_provenance)
        for precursor in winning_node.new_precursors
    )
    assert selected not in target_bond_keys(winning_states)

    frozen_tree = _tree(
        target,
        [cleavage],
        {selected: 2},
        min_mol_size=10,
    ).run()
    assert frozen_tree.winning_nodes == []
    assert frozen_tree.children[1] == set()
