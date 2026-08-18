"""Forward direction mode: the tree/evaluator coupling, and co-reactants for bimolecular rules.

Two independent silent failures. A forward search stops at the goal while a rollout evaluator that
was never told about it keeps rewarding the retro finish line; and a bimolecular forward rule
handed one structure matches nothing and yields zero reactions, which reads as "the rule does not
apply here" rather than "you gave it half the reactants".

Partner selection is deliberately not here: choosing WHICH co-reactant needs a model.
"""

import pytest
from chython import smiles
from test_tree_stats import FixedEvaluationStrategy, build_tree, make_mol

from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.reaction.reactor import apply_reaction_rule
from synplan.mcts.evaluation import RolloutEvaluationStrategy

# forward amide formation: two reactants, one product, and the chloride has to be deleted
AMIDE_FORWARD = "[C;X3:1](=[O:2])[Cl:3].[N;H2;X3:4]>>[C:1](=[O:2])[N:4]"


def _rollout(building_blocks, min_mol_size):
    return RolloutEvaluationStrategy(
        policy_network=None,
        reaction_rules=[],
        building_blocks=building_blocks,
        min_mol_size=min_mol_size,
        max_depth=4,
    )


def _molecule(smi: str):
    mol = smiles(smi)
    mol.canonicalize()
    return mol


# -- the tree/evaluator coupling ------------------------------------------------------------


def test_retro_is_the_default_and_is_not_policed():
    """The coupling check must not fire on the retro path — every existing caller lives there."""
    tree = build_tree()
    assert tree.config.direction == "retro"
    assert tree.building_blocks.keys == frozenset()


def test_forward_refuses_an_evaluator_scoring_a_different_finish_line():
    goal = {str(make_mol(7))}
    with pytest.raises(ValueError, match="same finish line"):
        build_tree(
            direction="forward",
            building_blocks=goal,
            evaluator=_rollout(building_blocks=set(), min_mol_size=6),
        )
    with pytest.raises(ValueError, match="same finish line"):
        build_tree(
            direction="forward",
            building_blocks=goal,
            evaluator=_rollout(building_blocks=goal, min_mol_size=0),
        )


def test_forward_accepts_an_evaluator_that_agrees():
    goal = {str(make_mol(7))}
    tree = build_tree(
        direction="forward",
        building_blocks=goal,
        evaluator=_rollout(building_blocks=goal, min_mol_size=6),
    )
    assert tree.building_blocks.keys == frozenset(goal)


def test_forward_refuses_an_empty_goal():
    """`building_blocks` is the goal in forward mode, so empty is unsatisfiable, not permissive."""
    with pytest.raises(ValueError, match="GOAL"):
        build_tree(direction="forward", evaluator=FixedEvaluationStrategy())


def test_an_evaluator_without_a_rollout_is_left_alone():
    """A value-network or random evaluator has no finish line of its own to disagree with."""
    goal = {str(make_mol(7))}
    tree = build_tree(
        direction="forward", building_blocks=goal, evaluator=FixedEvaluationStrategy()
    )
    assert tree.config.direction == "forward"


# -- co-reactants ---------------------------------------------------------------------------


def test_a_bimolecular_rule_needs_both_partners():
    """One structure yields nothing at all; the second one is what makes the rule fire."""
    rule = CanonicalRetroReactor.from_smarts(AMIDE_FORWARD, delete_atoms=True)
    chloride, amine = _molecule("c1ccccc1C(=O)Cl"), _molecule("CCN")

    assert list(apply_reaction_rule(chloride, rule)) == []
    products = [
        sorted(str(p) for p in out)
        for out in apply_reaction_rule(chloride, rule, co_reactants=(amine,))
    ]
    assert products == [["c1ccccc1C(NCC)=O"]]


def test_no_co_reactants_is_the_untouched_retro_path():
    """The default must be byte-identical to the call that has no idea the parameter exists."""
    rule = CanonicalRetroReactor.from_smarts(
        "[C;X3:1](=[O:2])[N;X3:3]>>[C:1](=[O:2])[Cl:90].[N:3]"
    )
    amide = _molecule("c1ccccc1C(=O)NCC")

    default = [sorted(str(p) for p in out) for out in apply_reaction_rule(amide, rule)]
    explicit = [
        sorted(str(p) for p in out)
        for out in apply_reaction_rule(amide, rule, co_reactants=())
    ]
    assert default == explicit == [["CCN", "c1ccccc1C(Cl)=O"]]
