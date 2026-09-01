import pytest
from chython import smiles
from frozendict import frozendict

from synplan.chem.building_blocks import (
    BuildingBlock,
    molecule_has_stereo,
    molecule_to_inchikey,
)
from synplan.mcts.config import TreeConfig
from synplan.mcts.evaluation import RolloutEvaluationStrategy
from synplan.mcts.tree import Tree


class _EmptyPolicy:
    def predict_reaction_rules(self, precursor, reaction_rules):
        return iter(())


def _catalogue():
    molecule = smiles("C[C@H](O)C(=O)O", ignore_stereo=False)
    block = BuildingBlock(
        smiles=str(molecule),
        inchikey=molecule_to_inchikey(molecule),
        vendors=frozendict({"vendor": 1.0}),
        has_stereo=True,
    )
    return (
        frozendict({block.inchikey: block}),
        frozendict({block.inchikey[:14]: (block,)}),
    )


@pytest.mark.parametrize(
    ("target_smiles", "expected_exact"),
    [("C[C@H](F)Cl", True), ("CC(F)Cl", False)],
)
def test_target_selects_one_catalogue_policy_for_tree_and_rollout(
    target_smiles, expected_exact
):
    blocks, candidates = _catalogue()
    policy = _EmptyPolicy()
    evaluator = RolloutEvaluationStrategy(
        policy_network=policy,
        reaction_rules=(),
        building_blocks=blocks,
        min_mol_size=0,
        max_depth=2,
        building_block_candidates=candidates,
    )
    target = smiles(target_smiles, ignore_stereo=False)
    tree = Tree(
        target=target,
        config=TreeConfig(max_iterations=1, min_mol_size=0, silent=True),
        reaction_rules=[],
        building_blocks=blocks,
        building_block_candidates=candidates,
        expansion_function=policy,
        evaluation_function=evaluator,
    )

    assert tree.use_full_inchikey is expected_exact
    assert evaluator.rollout.use_full_inchikey is expected_exact
    assert evaluator.rollout.building_block_candidates is candidates
    assert molecule_has_stereo(tree.nodes[1].curr_precursor.molecule) is expected_exact


def test_json_catalogue_is_rejected_for_forward_search():
    blocks, candidates = _catalogue()
    policy = _EmptyPolicy()
    evaluator = RolloutEvaluationStrategy(
        policy_network=policy,
        reaction_rules=(),
        building_blocks=blocks,
        min_mol_size=0,
        max_depth=2,
        building_block_candidates=candidates,
    )

    with pytest.raises(ValueError, match="only for retrosynthesis"):
        Tree(
            target=smiles("CCCCCCC"),
            config=TreeConfig(direction="forward", max_iterations=1, silent=True),
            reaction_rules=[],
            building_blocks=blocks,
            building_block_candidates=candidates,
            expansion_function=policy,
            evaluation_function=evaluator,
        )
