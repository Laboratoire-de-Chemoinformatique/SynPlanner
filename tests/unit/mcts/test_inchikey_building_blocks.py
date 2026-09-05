import pytest
from chython import smiles
from frozendict import frozendict

from synplan.chem.building_blocks import (
    BuildingBlock,
    molecule_has_stereo,
    molecule_to_inchikey,
)
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.config import TreeConfig
from synplan.mcts.evaluation import RolloutEvaluationStrategy
from synplan.mcts.tree import Tree

BOCEPREVIR_SMILES = (
    "CC1([C@@H]2[C@H]1[C@H](N(C2)C(=O)[C@H](C(C)(C)C)"
    "NC(=O)NC(C)(C)C)C(=O)NC(CC3CCC3)C(=O)C(=O)N)C"
)


class _EmptyPolicy:
    def predict_reaction_rules(self, precursor, reaction_rules):
        return iter(())


def test_boceprevir_uses_the_connectivity_only_key():
    molecule = mol_from_smiles(BOCEPREVIR_SMILES, clean_stereo=True)

    assert not molecule_has_stereo(molecule)
    assert molecule_to_inchikey(molecule) == "LHHCSNFAOIFYRV-UHFFFAOYSA-N"


def _catalogue():
    molecule = smiles("C[C@H](O)C(=O)O", ignore_stereo=False)
    block = BuildingBlock(
        smiles=str(molecule),
        inchikey=molecule_to_inchikey(molecule),
        vendors=frozendict({"vendor": 1.0}),
        has_stereo=True,
    )
    return frozendict({block.inchikey[:14]: (block,)})


@pytest.mark.parametrize("target_smiles", ["C[C@H](F)Cl", "CC(F)Cl"])
def test_tree_and_rollout_share_one_connectivity_catalogue(target_smiles):
    blocks = _catalogue()
    policy = _EmptyPolicy()
    evaluator = RolloutEvaluationStrategy(
        policy_network=policy,
        reaction_rules=(),
        building_blocks=blocks,
        min_mol_size=0,
        max_depth=2,
    )
    target = smiles(target_smiles, ignore_stereo=False)
    tree = Tree(
        target=target,
        config=TreeConfig(max_iterations=1, min_mol_size=0, silent=True),
        reaction_rules=[],
        building_blocks=blocks,
        expansion_function=policy,
        evaluation_function=evaluator,
    )

    assert not hasattr(tree, "use_full_inchikey")
    assert not hasattr(evaluator.rollout, "use_full_inchikey")
    assert tree.building_blocks is blocks
    assert evaluator.rollout.building_blocks is blocks
    assert not molecule_has_stereo(tree.nodes[1].curr_precursor.molecule)


def test_json_catalogue_is_rejected_for_forward_search():
    blocks = _catalogue()
    policy = _EmptyPolicy()
    evaluator = RolloutEvaluationStrategy(
        policy_network=policy,
        reaction_rules=(),
        building_blocks=blocks,
        min_mol_size=0,
        max_depth=2,
    )

    with pytest.raises(ValueError, match="only for retrosynthesis"):
        Tree(
            target=smiles("CCCCCCC"),
            config=TreeConfig(direction="forward", max_iterations=1, silent=True),
            reaction_rules=[],
            building_blocks=blocks,
            expansion_function=policy,
            evaluation_function=evaluator,
        )
