import pytest

from synplan.chem.utils import mol_from_smiles
from synplan.mcts.config import (
    RolloutEvaluationConfig,
    TreeConfig,
)
from synplan.mcts.tree import Tree
from synplan.utils.loading import (
    download_preset,
    load_building_blocks,
    load_evaluation_function,
    load_policy_function,
    load_reaction_rules,
)

SIMPLE_TARGET_SMILES = "CCNc1nc(Sc2ccc(C)cc2)cc(C(F)(F)F)n1"


@pytest.fixture(scope="module")
def data_paths():
    """Download preset data for the planner smoke test."""
    return download_preset(
        preset_name="synplanner-gps", save_to="./tutorials/synplan_data"
    )


@pytest.fixture(scope="module")
def building_blocks(data_paths):
    """Load building blocks for the planner smoke test."""
    return load_building_blocks(
        data_paths["building_blocks"], standardize=False, silent=True
    )


@pytest.fixture(scope="module")
def reaction_rules(data_paths):
    """Load reaction rules for the planner smoke test."""
    return load_reaction_rules(data_paths["reaction_rules"])


@pytest.fixture(scope="module")
def policy_network(data_paths):
    """Initialize policy network for the planner smoke test."""
    return load_policy_function(weights_path=data_paths["ranking_policy"])


@pytest.fixture(scope="module")
def planner_smoke_config():
    """Small MCTS configuration used only to verify planner wiring."""
    return TreeConfig(
        search_strategy="expansion_first",
        algorithm="UCT",
        enable_pruning=False,
        max_iterations=300,
        max_time=120,
        max_depth=6,
        min_mol_size=1,
        silent=True,
    )


def run_planner_smoke(
    target_smiles, building_blocks, reaction_rules, policy_network, tree_config
):
    """Run MCTS for one small target without exercising clustering."""
    target_molecule = mol_from_smiles(
        target_smiles, clean2d=True, standardize=True, clean_stereo=True
    )
    eval_config = RolloutEvaluationConfig(
        policy_network=policy_network,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        min_mol_size=tree_config.min_mol_size,
        max_depth=tree_config.max_depth,
        normalize=tree_config.normalize_scores,
    )
    evaluator = load_evaluation_function(eval_config)

    tree = Tree(
        target=target_molecule,
        config=tree_config,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        expansion_function=policy_network,
        evaluation_function=evaluator,
    )

    tree_solved = False
    for solved, _ in tree:
        if solved:
            tree_solved = True
    tree._log_final_stats("completed")
    return tree, tree_solved


@pytest.mark.integration
def test_simple_mcts_planner_smoke(
    building_blocks, reaction_rules, policy_network, planner_smoke_config
):
    """Keep one small planner smoke test separate from clustering assertions."""
    tree, tree_solved = run_planner_smoke(
        SIMPLE_TARGET_SMILES,
        building_blocks,
        reaction_rules,
        policy_network,
        planner_smoke_config,
    )

    assert tree_solved, f"Tree solving failed for molecule: {SIMPLE_TARGET_SMILES}"
    assert tree.winning_nodes, "Solved planner smoke test should record winning nodes"
