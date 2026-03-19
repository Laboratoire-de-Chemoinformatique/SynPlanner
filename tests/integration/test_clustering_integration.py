import logging

import pytest

from synplan.chem.reaction_routes.clustering import (
    cluster_routes,
    subcluster_all_clusters,
)
from synplan.chem.reaction_routes.route_cgr import (
    compose_all_route_cgrs,
    compose_all_sb_cgrs,
)
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.tree import Tree
from synplan.utils.config import RolloutEvaluationConfig, TreeConfig
from synplan.utils.loading import (
    download_preset,
    load_building_blocks,
    load_evaluation_function,
    load_policy_function,
    load_reaction_rules,
)

# Test molecules with different complexity levels
TEST_MOLECULES = {
    "simple": "CCNc1nc(Sc2ccc(C)cc2)cc(C(F)(F)F)n1",
    "medium": "c1cc(ccc1C2=NN(C(C=C2)=N)CCCC(O)=O)OCc3cc([N+]([O-])=O)ccc3",
    "complex": "c1cnccc1C(c2cncs2)(c3ccc4c(c(c(c(n4)Cl)Cc5ccc(cc5)Cl)Cl)c3)O",
}


@pytest.fixture(scope="module")
def data_paths():
    """Download preset data."""
    return download_preset(
        preset_name="synplanner-article", save_to="./tutorials/synplan_data"
    )


@pytest.fixture(scope="module")
def building_blocks(data_paths):
    """Load building blocks."""
    return load_building_blocks(
        data_paths["building_blocks"], standardize=False, silent=True
    )


@pytest.fixture(scope="module")
def reaction_rules(data_paths):
    """Load reaction rules."""
    return load_reaction_rules(data_paths["reaction_rules"])


@pytest.fixture(scope="module")
def policy_network(data_paths):
    """Initialize policy network."""
    return load_policy_function(weights_path=data_paths["ranking_policy"])


@pytest.fixture(scope="module")
def tree_config():
    """Get tree configuration."""
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


def run_clustering_workflow(
    target_smiles, building_blocks, reaction_rules, policy_network, tree_config
):
    """Helper function to run the complete clustering workflow."""

    # Create target molecule
    target_molecule = mol_from_smiles(
        target_smiles, clean2d=True, standardize=True, clean_stereo=True
    )

    # Create evaluation config and strategy
    eval_config = RolloutEvaluationConfig(
        policy_network=policy_network,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        min_mol_size=tree_config.min_mol_size,
        max_depth=tree_config.max_depth,
        normalize=tree_config.normalize_scores,
    )
    evaluator = load_evaluation_function(eval_config)

    # Create and solve tree
    tree = Tree(
        target=target_molecule,
        config=tree_config,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        expansion_function=policy_network,
        evaluation_function=evaluator,
    )

    # Solve tree
    tree_solved = False
    for solved, _ in tree:
        if solved:
            tree_solved = True
    tree._log_final_stats("completed")

    if not tree_solved:
        pytest.fail(f"Tree solving failed for molecule: {target_smiles}")

    # Get route CGRs
    all_route_cgrs = compose_all_route_cgrs(tree)
    all_sb_cgrs = compose_all_sb_cgrs(all_route_cgrs)

    # Perform clustering
    clusters = cluster_routes(all_sb_cgrs, use_strat=False)

    # Perform subclustering
    subclusters = subcluster_all_clusters(clusters, all_sb_cgrs, all_route_cgrs)
    return tree, clusters, subclusters


def calc_num_routes_subclusters(subclusters):
    """Calculate the total number of routes in subclusters."""
    count = 0
    for cluster in subclusters.values():
        for subcluster in cluster.values():
            count += len(subcluster["routes_data"])
    return count


@pytest.mark.integration
def test_simple_molecule_clustering(
    building_blocks, reaction_rules, policy_network, tree_config, caplog
):
    """Test clustering workflow with a simple molecule (Aspirin)."""
    caplog.set_level(logging.DEBUG)
    target_smiles = TEST_MOLECULES["simple"]
    _tree, clusters, subclusters = run_clustering_workflow(
        target_smiles, building_blocks, reaction_rules, policy_network, tree_config
    )
    # Verify clustering results
    assert len(clusters) > 0, "Should have at least one cluster"
    total_routes = sum(cluster["group_size"] for cluster in clusters.values())
    assert total_routes > 0, "Should have at least one route"

    # Verify subclustering results
    assert len(subclusters) > 0, "Should have at least one subcluster"
    total_subclusters = calc_num_routes_subclusters(subclusters)
    assert (
        total_subclusters == total_routes
    ), "Total subclusters should match total routes"
    assert sorted(subclusters.keys()) == sorted(
        clusters.keys()
    ), "Subcluster keys should match cluster keys"


@pytest.mark.integration
def test_medium_molecule_clustering(
    building_blocks, reaction_rules, policy_network, tree_config, caplog
):
    """Test clustering workflow with a medium complexity molecule (Capivasertib)."""
    caplog.set_level(logging.DEBUG)
    target_smiles = TEST_MOLECULES["medium"]
    _tree, clusters, subclusters = run_clustering_workflow(
        target_smiles, building_blocks, reaction_rules, policy_network, tree_config
    )

    # Verify clustering results
    assert len(clusters) > 0, "Should have at least one cluster"
    total_routes = sum(cluster["group_size"] for cluster in clusters.values())
    assert total_routes > 0, "Should have at least one route"

    # Verify subclustering results
    assert len(subclusters) > 0, "Should have at least one subcluster"
    total_subclusters = calc_num_routes_subclusters(subclusters)
    assert (
        total_subclusters == total_routes
    ), "Total subclusters should match total routes"
    assert sorted(subclusters.keys()) == sorted(
        clusters.keys()
    ), "Subcluster keys should match cluster keys"


@pytest.mark.integration
def test_complex_molecule_clustering(
    building_blocks, reaction_rules, policy_network, tree_config, caplog
):
    """Test clustering workflow with a complex molecule (Ibuprofen)."""
    caplog.set_level(logging.DEBUG)
    target_smiles = TEST_MOLECULES["complex"]
    _tree, clusters, subclusters = run_clustering_workflow(
        target_smiles, building_blocks, reaction_rules, policy_network, tree_config
    )

    # Verify clustering results
    assert len(clusters) > 0, "Should have at least one cluster"
    total_routes = sum(cluster["group_size"] for cluster in clusters.values())
    assert total_routes > 0, "Should have at least one route"

    # Verify subclustering results
    assert len(subclusters) > 0, "Should have at least one subcluster"
    total_subclusters = calc_num_routes_subclusters(subclusters)
    assert (
        total_subclusters == total_routes
    ), "Total subclusters should match total routes"
    assert sorted(subclusters.keys()) == sorted(
        clusters.keys()
    ), "Subcluster keys should match cluster keys"
