import logging
import pytest
from synplan.chem.utils import mol_from_smiles
from synplan.chem.reaction_routes.route_cgr import (
    compose_all_route_cgrs,
    compose_all_sb_cgrs,
)
from synplan.chem.reaction_routes.clustering import (
    cluster_routes,
    subcluster_all_clusters,
)
from synplan.utils.loading import (
    load_building_blocks,
    load_reaction_rules,
    load_policy_function,
)
from synplan.mcts.tree import Tree
from synplan.utils.config import TreeConfig, RolloutEvaluationConfig
from synplan.utils.loading import download_selected_files, load_evaluation_function

# Test molecules with different complexity levels
TEST_MOLECULES = {
    "simple": "CCNc1nc(Sc2ccc(C)cc2)cc(C(F)(F)F)n1",
    "medium": "c1cc(ccc1C2=NN(C(C=C2)=N)CCCC(O)=O)OCc3cc([N+]([O-])=O)ccc3",
    "complex": "c1cnccc1C(c2cncs2)(c3ccc4c(c(c(c(n4)Cl)Cc5ccc(cc5)Cl)Cl)c3)O",
}


@pytest.fixture(scope="module")
def data_folder():
    """Load data."""
    assets = [
        ("building_blocks", "building_blocks_em_sa_ln.smi"),
        ("uspto", "uspto_reaction_rules.pickle"),
        ("uspto/weights", "ranking_policy_network.ckpt"),
    ]

    folder = download_selected_files(
        files_to_get=assets,
        save_to="./tutorials/synplan_data",
        extract_zips=True,
    )
    return folder


@pytest.fixture(scope="module")
def building_blocks(data_folder):
    """Load building blocks."""
    building_blocks_path = data_folder.joinpath(
        "building_blocks/building_blocks_em_sa_ln.smi"
    )
    return load_building_blocks(building_blocks_path, standardize=False, silent=True)


@pytest.fixture(scope="module")
def reaction_rules(data_folder):
    """Load reaction rules."""
    reaction_rules_path = data_folder.joinpath("uspto/uspto_reaction_rules.pickle")
    return load_reaction_rules(reaction_rules_path)


@pytest.fixture(scope="module")
def policy_network(data_folder):
    """Initialize policy network."""
    ranking_policy_network = data_folder.joinpath(
        "uspto/weights/ranking_policy_network.ckpt"
    )
    return load_policy_function(weights_path=ranking_policy_network)


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
    l = 0
    for cluster in subclusters.values():
        for subcluster in cluster.values():
            l += len(subcluster["routes_data"])
    return l


@pytest.mark.integration
def test_simple_molecule_clustering(
    building_blocks, reaction_rules, policy_network, tree_config, caplog
):
    """Test clustering workflow with a simple molecule (Aspirin)."""
    caplog.set_level(logging.DEBUG)
    target_smiles = TEST_MOLECULES["simple"]
    tree, clusters, subclusters = run_clustering_workflow(
        target_smiles, building_blocks, reaction_rules, policy_network, tree_config
    )
    # Verify clustering results
    expected_clusters = ["1.1", "2.1", "2.2", "3.1", "3.2", "3.3", "4.1"]
    assert len(clusters) > 0, "Should have at least one cluster"
    assert len(clusters) == len(expected_clusters), "Should have 7 clusters"
    total_routes = sum(cluster["group_size"] for cluster in clusters.values())
    assert total_routes > 0, "Should have at least one route"
    assert (
        len(set(tree.winning_nodes)) == total_routes == 158
    ), "Total routes should match winning nodes and 158 routes"
    assert (
        list(clusters.keys()) == expected_clusters
    ), f"SubCluster keys should be {expected_clusters}"

    # Verify subclustering results
    assert len(subclusters) > 0, "Should have at least one subcluster"
    total_subclusters = calc_num_routes_subclusters(subclusters)
    assert total_subclusters > 0, "Should have at least one subcluster group"
    assert (
        len(set(tree.winning_nodes)) == total_subclusters == 158
    ), "Total subclusters should match winning nodes"
    assert (
        list(subclusters.keys()) == expected_clusters
    ), f"SubCluster keys should be {expected_clusters}"


@pytest.mark.integration
def test_medium_molecule_clustering(
    building_blocks, reaction_rules, policy_network, tree_config, caplog
):
    """Test clustering workflow with a medium complexity molecule (Capivasertib)."""
    caplog.set_level(logging.DEBUG)
    target_smiles = TEST_MOLECULES["medium"]
    tree, clusters, subclusters = run_clustering_workflow(
        target_smiles, building_blocks, reaction_rules, policy_network, tree_config
    )

    # Verify clustering results
    expected_clusters = [
        "2.1",
        "2.2",
        "3.1",
        "3.2",
        "3.3",
        "3.4",
        "3.5",
        "3.6",
        "4.1",
        "4.2",
        "5.1",
        "6.1",
    ]
    assert len(clusters) > 0, "Should have at least one cluster"
    assert len(clusters) == len(
        expected_clusters
    ), f"Should have {len(expected_clusters)} clusters"
    total_routes = sum(cluster["group_size"] for cluster in clusters.values())
    assert total_routes > 0, "Should have at least one route"
    assert (
        len(set(tree.winning_nodes)) == total_routes
    ), "Total routes should match winning nodes"
    resulted_clusters = sorted(
        clusters.keys(), key=lambda s: [int(part) for part in s.split(".")]
    )
    assert (
        resulted_clusters == expected_clusters
    ), f"SubCluster keys should be {expected_clusters}"

    # Verify subclustering results
    assert len(subclusters) > 0, "Should have at least one subcluster"
    total_subclusters = calc_num_routes_subclusters(subclusters)
    assert total_subclusters > 0, "Should have at least one subcluster group"
    assert (
        len(set(tree.winning_nodes)) == total_subclusters
    ), "Total subclusters should match winning nodes"
    resulted_subclusters = sorted(
        subclusters.keys(), key=lambda s: [int(part) for part in s.split(".")]
    )
    assert (
        resulted_subclusters == expected_clusters
    ), f"SubCluster keys should be {expected_clusters}"


@pytest.mark.integration
def test_complex_molecule_clustering(
    building_blocks, reaction_rules, policy_network, tree_config, caplog
):
    """Test clustering workflow with a complex molecule (Ibuprofen)."""
    caplog.set_level(logging.DEBUG)
    target_smiles = TEST_MOLECULES["complex"]
    tree, clusters, subclusters = run_clustering_workflow(
        target_smiles, building_blocks, reaction_rules, policy_network, tree_config
    )

    # Verify clustering results
    expected_clusters = [
        "3.1",
        "3.2",
        "4.1",
        "4.2",
        "4.3",
        "4.4",
        "4.5",
        "4.6",
        "4.7",
        "5.1",
        "5.2",
    ]
    assert len(clusters) > 0, "Should have at least one cluster"
    assert len(clusters) == len(
        expected_clusters
    ), f"Should have {len(expected_clusters)} clusters"
    total_routes = sum(cluster["group_size"] for cluster in clusters.values())
    assert total_routes > 0, "Should have at least one route"
    assert (
        len(set(tree.winning_nodes)) == total_routes
    ), "Total routes should match winning nodes"
    resulted_clusters = sorted(
        clusters.keys(), key=lambda s: [int(part) for part in s.split(".")]
    )
    assert (
        resulted_clusters == expected_clusters
    ), f"SubCluster keys should be {expected_clusters}"

    # Verify subclustering results
    assert len(subclusters) > 0, "Should have at least one subcluster"
    total_subclusters = calc_num_routes_subclusters(subclusters)
    assert total_subclusters > 0, "Should have at least one subcluster group"
    assert (
        len(set(tree.winning_nodes)) == total_subclusters
    ), "Total subclusters should match winning nodes"
    resulted_subclusters = sorted(
        subclusters.keys(), key=lambda s: [int(part) for part in s.split(".")]
    )
    assert (
        resulted_subclusters == expected_clusters
    ), f"SubCluster keys should be {expected_clusters}"
