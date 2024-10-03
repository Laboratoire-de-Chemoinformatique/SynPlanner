"""Module containing functions for running value network tuning with reinforcement learning
approach."""

import os
import random
from collections import defaultdict
from pathlib import Path
from random import shuffle
from typing import Dict, List

import torch
from CGRtools.containers import MoleculeContainer
from pytorch_lightning import Trainer
from torch.utils.data import random_split
from torch_geometric.data.lightning import LightningDataset

from synplan.chem.precursor import compose_precursors
from synplan.mcts.evaluation import ValueNetworkFunction
from synplan.mcts.expansion import PolicyNetworkFunction
from synplan.mcts.tree import Tree
from synplan.ml.networks.value import ValueNetwork
from synplan.ml.training.preprocessing import ValueNetworkDataset
from synplan.utils.config import (
    PolicyNetworkConfig,
    TuningConfig,
    TreeConfig,
    ValueNetworkConfig,
)
from synplan.utils.files import MoleculeReader
from synplan.utils.loading import (
    load_building_blocks,
    load_reaction_rules,
    load_value_net,
)
from synplan.utils.logging import DisableLogger, HiddenPrints


def create_value_network(value_config: ValueNetworkConfig) -> ValueNetwork:
    """Creates the initial value network.

    :param value_config: The value network configuration.
    :return: The valueNetwork to be trained/tuned.
    """

    weights_path = Path(value_config.weights_path)
    value_network = ValueNetwork(
        vector_dim=value_config.vector_dim,
        batch_size=value_config.batch_size,
        dropout=value_config.dropout,
        num_conv_layers=value_config.num_conv_layers,
        learning_rate=value_config.learning_rate,
    )

    with DisableLogger(), HiddenPrints():
        trainer = Trainer()
        trainer.strategy.connect(value_network)
        trainer.save_checkpoint(weights_path)

    return value_network


def create_targets_batch(
    targets: List[MoleculeContainer], batch_size: int
) -> List[List[MoleculeContainer]]:
    """Creates the targets batches for planning simulations and value network tuning.

    :param targets: The list of target molecules.
    :param batch_size: The size of each target batch.
    :return: The list of lists corresponding to each target batch.
    """

    num_targets = len(targets)
    batch_splits = list(
        range(num_targets // batch_size + int(bool(num_targets % batch_size)))
    )

    if int(num_targets / batch_size) == 0:
        print(f"1 batch were created with {num_targets} molecules")
    else:
        print(
            f"{len(batch_splits)} batches were created with {batch_size} molecules each"
        )

    targets_batch_list = []
    for batch_id in batch_splits:
        batch_slices = [
            i
            for i in range(batch_id * batch_size, (batch_id + 1) * batch_size)
            if i < len(targets)
        ]
        targets_batch_list.append([targets[i] for i in batch_slices])

    return targets_batch_list


def run_tree_search(
    target: MoleculeContainer,
    tree_config: TreeConfig,
    policy_config: PolicyNetworkConfig,
    value_config: ValueNetworkConfig,
    reaction_rules_path: str,
    building_blocks_path: str,
) -> Tree:
    """Runs tree search for the given target molecule.

    :param target: The target molecule.
    :param tree_config: The planning configuration of tree search.
    :param policy_config: The policy network configuration.
    :param value_config: The value network configuration.
    :param reaction_rules_path: The path to the file with reaction rules.
    :param building_blocks_path: The path to the file with building blocks.
    :return: The built search tree for the given molecule.
    """

    # policy and value function loading
    policy_function = PolicyNetworkFunction(policy_config=policy_config)
    value_function = ValueNetworkFunction(weights_path=value_config.weights_path)
    reaction_rules = load_reaction_rules(reaction_rules_path)
    building_blocks = load_building_blocks(building_blocks_path, standardize=True)

    # initialize tree
    tree_config.evaluation_type = "gcn"
    tree_config.silent = True
    tree = Tree(
        target=target,
        config=tree_config,
        reaction_rules=reaction_rules,
        building_blocks=building_blocks,
        expansion_function=policy_function,
        evaluation_function=value_function,
    )
    tree._tqdm = False

    # remove target from buildings blocs
    if str(target) in tree.building_blocks:
        tree.building_blocks.remove(str(target))

    # run tree search
    _ = list(tree)

    return tree


def extract_tree_precursor(tree_list: List[Tree]) -> Dict[str, float]:
    """Takes the built tree and extracts the precursor for value network tuning. The
    precursor from found retrosynthetic routes are labeled as a positive class and precursor
    from not solved routes are labeled as a negative class.

    :param tree_list: The list of built search trees.

    :return: The dictionary with the precursor SMILES and its class (positive - 1 or negative - 0).
    """
    extracted_precursor = defaultdict(float)
    for tree in tree_list:
        for idx, node in tree.nodes.items():
            # add solved nodes to set
            if node.is_solved():
                parent = idx
                while parent and parent != 1:
                    composed_smi = str(
                        compose_precursors(tree.nodes[parent].new_precursors)
                    )
                    extracted_precursor[composed_smi] = 1.0
                    parent = tree.parents[parent]
            else:
                composed_smi = str(compose_precursors(tree.nodes[idx].new_precursors))
                extracted_precursor[composed_smi] = 0.0

    # shuffle extracted precursor
    processed_keys = list(extracted_precursor.keys())
    shuffle(processed_keys)
    extracted_precursor = {i: extracted_precursor[i] for i in processed_keys}

    return extracted_precursor


def balance_extracted_precursor(extracted_precursor):
    extracted_precursor_balanced = {}
    neg_list = [i for i, j in extracted_precursor.items() if j == 0]
    for k, v in extracted_precursor.items():
        if v == 1:
            extracted_precursor_balanced[k] = v
        if len(extracted_precursor_balanced) < len(neg_list):
            neg_list.pop(random.choice(range(len(neg_list))))
    return extracted_precursor_balanced


def create_updating_set(
    extracted_precursor: Dict[str, float], batch_size: int = 1
) -> LightningDataset:
    """Creates the value network updating dataset from precursor extracted from the planning
    simulation.

    :param extracted_precursor: The dictionary with the extracted precursor and their
        labels.
    :param batch_size: The size of the batch in value network updating.
    :return: A LightningDataset object, which contains the tuning set for value network
        tuning.
    """

    extracted_precursor = balance_extracted_precursor(extracted_precursor)

    full_dataset = ValueNetworkDataset(extracted_precursor)
    train_size = int(0.6 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_set, val_set = random_split(
        full_dataset, [train_size, val_size], torch.Generator().manual_seed(42)
    )

    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")

    return LightningDataset(
        train_set, val_set, batch_size=batch_size, pin_memory=True, drop_last=True
    )


def tune_value_network(
    datamodule: LightningDataset, value_config: ValueNetworkConfig
) -> None:
    """Trains the value network using a given tuning data and saves the trained neural
    network.

    :param datamodule: The tuning dataset (LightningDataset).
    :param value_config: The value network configuration.
    :return: None.
    """

    current_weights = value_config.weights_path
    value_network = load_value_net(ValueNetwork, current_weights)

    with DisableLogger(), HiddenPrints():
        trainer = Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=value_config.num_epoch,
            enable_checkpointing=False,
            logger=False,
            gradient_clip_val=1.0,
            enable_progress_bar=False,
        )

        trainer.fit(value_network, datamodule)
        val_score = trainer.validate(value_network, datamodule.val_dataloader())[0]
        trainer.save_checkpoint(current_weights)

    print(f"Value network balanced accuracy: {val_score['val_balanced_accuracy']}")


def run_training(
    extracted_precursor: Dict[str, float] = None,
    value_config: ValueNetworkConfig = None,
) -> None:
    """Runs the training stage in value network tuning.

    :param extracted_precursor: The precursor extracted from the planing simulations.
    :param value_config: The value network configuration.
    :return: None.
    """

    # create training set
    training_set = create_updating_set(
        extracted_precursor=extracted_precursor, batch_size=value_config.batch_size
    )

    # retrain value network
    tune_value_network(datamodule=training_set, value_config=value_config)


def run_planning(
    targets_batch: List[MoleculeContainer],
    tree_config: TreeConfig,
    policy_config: PolicyNetworkConfig,
    value_config: ValueNetworkConfig,
    reaction_rules_path: str,
    building_blocks_path: str,
    targets_batch_id: int,
):
    """Performs planning stage (tree search) for target molecules and save extracted
    from built trees precursor for further tuning the value network in the training stage.

    :param targets_batch:
    :param tree_config:
    :param policy_config:
    :param value_config:
    :param reaction_rules_path:
    :param building_blocks_path:
    :param targets_batch_id:
    """
    from tqdm import tqdm

    print(f"\nProcess batch number {targets_batch_id}")
    tree_list = []
    tree_config.silent = False
    for target in tqdm(targets_batch):

        try:
            tree = run_tree_search(
                target=target,
                tree_config=tree_config,
                policy_config=policy_config,
                value_config=value_config,
                reaction_rules_path=reaction_rules_path,
                building_blocks_path=building_blocks_path,
            )
            tree_list.append(tree)

        except Exception as e:
            print(e)
            continue

    num_solved = sum([len(i.winning_nodes) > 0 for i in tree_list])
    print(f"Planning is finished with {num_solved} solved targets")

    return tree_list


def run_updating(
    targets_path: str,
    tree_config: TreeConfig,
    policy_config: PolicyNetworkConfig,
    value_config: ValueNetworkConfig,
    reinforce_config: TuningConfig,
    reaction_rules_path: str,
    building_blocks_path: str,
    results_root: str = None,
) -> None:
    """Performs updating of value network.

    :param targets_path: The path to the file with target molecules.
    :param tree_config: The search tree configuration.
    :param policy_config: The policy network configuration.
    :param value_config: The value network configuration.
    :param reinforce_config: The value network tuning configuration.
    :param reaction_rules_path: The path to the file with reaction rules.
    :param building_blocks_path: The path to the file with building blocks.
    :param results_root: The path to the directory where trained value network will be
        saved.
    :return: None.
    """

    # create results root folder
    results_root = Path(results_root)
    if not results_root.exists():
        results_root.mkdir()

    # load targets list
    with MoleculeReader(targets_path) as targets:
        targets = list(targets)

    # create value neural network
    value_config.weights_path = os.path.join(results_root, "value_network.ckpt")
    create_value_network(value_config)

    # create targets batch
    targets_batch_list = create_targets_batch(
        targets, batch_size=reinforce_config.batch_size
    )

    # run value network tuning
    for batch_id, targets_batch in enumerate(targets_batch_list, start=1):

        # start tree planning simulation for batch of targets
        tree_list = run_planning(
            targets_batch=targets_batch,
            tree_config=tree_config,
            policy_config=policy_config,
            value_config=value_config,
            reaction_rules_path=reaction_rules_path,
            building_blocks_path=building_blocks_path,
            targets_batch_id=batch_id,
        )

        # extract pos and neg precursor from the list of built trees
        extracted_precursor = extract_tree_precursor(tree_list)

        # train value network for extracted precursor
        run_training(extracted_precursor=extracted_precursor, value_config=value_config)
