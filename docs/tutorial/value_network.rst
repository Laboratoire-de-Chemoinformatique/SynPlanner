.. _value_network:

Value network training
======================

This page explains how to train a value network in SynPlanner.

Introduction
---------------------------

**Node evaluation**. During the evaluation step, the value function (or evaluation function) is used to estimate the
retrosynthetic feasibility of newly created nodes. In SynPlanner, there are three types of evaluation functions implemented:

    * `random function` (assigns a random value between 0 and 1 to the new node). Mostly used as a baseline.
    * `rollout function` (default evaluation type in MCTS). In the current implementation it does a series of node expansions until it reaches some stope criterion (maximum simulation depth, discovered retrosynthetic route, etc.). Based on the simulation results it assigns the value between (-1 and 1) to the new node.
    * `value network` (instantly predicts the value between 0 and 1). The value neural network is trained on the data from planning simulations (performed with the previous version of the value network) including examples with precursors leading to the solutions and those which are part of the unsuccessful routes.

**Value network tuning**. The training set for the value neural network is generated from the simulations of planning sessions.
In the first iteration, the value network is initialized with random weights and is used for the initial retrosynthesis
planning session for N target molecules. Then, precursors that were part of a successful retrosynthesis path leading
to building block molecules are labeled with a positive label, and precursors that did not lead to building blocks are
labeled with a negative label. This generated training data is used to train the value network to better recognize precursors
leading to possible successful retrosynthetic paths. The trained value network is used in the next iteration of the simulated
planning session alternat-ed by the retraining of the value network until it reaches the acceptable accuracy of predictions.

Configuration
---------------------------
The network architecture and training hyperparameters can be adjusted in the training configuration yaml file below.

.. code-block:: yaml

    tree:
      max_iterations: 100
      max_tree_size: 10000
      max_time: 120
      max_depth: 9
      search_strategy: expansion_first
      ucb_type: uct
      c_ucb: 0.1
      backprop_type: muzero
      exclude_small: True
      min_mol_size: 6
      init_node_value: 0.5
      epsilon: 0
      silent: True
    node_evaluation:
      evaluation_type: rollout
      evaluation_agg: max
    node_expansion:
      top_rules: 50
      rule_prob_threshold: 0.0
      priority_rules_fraction: 0.5
    value_network:
      vector_dim: 512
      num_conv_layers: 5
      learning_rate: 0.0005
      dropout: 0.4
      num_epoch: 100
      batch_size: 1000
    tuning:
      batch_size: 100
      num_simulations: 1

**Configuration parameters**:

.. table::
    :widths: 45 10 50

    ======================================== ================ ==========================================================
    Parameter                                Default          Description
    ======================================== ================ ==========================================================
    tree:max_iterations                      100              The maximum number of iterations of the tree search algorithm
    tree:max_tree_size                       10000            The maximum number of nodes that can be created in the search tree
    tree:max_time                            240              The maximum time (in seconds) for the tree search execution
    tree:max_depth                           9                The maximum depth of the tree, controlling how far the search can go from the root node
    tree:ucb_type                            uct              The type of Upper Confidence Bound (UCB) statistics used in the tree search. Options include "puct" (predictive UCB), "uct" (standard UCB), and "value"
    tree:backprop_type                       muzero           The backpropagation method used during the tree search. Options are "muzero" (model-based approach) and "cumulative" (cumulative value approach)
    tree:search_strategy                     expansion_first  The strategy for navigating the tree. Options are "expansion_first" (prioritizing the expansion of new nodes) and "evaluation_first" (prioritizing the evaluation of new nodes)
    tree:exclude_small                       True             If True, excludes small molecules from the tree, typically focusing on more complex molecules
    tree:min_mol_size                        6                The minimum size of a molecule (the number of heavy atoms) to be considered in the search. Molecules smaller than this threshold are typically considered readily available building blocks
    tree:init_node_value                     0.5              The initial value for newly created nodes in the tree (for expansion_first search strategy)
    tree:epsilon                             0                This parameter is used in the epsilon-greedy strategy during the node selection, representing the probability of choosing a random action for exploration. A higher value leads to more exploration
    tree:silent                              True             If True, suppresses the progress logging of the tree search
    node_evaluation:evaluation_agg           max              The way the evaluation scores are aggregated. Options are "max" (using the maximum score of the child nodes) and "average" (using the average score of the child nodes)
    node_evaluation:evaluation_type          rollout          The method used for node evaluation. Options include "random" (random number between 0 and 1), "rollout" (using rollout simulations), and "gcn" (graph convolutional value network)
    node_expansion:top_rules                 50               The maximum amount of rules to be selected for node expansion from the list of predicted reaction rules
    node_expansion:rule_prob_threshold       0.0              The reaction rules with predicted probability lower than this parameter will be discarded
    node_expansion:priority_rules_fraction   0.5              The fraction of priority rules in comparison to the regular rules
    value_network:vector_dim                 512              The dimension of the hidden layers
    value_network:num_conv_layers            5                The number of convolutional layers
    value_network:dropout                    0.4              The dropout value
    value_network:learning_rate              0.0005           The learning rate
    value_network:num_epoch                  100              The number of training epochs
    value_network:batch_size                 1000             The size of the batch of input molecular graphs
    tuning:batch_size                        100              The size of the batch of target molecules used for planning simulation and value network update
    ======================================== ================ ==========================================================

CLI
---------------------------
Value network training can be performed with the below command.

**Important:** If you use your custom building blocks, be sure to canonicalize them before planning simulations in value network tuning.

.. code-block:: bash

    synplan building_blocks_canonicalizing --input building_blocks_init.smi --output building_blocks.smi
    synplan value_network_tuning --config tuning.yaml --targets targets.smi --reaction_rules reaction_rules.pickle --policy_network policy_network.ckpt --building_blocks building_blocks.smi --results_dir value_network

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``targets`` - the path to the file with target molecules for planning simulations.
    - ``reaction_rules`` - the path to the file with reactions rules.
    - ``building_blocks`` - the path to the file with building blocks.
    - ``policy_network`` - the path to the file with trained policy network (ranking or filtering policy network).
    - ``results_dir`` - the path to the directory where the trained value network will be to be stored.



