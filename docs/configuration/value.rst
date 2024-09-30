.. _value:

================
Value network
================

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
    reinforcement:
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
    reinforcement:batch_size                 100              The size of the batch of target molecules used for planning simulation and value network update
    ======================================== ================ ==========================================================
