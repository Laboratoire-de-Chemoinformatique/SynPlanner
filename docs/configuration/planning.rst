.. _planning_config:

================================
Retrosynthetic planning
================================

The retrosynthesis planning algorithm can be adjusted by the configuration file.

Download example configuration
------------------------------

- GitHub: `configs/planning.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/planning.yaml>`_

Quickstart (CLI)
----------------

Run planning using the repository configuration in ``configs/planning.yaml``:

.. code-block:: bash

   synplan planning \
     --config configs/planning.yaml \
     --targets targets.smi \
     --reaction_rules reaction_rules.pickle \
     --building_blocks building_blocks_stand.smi \
     --policy_network policy_network.ckpt \
     --results_dir planning_results

**Configuration file**

.. code-block:: yaml

    tree:
      algorithm: uct
      max_iterations: 100
      max_tree_size: 10000
      max_time: 120
      max_depth: 9
      search_strategy: expansion_first
      ucb_type: uct
      c_ucb: 0.1
      backprop_type: muzero
      exclude_small: True
      init_node_value: 0.5
      min_mol_size: 6
      epsilon: 0
      silent: True
      # NMCS-specific parameters (used when algorithm is "nmcs" or "lazy_nmcs")
      nmcs_level: 2
      nmcs_playout_mode: greedy
      lnmcs_ratio: 0.2
    node_evaluation:
      evaluation_type: rollout
      evaluation_agg: max
    node_expansion:
      top_rules: 50
      rule_prob_threshold: 0.0
      priority_rules_fraction: 0.5

**Configuration parameters**

.. table::
    :widths: 45 50

    ======================================== ==========================================================
    Parameter                                Description
    ======================================== ==========================================================
    tree:algorithm                           The search algorithm to use. Options are "uct" (Upper Confidence Tree, default), "nmcs" (Nested Monte Carlo Search), "lazy_nmcs" (Lazy NMCS with pruning), "best_first", "breadth_first", and "beam"
    tree:max_iterations                      The maximum number of iterations the tree search algorithm will perform
    tree:max_tree_size                       The maximum number of nodes that can be created in the search tree
    tree:max_time                            The maximum time (in seconds) for the tree search execution
    tree:max_depth                           The maximum depth of the tree, controlling how far the search can go from the root node
    tree:ucb_type                            The type of Upper Confidence Bound (UCB) used in the tree search. Options include "puct" (predictive UCB), "uct" (standard UCB), and "value" (the initial node value)
    tree:backprop_type                       The backpropagation method used during the tree search. Options are "muzero" (model-based approach) and "cumulative" (cumulative reward approach)
    tree:search_strategy                     The strategy for navigating the tree. Options are "expansion_first" (prioritizing the expansion of new nodes) and "evaluation_first" (prioritizing the evaluation of existing nodes)
    tree:exclude_small                       If True, excludes small molecules from the tree, typically focusing on more complex molecules
    tree:min_mol_size                        The minimum size of a molecule (the number of heavy atoms) to be considered in the search. Molecules smaller than this threshold are typically considered readily available building blocks
    tree:init_node_value                     The initial value for newly created nodes in the tree (for expansion_first search strategy)
    tree:epsilon                             This parameter is used in the epsilon-greedy strategy during the node selection, representing the probability of choosing a random action for exploration. A higher value leads to more exploration
    tree:silent                              If True, suppresses the progress logging of the tree search
    tree:nmcs_level                          Nesting level for NMCS and LazyNMCS algorithms. Higher levels provide more thorough search but are more computationally expensive. Defaults to 2
    tree:nmcs_playout_mode                   Playout mode for NMCS base-level rollouts. Options are "greedy" (best value), "random", or "policy" (best policy probability). Defaults to "greedy"
    tree:lnmcs_ratio                         Pruning percentile for LazyNMCS algorithm. Only candidates scoring above this percentile threshold are explored. Value in range [0.0, 1.0]. Defaults to 0.2
    node_evaluation:evaluation_agg           The way the evaluation scores are aggregated. Options are "max" (using the maximum score) and "average" (using the average score)
    node_evaluation:evaluation_type          The method used for node evaluation. Options include "random" (random number between 0 and 1), "rollout" (using rollout simulations), and "gcn" (graph convolutional networks)
    node_expansion:top_rules                 The maximum amount of rules to be selected for node expansion from the list of predicted reaction rules
    node_expansion:rule_prob_threshold       The reaction rules with predicted probability lower than this parameter will be discarded
    node_expansion:priority_rules_fraction   The fraction of priority rules in comparison to the regular rules (only for filtering policy)
    ======================================== ==========================================================
