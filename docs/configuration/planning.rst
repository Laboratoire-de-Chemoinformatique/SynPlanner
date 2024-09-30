.. _planning:

================================
Retrosynthetic planning
================================

The retrosynthesis planning algorithm can be adjusted by the configuration yaml file:

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
      init_node_value: 0.5
      min_mol_size: 6
      epsilon: 0
      silent: True
    node_evaluation:
      evaluation_type: rollout
      evaluation_agg: max
    node_expansion:
      top_rules: 50
      rule_prob_threshold: 0.0
      priority_rules_fraction: 0.5

**Configuration parameters**:

.. table::
    :widths: 45 10 50

    ======================================== ================ ==========================================================
    Parameter                                Default          Description
    ======================================== ================ ==========================================================
    tree:max_iterations                      100              The maximum number of iterations the tree search algorithm will perform
    tree:max_tree_size                       10000            The maximum number of nodes that can be created in the search tree
    tree:max_time                            240              The maximum time (in seconds) for the tree search execution
    tree:max_depth                           9                The maximum depth of the tree, controlling how far the search can go from the root node
    tree:ucb_type                            uct              The type of Upper Confidence Bound (UCB) used in the tree search. Options include "puct" (predictive UCB), "uct" (standard UCB), and "value" (the initial node value)
    tree:backprop_type                       muzero           The backpropagation method used during the tree search. Options are "muzero" (model-based approach) and "cumulative" (cumulative reward approach)
    tree:search_strategy                     expansion_first  The strategy for navigating the tree. Options are "expansion_first" (prioritizing the expansion of new nodes) and "evaluation_first" (prioritizing the evaluation of existing nodes)
    tree:exclude_small                       True             If True, excludes small molecules from the tree, typically focusing on more complex molecules
    tree:min_mol_size                        6                The minimum size of a molecule (the number of heavy atoms) to be considered in the search. Molecules smaller than this threshold are typically considered readily available building blocks
    tree:init_node_value                     0.5              The initial value for newly created nodes in the tree (for expansion_first search strategy)
    tree:epsilon                             0                This parameter is used in the epsilon-greedy strategy during the node selection, representing the probability of choosing a random action for exploration. A higher value leads to more exploration
    tree:silent                              True             If True, suppresses the progress logging of the tree search
    node_evaluation:evaluation_agg           max              The way the evaluation scores are aggregated. Options are "max" (using the maximum score) and "average" (using the average score)
    node_evaluation:evaluation_type          rollout          The method used for node evaluation. Options include "random" (random number between 0 and 1), "rollout" (using rollout simulations), and "gcn" (graph convolutional networks)
    node_expansion:top_rules                 50               The maximum amount of rules to be selected for node expansion from the list of predicted reaction rules
    node_expansion:rule_prob_threshold       0.0              The reaction rules with predicted probability lower than this parameter will be discarded
    node_expansion:priority_rules_fraction   0.5              The fraction of priority rules in comparison to the regular rules (only for filtering policy)
    ======================================== ================ ==========================================================

Results analysis
---------------------------
After the retrosynthesis planning is finished, the planning results will be stored to the determined directory.
This directory will contain the following directories/files:

- `tree_search_stats.csv` – the CSV table with planning statistics.
- `extracted_routes.json` – the retrosynthesis routes extracted from the search trees. Can be used for route analysis with programming utils.
- `extracted_routes_html` – the directory containing html files with visualized retrosynthesis routes extracted from the search trees. Can be used for the visual analysis of the extracted retrosynthesis routes.