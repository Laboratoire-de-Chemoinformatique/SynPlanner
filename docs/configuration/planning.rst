.. _planning_config:

================================
Retrosynthetic planning
================================

The retrosynthesis planning algorithm can be adjusted by the configuration file.

Download example configuration
------------------------------

- GitHub: `configs/planning_standard.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/planning_standard.yaml>`_ — rollout evaluation, single ranking policy
- GitHub: `configs/planning_value.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/planning_value.yaml>`_ — value-network evaluation (adds ``node_evaluation: {evaluation_type: gcn}``; requires ``--value_network``)
- GitHub: `configs/planning_combined_policies.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/planning_combined_policies.yaml>`_ — combined filtering + ranking policy

Quickstart (CLI)
----------------

Run planning using the repository configuration in ``configs/planning_standard.yaml``:

.. code-block:: bash

   synplan planning \
     --config configs/planning_standard.yaml \
     --targets targets.smi \
     --reaction_rules reaction_rules.tsv \
     --building_blocks building_blocks_stand.smi \
     --policy_network policy_network.ckpt \
     --results_dir planning_results

**Configuration file**

.. code-block:: yaml

    tree:
      max_iterations: 100
      max_tree_size: 1000000
      max_time: 600
      max_depth: 6
      search_strategy: expansion_first
      ucb_type: uct
      c_ucb: 0.1
      backprop_type: muzero
      evaluation_agg: max
      exclude_small: True
      init_node_value: 0.5
      min_mol_size: 6
      epsilon: 0.0
      silent: True
    node_expansion:
      top_rules: 50
      rule_prob_threshold: 0.0
      priority_rules_fraction: 0.5

The ``tree:`` and ``node_expansion:`` sections are both **required** by
``synplan planning`` — omitting either raises ``KeyError`` (the only exception is
``node_expansion:``, which may be replaced by ``combined_policy:``, see below).
Keys absent *within* those sections fall back to the defaults of
:class:`~synplan.utils.config.TreeConfig` and
:class:`~synplan.utils.config.PolicyNetworkConfig`. The optional ones are listed
below; ``algorithm``, ``use_priority``, ``priority_rule_multiapplication`` and
the ``nmcs_*`` / ``lnmcs_ratio`` keys go under ``tree:``, and the optional
``node_evaluation:`` section takes ``evaluation_type``, ``normalize`` and
``score_function``.

.. warning::
    ``tree:`` and ``node_expansion:`` are validated with ``extra="forbid"``, so a
    misspelled key there is a loud error. ``node_evaluation:`` is **not** — it is
    read key by key with ``dict.get()``, so a misspelled key inside it is silently
    ignored and the default is used instead.

    Score normalisation at planning time is ``node_evaluation:normalize``.
    :class:`~synplan.utils.config.TreeConfig` also accepts ``normalize_scores``,
    but ``synplan planning`` never reads it — it only takes effect during
    ``synplan value_network_tuning``. Setting ``tree: normalize_scores: true`` in a
    planning config is accepted and does nothing.

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
    tree:use_priority                        Try curated priority rules passed via ``Tree(priority_rules=...)`` ahead of the policy on every expansion. Requires a non-empty ``priority_rules`` mapping. Defaults to False
    tree:priority_rule_multiapplication      Apply each priority rule to its product set until no new tuple is produced (BFS to fixpoint), instead of stopping at the first match. Affects priority rules only, not the policy. Defaults to False
    tree:evaluation_agg                      The way the evaluation scores are aggregated. Options are "max" (using the maximum score) and "average" (using the average score)
    tree:c_ucb                               The exploration-exploitation balance coefficient of the Upper Confidence Bound. Defaults to 0.1
    tree:beam_width                          Number of nodes expanded per step by ``algorithm: beam``. Ignored by every other algorithm. Defaults to 10
    node_evaluation:evaluation_type          The method used for node evaluation. Options are "rollout" (rollout simulations, default), "gcn" (value network), "random" (random number between 0 and 1), "policy" (policy probability), and "rdkit" (RDKit descriptor score, see ``score_function``)
    node_evaluation:normalize                Rescale evaluation scores to [0, 1]. Defaults to False. This is the only normalisation switch planning reads
    node_evaluation:score_function           Only for ``evaluation_type: rdkit``. One of "sascore" (default), "weight", "heavyAtomCount", "weightXsascore", "WxWxSAS"
    node_expansion:top_rules                 The maximum amount of rules to be selected for node expansion from the list of predicted reaction rules
    node_expansion:rule_prob_threshold       The reaction rules with predicted probability lower than this parameter will be discarded
    node_expansion:priority_rules_fraction   The fraction of priority rules in comparison to the regular rules (only for filtering policy)
    ======================================== ==========================================================

Combined filtering + ranking policy
-----------------------------------

Replacing ``node_expansion:`` with a ``combined_policy:`` section switches planning to
a weighted sum of a filtering and a ranking checkpoint
(``combined_logits = filtering_logits + ranking_weight * ranking_logits``, then
``softmax(combined_logits / temperature)``). Both checkpoints come from the config, so
``--policy_network`` is ignored for expansion — but it is still required by the CLI, and
it is the checkpoint used for ``evaluation_type: rollout``. Both checkpoints must have
been trained on the same rule set in the same order.

.. code-block:: yaml

    combined_policy:
      filtering_weights_path: "path/to/filtering_policy_network.ckpt"
      ranking_weights_path: "path/to/ranking_policy_network.ckpt"
      top_rules: 50
      rule_prob_threshold: 0.0
      ranking_weight: 1.0
      temperature: 1.0

.. table::
    :widths: 45 50

    ======================================== ==========================================================
    Parameter                                Description
    ======================================== ==========================================================
    combined_policy:filtering_weights_path   Path to the filtering policy checkpoint. Required
    combined_policy:ranking_weights_path     Path to the ranking policy checkpoint. Required
    combined_policy:top_rules                The maximum amount of rules returned. Defaults to 50
    combined_policy:rule_prob_threshold      Rules below this probability are discarded. Defaults to 0.0
    combined_policy:ranking_weight           Weight of the ranking logits. Values above 1.0 favour feasibility over applicability. Defaults to 1.0
    combined_policy:temperature              Softmax temperature. Values above 1.0 give softer, more exploratory distributions. Defaults to 1.0
    ======================================== ==========================================================

.. note::
    ``configs/combined_ranking_filtering_policy.yaml`` holds the ``combined_policy``
    block on its own, without the ``tree:`` wrapper. It is a Python-API config for
    :class:`~synplan.utils.config.CombinedPolicyConfig`; passing it to
    ``synplan planning --config`` raises ``KeyError: 'tree'``. Use
    ``configs/planning_combined_policies.yaml`` from the CLI.
