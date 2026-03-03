.. _mcts:

================================
Monte-Carlo tree search
================================

The retrosynthesis planning in ``SynPlanner`` is executed with the MCTS algorithm. The nodes in the MCTS algorithm are expanded
by the expansion function predicting reaction rules applicable to the current precursor and evaluated by
the evaluation function navigating the tree exploration in the promising directions. The tree search is limited
by tree parameters: number of iterations, time of the search, and size of the tree (total number of nodes).
Retrosynthesis planning in ``SynPlanner`` can be performed using two search strategies:
the evaluation-first and the expansion-first strategy.

**Expansion-first strategy.** In the expansion-first strategy, each newly created node is assigned a predefined constant value.
This approach is characterized by a more stochastic selection of nodes for expansion but allows for a reduction in the
computational resources.

**Evaluation-first strategy.** In the evaluation-first strategy, each newly created node immediately is evaluated with
the evaluation function, which allows for more exhaustive tree exploration. Although the node evaluation in the
evaluation-first strategy imposes an additional computational overhead, this problem can be overcome by the application
of fast evaluation functions, such as one approximated by a value neural network.

**Rollout evaluation.** The current implementation of rollout evaluation in ``SynPlanner``. For the given precursor,
a policy network predicts a set of applicable reaction rules sorted by their predicted probability. Then all reaction rules
are applied one by one and the first successfully applied reaction rule from this set generates new precursors. Then, the policy network
predicts the reaction rules for obtained precursors. This dissection proceeds until the stop criterion is reached with the corresponding value:

    - If the precursor is a building_block, return 1.0
    - If the reaction is not successful, return -1.0 (all predicted reaction rules are not applicable).
    - If the reaction is successful, but the generated precursors are not the building_blocks and cannot be generated without exceeding the maximum tree depth, return -0.5.
    - If the reaction is successful, but the precursors are not the building_blocks and cannot be generated, return -1.0.

Alternative Search Algorithms
-----------------------------

In addition to the standard UCT (Upper Confidence Tree) algorithm, ``SynPlanner`` supports several alternative search algorithms
that can be selected via the ``algorithm`` configuration parameter.

**Nested Monte Carlo Search (NMCS).** NMCS is a recursive search algorithm introduced by Cazenave (2009) for
single-player optimization problems. Unlike iterative MCTS,
NMCS performs a deterministic, nested search:

- At **level 0**: A playout is performed using the configured mode (greedy, random, or policy-guided)
- At **level n** (n > 0): For each possible move, a level (n-1) search is performed, and the move leading to the best outcome is selected

The key insight is that higher nesting levels allow the algorithm to make more informed decisions by looking deeper into
the consequences of each choice. NMCS typically completes its search in a single iteration, exploring the tree exhaustively
according to the nesting level.

Configuration parameters:

- ``nmcs_level``: Controls the nesting depth (default: 2). Higher values provide more thorough search but increase computation time exponentially.
- ``nmcs_playout_mode``: Controls how level-0 playouts select moves. Options are "greedy" (highest value), "random", or "policy" (highest policy probability).

**Lazy Nested Monte Carlo Search (LazyNMCS).** LazyNMCS is an extension of NMCS that uses percentile-based pruning to
reduce the branching factor. For each decision point:

1. All candidate moves are quickly evaluated using greedy playouts
2. Only moves scoring above a configurable percentile threshold are explored with full NMCS recursion

This approach reduces computation time while maintaining search quality by focusing on the most promising branches.

Configuration parameters:

- ``lnmcs_ratio``: The percentile threshold for pruning (default: 0.2). A value of 0.2 means only candidates in the top 80% are explored.

**Note on iteration behavior:** Unlike UCT which can be run for multiple iterations to progressively refine the search,
NMCS and LazyNMCS are designed as one-shot algorithms that complete their search in the first iteration. The ``max_iterations``
parameter should be set to 1 when using these algorithms.

Other Algorithms
----------------

``SynPlanner`` also supports simpler search strategies:

- **Breadth-First Search (breadth_first):** Explores nodes level by level in FIFO order
- **Best-First Search (best_first):** Prioritizes nodes with highest evaluation scores
- **Beam Search (beam):** Like best-first but expands only the top-k nodes at each level (controlled by ``beam_width``)

Tree Analytics
--------------

After a search completes, the ``Tree`` object provides statistics about policy performance, search dynamics,
and tree structure. These are useful for debugging failed searches, evaluating policy quality, and comparing
different configurations.

**Search counters** (``tree.stats``). Lightweight counters collected during search with zero overhead:

.. table::
    :widths: 40 60

    ============================================= ===========================================================
    Counter                                       Description
    ============================================= ===========================================================
    ``expansion_calls``                           Number of nodes the search attempted to expand
    ``expansion_successes``                       Expansions that produced at least one child
    ``total_rules_tried``                         Total reaction rules applied across all expansions
    ``total_rules_succeeded``                     Rules that produced valid products
    ``dead_end_nodes``                            Non-root nodes where no rule produced children
    ``first_solution_iteration``                  Iteration when first solved route was found
    ``first_solution_time``                       Wall-clock seconds to first solution
    ``routes_found_at``                           List of (iteration, time) pairs for each route discovery
    ============================================= ===========================================================

**Analysis methods**. Computed lazily on demand after search:

.. table::
    :widths: 40 60

    ============================================= ===========================================================
    Method                                        Description
    ============================================= ===========================================================
    ``rule_applicability_rate()``                  Fraction of tried rules that produced valid products (0–1)
    ``winning_rule_ranks()``                       Rank of the winning rule among siblings at each step of each solved route
    ``branching_profile()``                        Mean branching factor per depth level (expanded nodes only)
    ``route_details(node_id)``                     Per-step breakdown of a specific route (rule, prob, value, visits)
    ``to_stats_dict()``                            Flat dict with all metrics for CSV/JSON export
    ============================================= ===========================================================

**Interpreting results**:

- **Rule applicability rate** > 0.5 indicates the policy predicts mostly applicable rules.
  Below 0.2 suggests the policy may need retraining or the rule set is too specific.
- **Winning rule rank** close to 1 means the policy's top predictions lead to solutions.
  Higher ranks indicate the search had to explore further down the prediction list.
- **First solution iteration** relative to total iterations shows search efficiency.
  Late first solutions suggest increasing ``max_iterations`` or tuning ``c_ucb``.
- **Dead-end nodes** indicate nodes where no predicted rules produced children.
  High dead-end rates may point to limited building blocks or overly specific rules.

For a hands-on tutorial, see the `Tree Analysis notebook <10_Tree_Analysis.ipynb>`_.