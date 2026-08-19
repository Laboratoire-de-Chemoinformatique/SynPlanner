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
This approach selects nodes more stochastically and reduces the computational cost.

**Evaluation-first strategy.** In the evaluation-first strategy, each newly created node immediately is evaluated with
the evaluation function, which lets the algorithm explore the tree more exhaustively. Although the node evaluation in the
evaluation-first strategy imposes an additional computational overhead, this problem can be overcome by the application
of fast evaluation functions, such as one approximated by a value neural network.

**Rollout evaluation.** The current implementation of rollout evaluation in ``SynPlanner``. For the given precursor,
a policy network predicts a set of applicable reaction rules sorted by their predicted probability. Then all reaction rules
are applied one by one and the first successfully applied reaction rule from this set generates new precursors. Then, the policy network
predicts the reaction rules for obtained precursors. This dissection proceeds until the stop criterion is reached with the corresponding value:

.. list-table::
   :header-rows: 1
   :widths: 60 20

   * - Condition
     - Return value
   * - Precursor is a building block
     - 1.0
   * - Reaction fails (no predicted rule is applicable)
     - −1.0
   * - Reaction succeeds, but precursors are not building blocks and the maximum tree depth would be exceeded
     - −0.5
   * - Reaction succeeds, but precursors are not building blocks and cannot be further expanded
     - −1.0

Alternative Search Algorithms
-----------------------------

In addition to the standard UCT (Upper Confidence Tree) algorithm, ``SynPlanner`` supports several alternative search algorithms
that can be selected via the ``algorithm`` configuration parameter.

**Nested Monte Carlo Search (NMCS).** NMCS is a recursive search algorithm introduced by Cazenave (2009) for
single-player optimization problems. Unlike iterative MCTS,
NMCS performs a deterministic, nested search:

- At **level 0**: A playout is performed using the configured mode (greedy, random, or policy-guided)
- At **level n** (n > 0): For each possible move, a level (n-1) search is performed, and the move leading to the best outcome is selected

Higher nesting levels let the algorithm look deeper into the consequences of each choice before committing. NMCS typically completes its search in a single iteration, exploring the tree exhaustively
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
- **Best-First Search (best_first):** Prioritises nodes with the highest evaluation scores
- **Beam Search (beam):** Like best-first, but expands only the top-k nodes at each level (controlled by ``beam_width``)

Running a search
----------------

A ``Tree`` does no work when constructed. Call ``run()`` to search to completion:

.. code-block:: python

   tree = Tree(target=target, config=tree_config, ...)
   tree.run()

``run()`` returns the tree, so it chains: ``routes = extract_routes(Tree(...).run())``.
Search stops at whichever limit comes first — ``max_iterations``, ``max_tree_size``,
``max_time``, or the first route found when ``stop_at_first`` is set.

The ``Tree`` is also iterable, and each iteration yields ``(is_solved, node_ids)``
for that step. Iterate it directly when you want to see the search progress or stop
on your own condition:

.. code-block:: python

   for is_solved, node_ids in tree:
       if is_solved and len(tree.stats.solution_iterations) >= 5:
           break   # five routes is enough for this target

Both run the same search. ``run()`` simply discards the per-iteration results, so
reach for the loop only when you actually need them — ``run()`` is the default.

.. note::

   Older code exhausts the iterator with ``list(tree)``. That still works, but it
   allocates a tuple per iteration only to discard them. Prefer ``run()``.

Target bond constraints
-----------------------

The Python ``Tree`` API can require or forbid disconnections of specific bonds
in the mapped target molecule. Pass an optional ``bonds_state`` mapping whose
keys are pairs of Chython atom-map numbers:

.. table::
   :widths: 15 85

   ===== ================================================================
   State Meaning
   ===== ================================================================
   ``0`` The bond is unconstrained. Omitting the key has the same effect.
   ``1`` Every accepted route must break this bond at some search step.
   ``2`` Candidate reactions that break this bond are rejected.
   ===== ================================================================

Bond direction is irrelevant: ``(7, 8)`` and ``(8, 7)`` normalize to the same
key. Obtain the numbers from the standardized Chython target, for example by
enabling atom-map labels before depicting it. Every specified pair must identify
a real bond in that target.

The numbers name **target-derived atoms**, not whichever atoms happen to carry
the same integers later. ``Tree`` seeds immutable provenance after target
canonicalization and carries it through each generated precursor. A reaction
product inherits a target identity only for an atom present in the precursor
being expanded. If Chython reuses a number from another fragment for a newly
introduced atom, that atom has no target identity and cannot satisfy or violate
a constraint accidentally.

This complete example downloads and loads the current GPS preset, standardizes
the target, applies the tutorial's required and frozen bonds, and runs the search:

.. code-block:: python

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

   paths = download_preset("synplanner-gps", save_to="synplan_data")
   building_blocks = load_building_blocks(
       paths["building_blocks"], standardize=False
   )
   reaction_rules = load_reaction_rules(paths["reaction_rules"])
   policy_function = load_policy_function(weights_path=paths["ranking_policy"])

   tree_config = TreeConfig(
       search_strategy="expansion_first",
       max_iterations=300,
       max_time=120,
       max_depth=9,
       min_mol_size=1,
       init_node_value=0.5,
       ucb_type="uct",
       c_ucb=0.1,
   )
   evaluation_function = load_evaluation_function(
       RolloutEvaluationConfig(
           policy_network=policy_function,
           reaction_rules=reaction_rules,
           building_blocks=building_blocks,
           min_mol_size=tree_config.min_mol_size,
           max_depth=tree_config.max_depth,
       )
   )

   target_molecule = mol_from_smiles(
       "N#CC1(c2ccc(NC(=O)c3cccnc3NCc3ccncc3)cc2)CCCC1",
       standardize=True,
       clean_stereo=True,
   )

   bonds_state = {
       (7, 8): 1,    # this bond must be disconnected somewhere in the route
       (16, 17): 2,  # this bond may never be disconnected
   }

   tree = Tree(
       target=target_molecule,
       config=tree_config,
       reaction_rules=reaction_rules,
       building_blocks=building_blocks,
       expansion_function=policy_function,
       evaluation_function=evaluation_function,
       bonds_state=bonds_state,
   ).run()

Frozen bonds are enforced per candidate reaction. If one candidate from a rule
breaks a frozen bond, SynPlanner skips that candidate and continues considering
later valid candidates from the same rule.

Constraint decisions use **adjacency only**. A target bond is broken when its
two target-derived endpoints are no longer connected in the generated products.
Changing the bond order does not count as a break. Replacing the element at a
mapped endpoint also does not count as a break while the two target identities
remain adjacent. Consequently, bond-order changes and mapped element
substitutions are allowed by state ``2`` and do not satisfy state ``1``.
Frozen-bond checks compare product adjacency directly; they do not compose a CGR.

Required breaks are route-level conditions. Each tree branch tracks which
state-``1`` bonds remain unresolved, so the search may make unrelated preliminary
disconnections first. A terminal node becomes a winning route only after *all*
required bonds have been broken. A required bond retained in a fragment already
recognized as a building block remains unresolved. Provenance and outstanding
requirements are part of constrained candidate deduplication, cycle detection,
and pruning, so structurally identical frontiers with different target ancestry
cannot be merged. ``None``, an empty mapping, and a state-``0``-only mapping keep
the original structure-only identity path.

``Tree`` validates the mapping when it is constructed and raises ``ValueError``
for malformed pair keys, non-integer or unsupported states, self-bonds,
conflicting states supplied through reversed keys, or selected bonds absent from
the target. ``tree.bonds_state`` returns a defensive normalized snapshot;
mutating that returned dictionary cannot change an existing search.

.. note::

   Bond constraints are currently available only through the Python ``Tree``
   API. They are not fields in planning YAML, CLI options, or arguments to the
   batch ``run_search`` helper. Rollout and value-network evaluation remain
   advisory and unchanged; the hard guarantees apply to generated branches and
   accepted winning routes.

See :doc:`Tutorial 19 <../user_guide/19_Bond_freeze_break>` for a complete
baseline-versus-constrained search and clustering comparison.

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

For a hands-on tutorial, see the `Tree Analysis notebook <06_Tree_Analysis.ipynb>`_.