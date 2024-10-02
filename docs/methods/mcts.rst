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