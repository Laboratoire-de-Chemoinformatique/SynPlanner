.. _planning:

================================
Retrosynthetic planning
================================

Currently, in ``SynPlanner`` there are different configurations of planning algorithms are available. The two reasonable and
recommended configurations are  - default and advanced configuration.

**Default planning**. This planning configuration includes the ranking policy network for node expansion,
rollout simulations for node evaluation, and expansion-first search strategy. This default configuration
requires only reaction data for training the policy network and is independent of the building block set
(they can be changed) because the rollout simulations can be considered as an online evaluation function
interacting with the given set of building blocks.

.. code-block:: yaml

    tree:
      search_strategy: expansion_first
    node_evaluation:
      evaluation_type: rollout

**Advanced planning**. This planning configuration includes the ranking policy network for node expansion,
value neural network for instant node evaluation, and evaluation-first strategy. This configuration requires reaction data
for training the policy network and molecule data for planning simulations in value network tuning.
Because the building block set is used in planning simulations, the value network should be returned
if the building block set is changed. The evaluation-first strategy supposes more computations,
but the total time of search is partially reduced by instant predictions of node values by value neural network
instead of expansive rollout simulations.

.. code-block:: yaml

    tree:
      search_strategy: evaluation_first
    node_evaluation:
      evaluation_type: gcn

**Conclusion**. In general, the advanced planning algorithm is slower than the default (around 2x slow down),
but can be considered more powerful (because of more exhaustive search tree exploration) and may help
if the default planning algorithm fails to find a solution for the given molecule.
