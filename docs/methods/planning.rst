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
Because the building block set is used in planning simulations, the value network should be retrained
if the building block set is changed. The evaluation-first strategy supposes more computations,
but the total time of search is partially reduced by instant predictions of node values by value neural network
instead of expansive rollout simulations.

.. code-block:: yaml

    tree:
      search_strategy: evaluation_first
    node_evaluation:
      evaluation_type: gcn

**Conclusion**. The advanced algorithm is roughly 2x slower but explores the search tree more exhaustively.

Targets that are already purchasable
------------------------------------

``run_search`` checks each target against the building blocks before it builds a
tree. A target the catalogue already sells is named on the console and recorded
as ``target_in_stock`` in the statistics CSV, and no search is run for it.

The test is exact catalogue membership rather than
:meth:`~synplan.chem.precursor.Precursor.is_building_block`. That method also
accepts anything at or below ``min_mol_size``, which is right for a precursor --
a fragment too small to bother decomposing is treated as available -- and wrong
for a target, where a small molecule is small rather than purchasable.

This matters more than it sounds. Catalogues assembled from supplier data carry
finished drugs: paracetamol, celecoxib, sildenafil and paclitaxel are all in the
shipped eMolecules set. Without the check a run spends its budget planning them
and reports hundreds of routes, with nothing in the console output, the
statistics or the route HTML saying the molecule was on the shelf the whole
time.
