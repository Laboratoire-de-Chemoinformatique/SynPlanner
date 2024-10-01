.. _value:

================
Value network
================

**Node evaluation**. During the evaluation step, the value function (or evaluation function) is used to estimate the
retrosynthetic feasibility of newly created nodes. In ``SynPlanner``, there are three types of evaluation functions implemented:

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


