.. _policy_config:

================
Policy network
================

The ranking or filtering policy network architecture and training hyperparameters can be adjusted in the training configuration file.

Download example configuration
------------------------------

- GitHub: `configs/policy.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/policy.yaml>`_

Quickstart (CLI)
----------------

Train a policy network using the repository configuration in ``configs/policy.yaml``:

.. code-block:: bash

   synplan ranking_policy_training \
     --config configs/policy.yaml \
     --reaction_data reaction_data_filtered.smi \
     --reaction_rules reaction_rules.pickle \
     --results_dir ranking_policy_network

**Configuration file**

.. code-block:: yaml

    vector_dim: 512
    num_conv_layers: 5
    learning_rate: 0.0005
    dropout: 0.4
    num_epoch: 100
    batch_size: 1000

**Configuration parameters**

.. table::
    :widths: 20 50

    ================================== =========================================================================
    Parameter                          Description
    ================================== =========================================================================
    vector_dim                         The dimension of the hidden layers
    num_conv_layers                    The number of convolutional layers
    learning_rate                      The learning rate
    dropout                            The dropout value
    num_epoch                          The number of training epochs
    batch_size                         The size of the training batch of input molecular graphs
    ================================== =========================================================================
