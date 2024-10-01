.. _policy:

================
Policy network
================

The ranking or filtering policy network architecture and training hyperparameters can be adjusted in the training configuration file.

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
