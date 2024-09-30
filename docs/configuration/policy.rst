.. _policy:

================
Policy network
================

The ranking or filtering policy network architecture and training hyperparameters can be adjusted in the training configuration yaml file below.

.. code-block:: yaml

    vector_dim: 512
    num_conv_layers: 5
    learning_rate: 0.0005
    dropout: 0.4
    num_epoch: 100
    batch_size: 1000

**Configuration parameters**:

.. table::
    :widths: 20 10 50

    ================================== ======= =========================================================================
    Parameter                          Default  Description
    ================================== ======= =========================================================================
    vector_dim                         512     The dimension of the hidden layers
    num_conv_layers                    5       The number of convolutional layers
    learning_rate                      0.0005  The learning rate
    dropout                            0.4     The dropout value
    num_epoch                          100     The number of training epochs
    batch_size                         1000    The size of the training batch of input molecular graphs
    ================================== ======= =========================================================================
