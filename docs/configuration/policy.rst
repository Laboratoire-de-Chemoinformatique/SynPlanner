.. _policy_config:

================
Policy network
================

The ranking or filtering policy network architecture and training hyperparameters can be adjusted in the training configuration file.

Download example configuration
------------------------------

- GitHub: `configs/policy_training.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/policy_training.yaml>`_

Quickstart (CLI)
----------------

Train a policy network using the repository configuration in ``configs/policy_training.yaml``:

.. code-block:: bash

   synplan ranking_policy_training \
     --config configs/policy_training.yaml \
     --reaction_data reaction_data_filtered.smi \
     --reaction_rules reaction_rules.tsv \
     --results_dir ranking_policy_network

**Configuration file**

.. code-block:: yaml

    vector_dim: 512
    num_conv_layers: 5
    learning_rate: 0.0005
    dropout: 0.4
    num_epoch: 100
    batch_size: 1000

    logger:
      type: csv

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
    logger                             Training logger configuration (see below). Set to ``null`` to disable.
    ================================== =========================================================================

Training logger
---------------

The ``logger`` key enables `PyTorch Lightning experiment logging <https://lightning.ai/docs/pytorch/stable/extensions/logging.html>`_.
When set to ``null`` or omitted, no logger is created (the default prior behavior).
The ``type`` sub-key is required; all other sub-keys are passed directly as keyword
arguments to the corresponding Lightning logger constructor.
The ``save_dir`` parameter defaults to ``results_dir`` automatically.

You can also enable a logger from the command line without editing the YAML file:

.. code-block:: bash

   synplan ranking_policy_training \
     --config configs/policy_training.yaml \
     --policy_data reaction_rules_policy_data.tsv \
     --results_dir ranking_policy_network \
     --logger csv

.. table::
    :widths: 15 10 45

    ========================= ========== =========================================================================
    Sub-key                   Required   Description
    ========================= ========== =========================================================================
    type                      yes        Logger backend: ``csv``, ``tensorboard``, ``mlflow``, or ``wandb``
    save_dir                  no         Log output directory (defaults to ``results_dir``)
    *(other keys)*            no         Passed directly to the Lightning logger constructor
    ========================= ========== =========================================================================

**CSV logger** (no extra dependencies)

Logs training metrics to CSV files on disk. See the
`CSVLogger docs <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html>`_
for all available parameters.

.. code-block:: yaml

    # Minimal — logs to <results_dir>/lightning_logs/version_0/metrics.csv
    logger:
      type: csv

.. code-block:: yaml

    # Customized — flat output directory, flush more often
    logger:
      type: csv
      name: null                     # no "lightning_logs" subfolder
      flush_logs_every_n_steps: 50   # write to disk every 50 steps (default: 100)

CSV logger parameters:

.. table::
    :widths: 25 45

    ============================== =========================================================================
    Parameter                      Description
    ============================== =========================================================================
    name                           Subfolder name inside ``save_dir``. Default ``"lightning_logs"``.
                                   Set to ``null`` to log directly into ``save_dir/version_X/``.
    version                        Run version (int or str). Auto-increments if omitted.
    prefix                         String prepended to all metric keys. Default ``""``.
    flush_logs_every_n_steps       How often to write to disk. Default ``100``.
    ============================== =========================================================================

**MLflow logger** (requires ``pip install mlflow``)

Logs to an `MLflow <https://mlflow.org>`_ tracking server. See the
`MLFlowLogger docs <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.MLFlowLogger.html>`_
for all available parameters.

.. code-block:: yaml

    # Local file-based tracking
    logger:
      type: mlflow
      experiment_name: synplanner_ranking
      tracking_uri: file:./mlruns

.. code-block:: yaml

    # Remote tracking server
    logger:
      type: mlflow
      experiment_name: synplanner_ranking
      tracking_uri: http://localhost:5000
      run_name: gps-embedder-v1
