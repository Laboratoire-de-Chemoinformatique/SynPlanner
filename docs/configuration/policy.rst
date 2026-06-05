.. _policy_config:

================
Policy network
================

The ranking or filtering policy network architecture and training hyperparameters can be adjusted in the training configuration file.

Download example configuration
------------------------------

- GitHub: `configs/policy_training.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/policy_training.yaml>`_
- GitHub: `configs/mhn_ranking_policy_training.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/mhn_ranking_policy_training.yaml>`_

Quickstart (CLI)
----------------

Train a policy network using the repository configuration in ``configs/policy_training.yaml``:

.. code-block:: bash

   synplan ranking_policy_training \
     --config configs/policy_training.yaml \
     --policy_data reaction_rules_policy_data.tsv \
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

MHN ranking policy
------------------

``architecture: mhn_ranking`` replaces the fixed ranking head with a dense
molecule-rule association model inspired by
`MHNreact <https://github.com/ml-jku/mhn-react>`_ and the
`MHNreact paper <https://doi.org/10.1021/acs.jcim.1c01065>`_. SynPlanner keeps
its graph embedder for product molecules and can encode rules either from
Chython fingerprints or from native QueryCGR rule graphs. Rule embeddings are
encoded lazily on the first prediction and cached for reuse.

.. code-block:: bash

   synplan ranking_policy_training \
     --config configs/mhn_ranking_policy_training.yaml \
     --policy_data reaction_rules_policy_data.tsv \
     --results_dir mhn_ranking_policy_network

The rules TSV is inferred from the extracted policy mapping name:
``<base>_policy_data.tsv`` uses ``<base>.tsv``. Keep both generated files
together when training ``mhn_ranking``.

``embedder_type`` controls the product molecule encoder. Use
``mhn_rule_encoder_type: query_cgr_graph`` and ``mhn_rule_embedder_type: gps``
to embed labeled QueryCGR rule graphs instead of Morgan rule fingerprints;
``mhn_rule_fp_*`` fields are used only by ``mhn_rule_encoder_type: fingerprint``.
QueryCGR rule graphs currently require the rule-side GPS embedder because rule
bond dynamics are encoded as edge attributes. By default, the rule graph GPS shares ``vector_dim``, ``num_conv_layers``,
``heads``, and ``attn_type`` with the product graph encoder. Override
``mhn_rule_vector_dim``, ``mhn_rule_num_conv_layers``, ``mhn_rule_heads``, or
``mhn_rule_attn_type`` when the rule encoder should use a different GPS shape.
It uses the global ``dropout`` and ``attn_dropout`` values unless
``mhn_rule_dropout`` or ``mhn_rule_attn_dropout`` are set.

To switch the default rule-fingerprint configuration to QueryCGR rule graphs,
while keeping product GPS settings at ``vector_dim: 256``,
``num_conv_layers: 5``, and ``heads: 8`` but using Performer attention for the
rule GPS:

.. code-block:: yaml

   embedder_type: gps
   vector_dim: 256
   num_conv_layers: 5
   heads: 8
   attn_type: multihead

   mhn_rule_encoder_type: query_cgr_graph
   mhn_rule_embedder_type: gps
   mhn_rule_attn_type: performer

Common MHN configurations:

.. code-block:: yaml

   # Product GCN + rule fingerprints
   embedder_type: gcn
   mhn_rule_encoder_type: fingerprint

   # Product GCN + QueryCGR rule graphs
   embedder_type: gcn
   mhn_rule_encoder_type: query_cgr_graph
   mhn_rule_embedder_type: gps

   # Product GPS + rule fingerprints
   embedder_type: gps
   mhn_rule_encoder_type: fingerprint

   # Product GPS + QueryCGR rule graphs
   embedder_type: gps
   mhn_rule_encoder_type: query_cgr_graph
   mhn_rule_embedder_type: gps

Standalone MHN ranking checkpoints can score unseen, reordered, or replaced
runtime rule sets. Combined filtering + MHN ranking policies remain restricted
to the filtering checkpoint's ordered rule set because filtering heads have a
fixed output index. SynPlanner validates dimensions; the supplied filtering
rules must retain their training order.

.. note::

   Dynamic MHN rule associations are prepared by
   ``predict_reaction_rules(precursor, reaction_rules)``. The lighter
   ``predict_reaction_rules_light(precursor, reaction_rules_len)`` API receives
   only an integer count, so it cannot bind a new runtime rule set by itself;
   use the full prediction path when MHN rules may change, or call the light
   path only after the same wrapper has already prepared the same rule set.

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
    embedder_type                      Graph embedder: ``gcn``, ``gcn_concat``, or ``gps``; ``gcn_concat`` requires ``vector_dim`` divisible by ``num_conv_layers``
    architecture                       Ranking head: ``linear`` (default) or ``mhn_ranking``
    heads                              Number of attention heads for ``embedder_type: gps``
    attn_type                          GPS attention type: ``multihead``, ``performer``, or ``null``
    attn_dropout                       Attention dropout for GPS layers
    log_grad_norm                      If true, log module-level gradient norms during training
    logger                             Training logger configuration (see below). Set to ``null`` to disable.
    mhn_association_dim                MHN molecule-rule association dimension
    mhn_beta                           Scale applied to MHN association logits
    mhn_rule_encoder_type              Rule encoder mode: ``fingerprint`` (default) or ``query_cgr_graph``
    mhn_rule_embedder_type             Rule graph embedder for ``query_cgr_graph``: ``gps`` (required)
    mhn_rule_graph_batch_size          Rule graph batch size used while embedding all rules
    mhn_rule_graph_schema_version      QueryCGR rule graph schema version included in digests and caches
    mhn_rule_vector_dim                Optional hidden dimension override for the rule graph GPS; defaults to ``vector_dim``
    mhn_rule_num_conv_layers           Optional layer-count override for the rule graph GPS; defaults to ``num_conv_layers``
    mhn_rule_heads                     Optional attention-head override for the rule graph GPS; defaults to ``heads``
    mhn_rule_attn_type                 Optional attention type override for the rule graph GPS; defaults to ``attn_type``
    mhn_rule_dropout                   Optional dropout override for MHN rule-side projection and graph embedder; defaults to ``dropout``
    mhn_rule_attn_dropout              Optional attention-dropout override for the rule-side GPS embedder; defaults to ``attn_dropout``
    mhn_rule_fp_size                   Chython Morgan rule fingerprint size; must be a power of two
    mhn_rule_fp_min_radius             Minimum Chython Morgan fingerprint radius
    mhn_rule_fp_max_radius             Maximum Chython Morgan fingerprint radius
    mhn_rule_fp_active_bits            Active bits per Chython Morgan fingerprint feature
    mhn_rule_fp_type                   Rule fingerprint source: ``query_cgr`` (default) or ``legacy``
    mhn_rule_fp_schema_version         Rule fingerprint schema version included in digests and caches
    mhn_normalize_associations         Apply non-affine LayerNorm after each MHN projection
    ================================== =========================================================================

Benchmark recipe
----------------

Train the baseline and MHN ranking policies against the same extracted rules and
``*_policy_data.tsv`` mapping, then compare validation ``balanced_accuracy_y``,
``top5_accuracy_y``, and ``top10_accuracy_y`` logs. For planning benchmarks, use
the same targets, building blocks, reaction rules, and tree configuration for
both checkpoints. Record checkpoint size, first-expansion latency (which
includes lazy MHN rule binding), warm expansion latency, and the generated
``tree_search_stats.csv`` summary.

.. code-block:: bash

   synplan ranking_policy_training \
     --config configs/policy_training.yaml \
     --policy_data reaction_rules_policy_data.tsv \
     --results_dir benchmark/linear

   synplan ranking_policy_training \
     --config configs/mhn_ranking_policy_training.yaml \
     --policy_data reaction_rules_policy_data.tsv \
     --results_dir benchmark/mhn_ranking

   du -h benchmark/linear/*.ckpt benchmark/mhn_ranking/*.ckpt

Training logger
---------------

The ``logger`` key enables `PyTorch Lightning experiment logging <https://lightning.ai/docs/pytorch/stable/extensions/logging.html>`_.
When set to ``null`` or omitted, no logger is created (the default prior behavior).
The ``type`` sub-key is required; all other sub-keys are passed directly as keyword
arguments to the corresponding Lightning logger constructor.
The ``save_dir`` parameter defaults to ``results_dir`` automatically. For
``litlogger``, ``save_dir`` is treated as an alias for LitLogger's ``root_dir``.
Remote logger integrations are optional dependencies. Install only the backend
you need, for example ``SynPlanner[litlogger]``, ``SynPlanner[wandb]``,
``SynPlanner[mlflow]``, or ``SynPlanner[loggers]`` for all optional logger
backends. With ``uv`` in this repository, use ``uv sync --extra litlogger``,
``uv sync --extra wandb``, or ``uv sync --extra loggers``.

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
    type                      yes        Logger backend: ``csv``, ``tensorboard``, ``litlogger``, ``mlflow``, or ``wandb``
    save_dir                  no         Log output directory (defaults to ``results_dir``)
    *(other keys)*            no         Passed directly to the Lightning logger constructor
    ========================= ========== =========================================================================

**CSV logger** (no extra dependencies)

Logs training metrics to CSV files on disk. See the
`CSVLogger docs <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html>`_
for all available parameters.

.. code-block:: yaml

    # Minimal: logs to <results_dir>/lightning_logs/version_0/metrics.csv
    logger:
      type: csv

.. code-block:: yaml

    # Customized: flat output directory, flush more often
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

**LitLogger** (requires ``SynPlanner[litlogger]`` or ``SynPlanner[loggers]``)

Logs metrics, metadata, terminal output, and optionally model checkpoints to
Lightning AI. See the
`LitLogger docs <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.LitLogger.html>`_
for all available parameters.

.. code-block:: yaml

    logger:
      type: litlogger
      name: ranking_gps_g
      root_dir: /path/to/results
      log_model: true
      save_logs: true
      metadata:
        dataset: uspto_full
        policy: gps_g

**MLflow logger** (requires ``SynPlanner[mlflow]`` or ``SynPlanner[loggers]``)

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

**Weights & Biases logger** (requires ``SynPlanner[wandb]`` or ``SynPlanner[loggers]``)

Logs to `Weights & Biases <https://wandb.ai/>`_. See the
`WandbLogger docs <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html>`_
for all available parameters.

.. code-block:: yaml

    logger:
      type: wandb
      project: synplanner-ranking
      name: gps-embedder-v1
