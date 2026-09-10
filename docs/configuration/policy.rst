.. _policy_config:

================
Policy network
================

The ranking or filtering policy network architecture and training hyperparameters can be adjusted in the training configuration file.

ONNX inference on CPU
---------------------

The base installation includes CPU ONNX inference. Install the optional export tools, then export a ranking or filtering checkpoint
from a source checkout:

.. code-block:: bash

   uv sync --no-dev --extra cpu --extra training
   uv run --no-sync python -m scripts.export_policy_onnx ranking.ckpt ranking.onnx

Use the exported file with the existing policy loader or as ``weights_path``
in the planning configuration:

.. code-block:: python

   from synplan.utils.loading import load_policy_function

   policy = load_policy_function(weights_path="ranking.onnx", top_rules=50)

The export handles one molecule per call with variable atom and bond counts.
Use the same ordered reaction-rule library as the checkpoint. Planning with
``.onnx`` weights uses NumPy and ONNX Runtime and does not require Torch.
The ``training`` extra includes the export tools. For preset entries ending
in ``.ckpt``, ``download_preset`` checks Hugging Face for a same-name ``.onnx`` file
in the same folder and downloads it instead when available. Use the returned
``paths["ranking_policy"]`` (or the path printed by the CLI). If no ONNX file exists,
the original checkpoint is downloaded; export it once or install ``SynPlanner[cpu]``
to use it. Filtering exports retain both rule and priority heads; load them with
``policy_type="filtering"`` and their original ordered rule library. The legacy
article filtering model is incompatible with the GPS rule library. MHN models
are not supported by this exporter.

To export a value network, pass ``--value``:

.. code-block:: bash

   uv run --no-sync python -m scripts.export_policy_onnx value_network.ckpt value_network.onnx --value

The existing value-network evaluation configuration accepts the exported
``.onnx`` path and runs it on CPU without Torch.

``SynPlanner[curation]`` covers reaction data preparation and neural atom mapping
with Torch, Chytorch, and SciPy. ChemFrame and pandas analysis are included in the
base installation. ``SynPlanner[training]`` covers model
training, notebooks, and ONNX export, including Lightning and AdaBelief.
``SynPlanner[gui]`` adds the Streamlit planning interface.
``SynPlanner[all]`` includes all three workflows.
With ``uv``, combine ``curation``, ``training``, or ``all`` with ``cpu``, ``cu126``,
or ``cu128`` to select a Torch backend, for example
``uv sync --no-dev --extra curation --extra cu128`` for GPU mapping or
``uv sync --no-dev --extra training --extra cu128`` for GPU training.
Backend indexes are configured
for ``uv``; ``pip`` users select the Torch build through the PyTorch package index.

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
``rule_embedding_type: query_cgr_graph`` (with ``rule_embedder.embedder_type: gps``)
to embed labeled QueryCGR rule graphs instead of Morgan rule fingerprints;
``rule_fp_*`` fields are used only by ``rule_embedding_type: fingerprint``.
QueryCGR rule graphs currently require the rule-side GPS embedder because rule
bond dynamics are encoded as edge attributes. By default, the rule graph GPS
shares ``vector_dim``, ``num_conv_layers``, ``heads``, and ``attn_type`` with the
product graph encoder. Set ``rule_embedder.vector_dim``,
``rule_embedder.num_conv_layers``, ``rule_embedder.heads``, or
``rule_embedder.attn_type`` when the rule encoder should use a different GPS
shape. It uses the global ``dropout`` and ``attn_dropout`` values unless
``rule_embedder.dropout`` or ``rule_embedder.attn_dropout`` are set.

To switch the default rule-fingerprint configuration to QueryCGR rule graphs,
while keeping product GPS settings at ``vector_dim: 256``,
``num_conv_layers: 5``, and ``heads: 8`` but using Performer attention for the
rule GPS:

.. code-block:: yaml

   architecture: mhn_ranking
   embedder_type: gps
   vector_dim: 256
   num_conv_layers: 5
   heads: 8
   attn_type: multihead

   rule_embedding_type: query_cgr_graph
   rule_embedder:
     embedder_type: gps
     attn_type: performer

Common MHN configurations. Each block below is a **separate** config file, not four
stanzas of one file — ``architecture: mhn_ranking`` is mandatory in every one, because
every ``rule_*`` key is rejected by the default ``architecture: linear`` config
(the models use ``extra="forbid"``):

.. code-block:: yaml

   # Product GCN + rule fingerprints
   architecture: mhn_ranking
   embedder_type: gcn
   rule_embedding_type: fingerprint

.. code-block:: yaml

   # Product GCN + QueryCGR rule graphs
   architecture: mhn_ranking
   embedder_type: gcn
   rule_embedding_type: query_cgr_graph
   rule_embedder:
     embedder_type: gps

.. code-block:: yaml

   # Product GPS + rule fingerprints
   architecture: mhn_ranking
   embedder_type: gps
   rule_embedding_type: fingerprint

.. code-block:: yaml

   # Product GPS + QueryCGR rule graphs
   architecture: mhn_ranking
   embedder_type: gps
   rule_embedding_type: query_cgr_graph
   rule_embedder:
     embedder_type: gps

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
    attn_type                          GPS attention type: ``performer`` (default) or ``multihead``
    attn_dropout                       Attention dropout for GPS layers
    log_grad_norm                      If true, log module-level gradient norms during training
    logger                             Training logger configuration (see below). Set to ``null`` to disable.
    association_dim                    MHN molecule-rule association dimension
    beta                               Scale applied to MHN association logits
    normalize_associations             Apply non-affine LayerNorm after each MHN projection
    rule_embedding_type                Rule encoder mode: ``fingerprint`` (default) or ``query_cgr_graph``
    rule_graph_batch_size              Rule graph batch size used while embedding all rules
    rule_graph_schema_version          QueryCGR rule graph schema version included in digests and caches
    rule_fp_size                       Chython Morgan rule fingerprint size; must be a power of two
    rule_fp_min_radius                 Minimum Chython Morgan fingerprint radius
    rule_fp_max_radius                 Maximum Chython Morgan fingerprint radius
    rule_fp_active_bits                Active bits per Chython Morgan fingerprint feature
    rule_fp_type                       Rule fingerprint source: ``query_cgr`` (default), ``legacy``, or ``mhnreact_rdkit`` (RDKit MHNreact-compatible)
    rule_fp_schema_version             Rule fingerprint schema version included in digests and caches
    rule_embedder.embedder_type        Rule graph embedder for ``query_cgr_graph``: ``gps`` (required)
    rule_embedder.vector_dim           Optional hidden dimension override for the rule graph GPS; defaults to ``vector_dim``
    rule_embedder.num_conv_layers      Optional layer-count override for the rule graph GPS; defaults to ``num_conv_layers``
    rule_embedder.heads                Optional attention-head override for the rule graph GPS; defaults to ``heads``
    rule_embedder.attn_type            Optional attention type override for the rule graph GPS; defaults to ``attn_type``
    rule_embedder.dropout              Optional dropout override for the rule-side projection and graph embedder; defaults to ``dropout``
    rule_embedder.attn_dropout         Optional attention-dropout override for the rule-side GPS embedder; defaults to ``attn_dropout``
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
Every backend except ``csv`` needs a package SynPlanner does not depend on:

.. table::
    :widths: 15 40

    ============== ==========================================================
    Logger type    How to install its backend
    ============== ==========================================================
    csv            nothing to install
    tensorboard    ``pip install tensorboard`` (or ``tensorboardX``) — there is no SynPlanner extra for it
    mlflow         ``pip install mlflow``
    wandb          ``pip install wandb``
    litlogger      ``pip install litlogger``
    ============== ==========================================================

Install the tracking backend you use directly; SynPlanner does not define
logger-specific extras. In a ``uv`` project, use ``uv add mlflow`` or ``uv add wandb``.

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

**LitLogger** (requires ``pip install litlogger``; no SynPlanner extra provides it)

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

**Weights & Biases logger** (requires ``pip install wandb``)

Logs to `Weights & Biases <https://wandb.ai/>`_. See the
`WandbLogger docs <https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html>`_
for all available parameters.

.. code-block:: yaml

    logger:
      type: wandb
      project: synplanner-ranking
      name: gps-embedder-v1
