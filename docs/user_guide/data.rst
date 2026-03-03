.. _data:

Data
====

This section summarizes the datasets used by SynPlanner and how to obtain them.

Overview of datasets
--------------------

``SynPlanner`` operates reaction and molecule data stored in different formats.

.. table::
    :widths: 30 50

    ================================== =================================================================================
    Data type                          Description
    ================================== =================================================================================
    Reactions                          Reactions can be loaded and stored as the list of reaction smiles in the file (.smi) or RDF File (.rdf)
    Molecules                          Molecules can be loaded and stored as the list of molecule smiles in the file (.smi) or SDF File (.sdf)
    Reaction rules                     Reaction rules stored as TSV (.tsv, preferred) or pickled list (.pickle, legacy)
    Retrosynthetic models              Retrosynthetic models (neural networks) can be loaded and stored as serialized PyTorch models (.ckpt)
    Retrosynthetic routes              Retrosynthetic routes can be visualized and stored as HTML files (.html) and can be stored as JSON files (.json)
    ================================== =================================================================================

.. note::
    Reaction and molecule file formats are parsed and recognized automatically by ``SynPlanner`` from file extensions.
    Be sure to store the data with the correct extension.

Data repository structure
--------------------------

Data is hosted on HuggingFace in a component-based structure:

.. code-block:: text

    SynPlanner-data/
    ├── policy/                    # Reaction rules + policy network weights
    │   └── {architecture}/
    │       └── {rules_version}/
    │           ├── reaction_rules.tsv
    │           ├── pipeline.yaml
    │           └── {weights_version}/
    │               ├── ranking_policy.ckpt
    │               └── filtering_policy.ckpt
    ├── value/                     # Value network weights
    │   └── {architecture}/
    │       └── {version}/
    │           ├── value_network.ckpt
    │           └── meta.yaml
    ├── building_blocks/           # Building block sets
    │   └── {name}/
    │       ├── building_blocks.tsv
    │       └── meta.yaml
    ├── reaction_data/             # Raw → standardized → filtered pipeline
    │   └── {source}/
    │       ├── raw/
    │       ├── standardized/{YYYY-MM-DD}/
    │       └── filtered/{YYYY-MM-DD}/
    ├── training_data/             # Per-network training inputs
    │   ├── ranking_policy/{YYYY-MM-DD}/
    │   ├── filtering_policy/{YYYY-MM-DD}/
    │   └── value_network/{YYYY-MM-DD}/
    ├── presets/                   # Ready-to-use preset definitions
    │   └── {name}.yaml
    └── benchmarks/                # Benchmark target sets (downloaded separately)
        └── sascore/

**Versioning**: model components (``policy/``, ``value/``) are versioned by directory name
(architecture family, rules version, weights version). Pipeline data (``reaction_data/``,
``training_data/``) is versioned by processing date (``YYYY-MM-DD``).

Data sources and bundles
------------------------

| **policy/supervised_gcn/v1/** — reaction rules and policy weights
| ``reaction_rules.tsv`` — 24k reaction rules in SMARTS format (TSV)
| ``v1/ranking_policy.ckpt`` — ranking policy network trained on filtered USPTO and corresponding rules
| ``v1/filtering_policy.ckpt`` — filtering policy network trained on ChEMBL molecules and corresponding rules
| ``pipeline.yaml`` — full reproducibility manifest (standardization, filtration, extraction, training configs)

| **value/supervised_gcn/v1/** — value network weights
| ``value_network.ckpt`` — value network trained from planning simulations on ChEMBL targets

| **building_blocks/emolecules-salt-ln/** — building blocks
| ``building_blocks.tsv`` — 186k standardized building blocks (eMolecules + Sigma Aldrich)

| **reaction_data/uspto/** — reaction data pipeline
| ``raw/uspto_full_mapped.smi.zip`` — original USPTO dataset (1.48M reactions, compressed)
| ``standardized/2024-12-31/`` — standardized reactions + config + errors
| ``filtered/2024-12-31/`` — filtered reactions + config + errors

| **training_data/** — per-network training inputs (date-versioned)
| ``filtering_policy/2024-12-31/molecules_for_training.smi.zip`` — ChEMBL molecules for filtering policy training
| ``value_network/2024-12-31/targets_for_training.smi.zip`` — ChEMBL targets for value network tuning

| **benchmarks/sascore/** — SynPlanner original benchmarks (downloaded separately)
| 7 target subsets split by SAScore range (1.5–8.5)

Download data
-------------

.. include:: ../get_started/data_download.rst

Download from Hugging Face (browse)
-----------------------------------

- New repository: `Hugging Face – SynPlanner-data <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner-data/tree/main>`_
  - `policy/ <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner-data/tree/main/policy>`_
  - `value/ <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner-data/tree/main/value>`_
  - `building_blocks/ <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner-data/tree/main/building_blocks>`_
  - `reaction_data/ <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner-data/tree/main/reaction_data>`_
  - `benchmarks/ <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner-data/tree/main/benchmarks>`_
  - `presets/ <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner-data/tree/main/presets>`_

- Legacy repository: `Hugging Face – SynPlanner <https://huggingface.co/Laboratoire-De-Chemoinformatique/SynPlanner/tree/main>`_ (flat structure, deprecated)
