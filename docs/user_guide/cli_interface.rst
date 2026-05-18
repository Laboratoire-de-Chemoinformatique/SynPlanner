.. _cli_interface:

============================
Command-line interface
============================

Use SynPlanner from the command line to run data curation, training, and planning.

For installation and prebuilt Docker images, see :doc:`/get_started/index`. For Python usage, refer to :doc:`/api`.

Data download
---------------------------
Download a ready-to-use data preset from HuggingFace with all components needed for planning:

.. code-block:: bash

    synplan download_preset --preset synplanner-article --save_to synplan_data

**Parameters**:
    - ``preset`` - preset name (default: ``synplanner-article``).
    - ``save_to`` - the directory where downloaded data will be stored.

ORD conversion
---------------------------
ORD ``.pb`` datasets can be converted to SynPlanner-compatible reaction SMILES:

.. code-block:: bash

    synplan ord_convert --input reactions.pb --output reactions.smi

**Parameters**:
    - ``input`` - the path to the ORD ``.pb`` dataset.
    - ``output`` - the path to the output ``.smi`` file.

Building blocks standardization
-------------------------------
Standardize custom building blocks for compatibility with ``SynPlanner``.

.. code-block:: bash

    synplan building_blocks_standardizing --input building_blocks_original.smi --output building_blocks_standardized.smi

**Parameters**:
    - ``input`` - the path to the file (.smi or .rdf) with building blocks to be standardized.
    - ``output`` - the path to the file (.smi or .rdf) where standardized building blocks to be stored.

Reaction standardization
---------------------------
Reactions can be standardized with ``SynPlanner``. The list of applied standardizers (see the details here) should be provided
in the configuration file (see the details here). ``SynPlanner`` takes the file with the list of reaction smiles and records
the standardized reactions as reaction smiles in the output file. If the reaction standardization fails by some reason
(e.g. incorrect reaction or corrupt smiles), the corresponding reactions will be discarded, which means that ``SynPlanner``
also works as a general reaction data cleaner.

For SMI inputs with source/provenance columns, such as mapped USPTO rows in the
form ``reaction_smiles<TAB>row_id<TAB>patent_ids``, standardization preserves
those source columns in successful output rows. When ``--ignore-errors`` is
used, failed rows are removed from the standardized output and written to the
error TSV with a ``source_info`` column.

.. code-block:: bash

    synplan reaction_standardizing --config configs/reactions_standardization.yaml --input reaction_data_original.smi --output reaction_data_standardized.smi

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``input`` - the path to the file (.smi or .rdf) with reactions to be standardized.
    - ``output`` - the path to the file (.smi or .rdf) where standardized reactions to be stored.
    - ``--num_cpus`` - number of worker processes.
    - ``--batch_size`` - number of reactions per worker batch.
    - ``--ignore-errors`` / ``--no-ignore-errors`` - skip bad reactions or fail fast.
    - ``--error-file`` - path for failed reaction rows.
    - ``--silent`` - suppress the progress bar. By default, the CLI shows progress.

Reaction filtration
---------------------------
Reaction filtration allows the discarding of unreasonable and unrealistic chemical reactions, which should help in the
prediction of better-quality retrosynthetic routes. The list of applied reaction filters (see the details here) should
be provided in the configuration file (see the details here). Only reactions successfully passed the specified reaction
filters will be stored in the output file.

.. code-block:: bash

    synplan reaction_filtering --config configs/reactions_filtration.yaml --input reaction_data_standardized.smi --output reaction_data_filtered.smi

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``input`` - the path to the file (.smi or .rdf) with reactions to be filtered.
    - ``output`` - the path to the file (.smi or .rdf) where filtered reactions to be stored.
    - ``--num_cpus`` - number of worker processes.
    - ``--batch_size`` - number of reactions per worker batch.
    - ``--ignore-errors`` / ``--no-ignore-errors`` - skip bad reactions or fail fast.
    - ``--error-file`` - path for failed or filtered reaction rows.

Reaction mapping
---------------------------
Reaction atoms can be mapped with the neural mapper:

.. code-block:: bash

    synplan reaction_mapping --input reaction_data_original.smi --output reaction_data_mapped.smi

**Parameters**:
    - ``config`` - optional mapping configuration file.
    - ``input`` - the path to the file with reactions to be mapped.
    - ``output`` - the path where mapped reactions will be stored.
    - ``--workers`` - CPU worker count (0 = auto).
    - ``--device`` - torch device: ``cuda``, ``mps``, or ``cpu``.
    - ``--no-amp`` - disable automatic mixed precision.
    - ``--batch-size`` - GPU batch size.
    - ``--ignore-errors`` / ``--no-ignore-errors`` - skip bad reactions or fail fast.
    - ``--error-file`` - path for failed reaction rows.

Reaction rule extraction
---------------------------
Reaction rules extraction should be performed for high-quality (cleaned, standardized, and filtered) reaction data
to ensure the extraction of meaningful reaction rules. The specificity of extracted reaction rules can be adjusted by
the configuration file (see the details here). The extracted reaction rules will be stored in TSV format.
A policy training mapping file (``*_policy_data.tsv``) is also generated alongside the rules,
containing product SMILES and rule IDs ready for ranking policy training.

.. code-block:: bash

    synplan rule_extracting --config configs/rules_extraction.yaml --input reaction_data_filtered.smi --output reaction_rules.tsv

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``input`` - the path to the file (.smi or .rdf) with reactions for reaction rule extraction.
    - ``output`` - the path to the file (.tsv) where extracted reaction rules will be stored.
      A ``*_policy_data.tsv`` file for ranking policy training is generated alongside.
    - ``--num_cpus`` - number of worker processes.
    - ``--batch_size`` - number of reactions per worker batch.
    - ``--ignore-errors`` / ``--no-ignore-errors`` - skip bad reactions or fail fast.
    - ``--error-file`` - path for failed reaction rows.

Policy networks training
---------------------------
Ranking and filtering policy networks (see the details here) can be trained with ``SynPlanner``. The architecture of both
types of policy networks is configured by the same configuration file (see the details here).

**Ranking policy network**

.. code-block:: bash

    synplan ranking_policy_training --config configs/policy_training.yaml --policy_data reaction_rules_policy_data.tsv --results_dir ranking_policy_network

**Parameters**:
    - ``config`` - the path to the policy configuration file.
    - ``policy_data`` - the path to the policy training mapping file (``*_policy_data.tsv``) generated during rule extraction.
    - ``results_dir`` - the path to the directory where the trained policy network will be stored.
    - ``--workers`` - CPU workers for ranking dataset preprocessing (0 = auto).
    - ``--no-cache`` - disable dataset cache reuse.
    - ``--logger`` - logger backend: ``csv``, ``tensorboard``, ``mlflow``, ``wandb``, or ``litlogger``.
      Optional remote backends require the matching extra, e.g. ``SynPlanner[litlogger]``,
      ``SynPlanner[wandb]``, ``SynPlanner[mlflow]``, or ``SynPlanner[loggers]``.

**Filtering policy network**

.. code-block:: bash

    synplan filtering_policy_training --config configs/policy_training.yaml --molecule_data molecules_data.smi --reaction_rules reaction_rules.tsv --results_dir filtering_policy_network

**Parameters**:
    - ``config`` - the path to the policy configuration file.
    - ``molecule_data`` - the path to the file with molecules for filtering policy training.
    - ``reaction_rules`` - the path to the file with extracted reaction rules.
    - ``results_dir`` - the path to the directory where the trained policy network will be stored.
    - ``--num_cpus`` - CPUs for filtering dataset preparation.
    - ``--no-cache`` - disable dataset cache reuse.
    - ``--logger`` - logger backend: ``csv``, ``tensorboard``, ``mlflow``, ``wandb``, or ``litlogger``.
      Optional remote backends require the matching extra, e.g. ``SynPlanner[litlogger]``,
      ``SynPlanner[wandb]``, ``SynPlanner[mlflow]``, or ``SynPlanner[loggers]``.

Value network training
---------------------------
Value neural networks (see the details here) can be used instead of rollout simulations for node evaluation in MCTS.
The value network training involves the extracted reaction rules, trained policy network, and planning simulations.
The architecture of the value network, planning parameters, and value network tuning parameters can be specified
with the configuration file (see the details here).

.. code-block:: bash

    synplan value_network_tuning --config configs/tuning.yaml --targets targets.smi --reaction_rules reaction_rules.tsv --policy_network policy_network.ckpt --building_blocks building_blocks.smi --results_dir value_network

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``targets`` - the path to the file with target molecules for planning simulations.
    - ``reaction_rules`` - the path to the file with reactions rules.
    - ``building_blocks`` - the path to the file with building blocks.
    - ``policy_network`` - the path to the file with trained policy network (ranking or filtering policy network).
    - ``results_dir`` - the path to the directory where the trained value network will be to be stored.

Retrosynthetic planning
---------------------------
Retrosynthetic planning can be performed in ``SynPlanner``.

.. code-block:: bash

    synplan planning --config configs/planning_standard.yaml --targets targets.smi --reaction_rules reaction_rules.tsv --building_blocks building_blocks_stand.smi --policy_network policy_network.ckpt --results_dir planning_results

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``targets`` - the path to the file with target molecule for retrosynthetic planning.
    - ``reaction_rules`` - the path to the file with reaction rules.
    - ``building_blocks`` - the path to the file with building blocks.
    - ``policy_network`` - the path to the file with trained policy network (ranking or filtering).
    - ``value_network`` - the path to the file with trained value network if available (default is None).
    - ``results_dir`` - the path to the directory where the trained value network will be to be stored.
