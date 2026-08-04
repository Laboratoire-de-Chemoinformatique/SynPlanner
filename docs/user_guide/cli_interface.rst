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

    synplan download_preset --preset synplanner-gps --save_to synplan_data

**Parameters**:
    - ``preset`` - preset name (default: ``synplanner-gps``).
    - ``save_to`` - the directory where downloaded data will be stored.

Protection source-data conversion
---------------------------------
Convert the Westerlund protection-strategy source dataset into the YAML and
TSV files consumed by SynPlanner's protection configuration. The source
directory must contain ``protection_reactive_function_SMARTS.txt``,
``halogen_reactive_function_SMARTS.txt``, and
``protection_SMARTS_incompatibility.csv``. Supporting template and label
mapping files are copied when present.

Use an explicit output directory so generated files are written to a known
location:

.. code-block:: bash

    synplan-convert-protection-data /path/to/protection-source \
        --output-dir synplan_data/protection

The command writes ``competing_groups.yaml``, ``halogen_groups.yaml``, and
``incompatibility_matrix.tsv``, together with any supporting files copied from
the source directory. Without ``--output-dir``, output is written relative to
the installed converter script.

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

.. note::
    This command **deduplicates by default** (``deduplicate: true`` in
    ``configs/reactions_standardization.yaml``, and also the model default, so removing
    the key does not disable it). Duplicates are matched on the CGR of the standardized
    reaction, so records differing only in atom numbering, component order or SMILES
    writing collapse into one. They are counted in the summary's ``failed`` total, listed
    separately as ``duplicates removed``, and deliberately not written to the error TSV.
    Output count below input count is therefore expected even on perfectly clean data.

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
      ``csv`` and ``tensorboard`` work out of the box; ``wandb`` and ``mlflow``
      need ``SynPlanner[wandb]``, ``SynPlanner[mlflow]`` or ``SynPlanner[loggers]``.
      ``litlogger`` needs ``pytorch-lightning>=2.6.1`` (which is where the
      ``LitLogger`` class lives) plus ``pip install litlogger``. There is no
      ``SynPlanner[litlogger]`` extra — ``pip`` only warns and installs nothing
      when you ask for one — and ``SynPlanner[loggers]`` covers wandb + mlflow only.

**MHN ranking policy tuning**

.. code-block:: bash

    synplan mhn_network_tuning --config configs/mhn_ranking_policy_training.yaml --policy_network policy_network.ckpt --new_policy_data new_reaction_rules_policy_data.tsv --results_dir mhn_tuned

**Parameters**:
    - ``config`` - the path to the policy configuration file. Fine-tuning
      epochs, batch size, learning rate, logger settings, and Trainer options
      are read from this YAML file.
    - ``policy_network`` - the path to an already trained MHN ranking checkpoint.
    - ``new_policy_data`` - the path to the new ranking policy mapping file (``*_policy_data.tsv``) generated during rule extraction.
    - ``results_dir`` - the path to the directory where the tuned MHN checkpoint will be stored.
    - ``--workers`` - CPU workers for ranking dataset preprocessing (0 = auto).
    - ``--no-cache`` - disable dataset cache reuse.

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
      ``csv`` and ``tensorboard`` work out of the box; ``wandb`` and ``mlflow``
      need ``SynPlanner[wandb]``, ``SynPlanner[mlflow]`` or ``SynPlanner[loggers]``.
      ``litlogger`` needs ``pytorch-lightning>=2.6.1`` (which is where the
      ``LitLogger`` class lives) plus ``pip install litlogger``. There is no
      ``SynPlanner[litlogger]`` extra — ``pip`` only warns and installs nothing
      when you ask for one — and ``SynPlanner[loggers]`` covers wandb + mlflow only.

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
    - ``value_network`` - optional path to value network weights to start tuning from.
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
    - ``value_network`` - the path to the file with trained value network. Read **only** when the
      config sets ``node_evaluation: evaluation_type: gcn`` (as ``configs/planning_value.yaml``
      does). With ``configs/planning_standard.yaml`` the evaluator defaults to ``rollout`` and this
      option is silently ignored.
    - ``results_dir`` - the path to the directory where the planning results will be stored.
    - ``--reconcile-mapping`` - reconcile atom-map numbering across route steps (slower).
    - ``--export_routes`` - additionally write ``results.json.gz`` + ``manifest.json``.
