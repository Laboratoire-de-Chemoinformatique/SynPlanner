.. _cli_interface:

============================
Command-line interface
============================

Use SynPlanner from the command line to run data curation, training, and planning.

For installation and prebuilt Docker images, see :doc:`/get_started/index`. For Python usage, refer to :doc:`/api`.

Data download
---------------------------
Reaction and molecule data needed for training retrosynthetic models and retrosynthetic planning with pre-trained models can be downloaded directly from ``SynPlanner``.
See the purpose and description of downloaded data **here**.

.. code-block:: bash

    synplan download_all_data --save_to synplan_data

**Parameters**:
    - ``save_to`` - the location where the downloaded data will be stored.

Building blocks standardization
-------------------------------
It is crucial to standardize custom building blocks for compatibility with ``SynPlanner``.

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

.. code-block:: bash

    synplan reaction_standardizing --config configs/standardization.yaml --input reaction_data_original.smi --output reaction_data_standardized.smi

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``input`` - the path to the file (.smi or .rdf) with reactions to be standardized.
    - ``output`` - the path to the file (.smi or .rdf) where standardized reactions to be stored.

Reaction filtration
---------------------------
Reaction filtration allows the discarding of unreasonable and unrealistic chemical reactions, which should help in the
prediction of better-quality retrosynthetic routes. The list of applied reaction filters (see the details here) should
be provided in the configuration file (see the details here). Only reactions successfully passed the specified reaction
filters will be stored in the output file.

.. code-block:: bash

    synplan reaction_filtering --config configs/filtration.yaml --input reaction_data_standardized.smi --output reaction_data_filtered.smi

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``input`` - the path to the file (.smi or .rdf) with reactions to be filtered.
    - ``output`` - the path to the file (.smi or .rdf) where filtered reactions to be stored.

Reaction rule extraction
---------------------------
Reaction rules extraction should be performed for high-quality (cleaned, standardized, and filtered) reaction data
to ensure the extraction of meaningful reaction rules. The specificity of extracted reaction rules can be adjusted by
the configuration file (see the details here). The extracted reaction rules will be stored in a pickled list of reaction rules
compatible with the CGRTools package.

.. code-block:: bash

    synplan rule_extracting --config configs/extraction.yaml --input reaction_data_filtered.smi --output reaction_rules.pickle

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``input`` - the path to the file (.smi or .rdf) with reactions for reaction rule extraction.
    - ``output`` - the path to the file (.pickle) where extracted reactions rules to be stored.

Policy networks training
---------------------------
Ranking and filtering policy networks (see the details here) can be trained with ``SynPlanner``. The architecture of both
types of policy networks is configured by the same configuration file (see the details here).

**Ranking policy network**

.. code-block:: bash

    synplan ranking_policy_training --config configs/policy.yaml --reaction_data reaction_data_filtered.smi --reaction_rules reaction_rules.pickle --results_dir ranking_policy_network

**Parameters**:
    - ``config`` - the path to the policy configuration file.
    - ``reaction_data`` - the path to the file with reactions for ranking policy training.
    - ``reaction_rules`` - the path to the file with extracted reaction rules.
    - ``results_dir`` - the path to the directory where the trained policy network will be stored.

**Filtering policy network**

.. code-block:: bash

    synplan filtering_policy_training --config configs/policy.yaml --molecule_data molecules_data.smi --reaction_rules reaction_rules.pickle --results_dir filtering_policy_network

**Parameters**:
    - ``config`` - the path to the policy configuration file.
    - ``molecule_data`` - the path to the file with molecules for filtering policy training.
    - ``reaction_rules`` - the path to the file with extracted reaction rules.
    - ``results_dir`` - the path to the directory where the trained policy network will be stored.

Value network training
---------------------------
Value neural networks (see the details here) can be used instead of rollout simulations I no evaluation in MCTS.
The value network training involves the extracted reaction rules, trained policy network, and planning simulations.
The architecture of the value network, planning parameters, and value network tuning parameters can be specified
with the configuration file (see the details here).

.. code-block:: bash

    synplan value_network_tuning --config configs/tuning.yaml --targets targets.smi --reaction_rules reaction_rules.pickle --policy_network policy_network.ckpt --building_blocks building_blocks.smi --results_dir value_network

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

    synplan planning --config configs/planning.yaml --targets targets.smi --reaction_rules reaction_rules.pickle --building_blocks building_blocks_stand.smi --policy_network policy_network.ckpt --results_dir planning_results

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``targets`` - the path to the file with target molecule for retrosynthetic planning.
    - ``reaction_rules`` - the path to the file with reaction rules.
    - ``building_blocks`` - the path to the file with building blocks.
    - ``policy_network`` - the path to the file with trained policy network (ranking or filtering).
    - ``value_network`` - the path to the file with trained value network if available (default is None).
    - ``results_dir`` - the path to the directory where the trained value network will be to be stored.

