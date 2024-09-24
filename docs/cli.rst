.. _cli:

======================
Command-line interface
======================

Reaction standardization
---------------------------
Reaction standardization can be performed with the below command.

.. code-block:: bash

    synplan reaction_standardizing --config standardization.yaml --input reaction_data_mapped.smi --output reaction_data_standardized.smi

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``input`` - the path to the file (.smi or .rdf) with reactions to be standardized.
    - ``output`` - the path to the file (.smi or .rdf) where standardized reactions will be stored.

The extension of the input/output files will be automatically parsed.


Reaction filtration
---------------------------
Reaction filtration can be performed with the below command.

.. code-block:: bash

    synplan reaction_filtering --config filtration.yaml --input reaction_data_standardized.smi --output reaction_data_filtered.smi

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``input`` - the path to the file (.smi or .rdf) with reactions to be filtered.
    - ``output`` - the path to the file (.smi or .rdf) where filtered reactions to be stored.

The extension of the input/output files will be automatically parsed.


Reaction rule extraction
---------------------------
Reaction rules extraction can be performed with the below command.

.. code-block:: bash

    synplan rule_extracting --config extraction.yaml --input reaction_data_filtered.smi --output reaction_rules.pickle

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``input`` - the path to the file (.smi or .rdf) with reactions for reaction rule extraction.
    - ``output`` - the path to the file (.pickle) where extracted reactions rules will be stored.

The extension of the input/output files will be automatically parsed.


Policy networks training
---------------------------
Ranking and filtering policy network training can be performed with the below commands.

**Ranking policy network**

.. code-block:: bash

    synplan ranking_policy_training --config policy.yaml --reaction_data reaction_data_filtered.smi --reaction_rules reaction_rules.pickle --results_dir ranking_policy_network

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``reaction_data`` - the path to the file with reactions for ranking policy training.
    - ``reaction_rules`` - the path to the file with extracted reaction rules.
    - ``results_dir`` - the path to the directory where the trained policy network will be stored.

**Filtering policy network**

.. code-block:: bash

    synplan filtering_policy_training --config policy.yaml --molecule_data molecules_data.smi --reaction_rules reaction_rules.pickle --results_dir filtering_policy_network

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``molecule_data`` - the path to the file with molecules for filtering policy training.
    - ``reaction_rules`` - the path to the file with extracted reaction rules.
    - ``results_dir`` - the path to the directory where the trained policy network will be stored.


Value network training
---------------------------
Value network training can be performed with the below command.

**Important:** If you use your custom building blocks, be sure to canonicalize them before planning simulations in value network tuning.

.. code-block:: bash

    synplan building_blocks_canonicalizing --input building_blocks_init.smi --output building_blocks.smi
    synplan value_network_tuning --config tuning.yaml --targets targets.smi --reaction_rules reaction_rules.pickle --policy_network policy_network.ckpt --building_blocks building_blocks.smi --results_dir value_network

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``targets`` - the path to the file with target molecules for planning simulations.
    - ``reaction_rules`` - the path to the file with reactions rules.
    - ``building_blocks`` - the path to the file with building blocks.
    - ``policy_network`` - the path to the file with trained policy network (ranking or filtering policy network).
    - ``results_dir`` - the path to the directory where the trained value network will be to be stored.


Retrosynthetic planning
---------------------------
Retrosynthetic planning can be performed with the below command.
If you use your custom building blocks, be sure to canonicalize them before planning.

.. code-block:: bash

    synplan building_blocks_canonicalizing --input building_blocks_init.smi --output building_blocks.smi
    synplan planning --config planning.yaml --targets targets.smi --reaction_rules reaction_rules.pickle --building_blocks building_blocks_stand.smi --policy_network policy_network.ckpt --results_dir planning

**Parameters**:
    - ``config`` - the path to the configuration file.
    - ``targets`` - the path to the file with target molecule for retrosynthetic planning.
    - ``reaction_rules`` - the path to the file with reaction rules.
    - ``building_blocks`` - the path to the file with building blocks.
    - ``policy_network`` - the path to the file with trained policy network (ranking or filtering).
    - ``value_network`` - the path to the file with trained value network if available (default is None).
    - ``results_dir`` - the path to the directory where the trained value network will be to be stored.
