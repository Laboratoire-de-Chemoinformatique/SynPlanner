.. _data_download:

Data download
===========================

This page explains how to download data for retrosynthetic models training and retrosynthesis planning in SynPlanner.

Introduction
---------------------------
**Retrosynthetic models training.** For the training of the retrosynthetic models (policy and value network, reaction rules)
the following types of data are needed:

.. table::
    :widths: 15 50

    ======================= ============================================================================================
    Data                    Description
    ======================= ============================================================================================
    Reaction data           Needed for reaction rules extraction and ranking policy network training
    Molecule data           Needed for filtering policy network training
    Targets data            Needed for value network training (targets for planning simulations in value network tuning)
    Building blocks         Needed for retrosynthesis planning simulations in value network tuning
    ======================= ============================================================================================

**Retrosynthesis planning.** For the retrosynthesis planning the following data and files are needed:

.. table::
    :widths: 15 50

    ======================= ============================================================================================
    Data / Files            Description
    ======================= ============================================================================================
    Reaction rules          Extracted reaction rules for precursors dissection in retrosynthesis planning
    Policy network          Trained ranking or filtering policy network for node expansion in tree search
    Value network           Trained value neural network for node evaluation in tree search (optional, the default evaluation method is rollout)
    Building blocks         Set of building block molecules, which are used as terminal materials in the retrosynthesis route planning
    ======================= ============================================================================================

As a source of reaction and molecule data public databases are used such as USPTO, ChEMBL, and COCONUT.

**Important:** the current available data formats are SMILES (.smi) and RDF (.rdf) for reactions and SMILES (.smi) and SDF (.sdf) for molecules.
The extracted reaction rules are stored as CGRTools objects in a pickle file and currently cannot be stored in text format (e.g. reaction SMARTS).

Configuration
---------------------------
Data download does not require any special configuration in the current version of SynPlanner.

CLI
---------------------------
Data download can be performed with the below commands.

**Data download for retrosynthetic models training**

.. code-block:: bash

    synplan download_training_data your/local/dir

By default, the files will be stored in the ``your/local/dir`` directory.

**Data download for retrosythesis planning**

.. code-block:: bash

    synplan download_planning_data your/local/dir

By default, the files will be stored in the ``your/local/dir`` directory in the current location (./).
