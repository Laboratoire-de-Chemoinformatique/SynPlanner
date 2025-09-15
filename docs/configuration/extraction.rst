.. _extraction_config:

================================
Reaction rules
================================

The reaction rules extraction protocol is flexible and can adjust the specificity of extracted reaction rules.

Download example configuration
--------------------------------

- GitHub: `configs/extraction.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/extraction.yaml>`_

Quickstart (CLI)
----------------

Extract rules using the repository configuration in ``configs/extraction.yaml``:

.. code-block:: bash

   synplan rule_extracting \
     --config configs/extraction.yaml \
     --input reaction_data_filtered.smi \
     --output reaction_rules.pickle

**Configuration file**

.. code-block:: yaml

    min_popularity: 3
    environment_atom_count: 1
    multicenter_rules: True
    include_rings: False
    include_func_groups: False
    func_groups_list: []
    keep_leaving_groups: False
    keep_incoming_groups: False
    keep_reagents: False
    atom_info_retention:
      reaction_center:
        neighbors: True
        hybridization: True
        implicit_hydrogens: False
        ring_sizes: False
      environment:
        neighbors: False
        hybridization: False
        implicit_hydrogens: False
        ring_sizes: False

**Configuration parameters**

.. table::
    :widths: 30 50

    ================================== =================================================================================
    Parameter                          Description
    ================================== =================================================================================
    environment_atom_count             Determines the number of layers of atoms around the reaction center to be included in the reaction rule. A value of 0 includes only the reaction center, 1 includes the first surrounding layer, and so on.
    min_popularity                     Determines the minimum number of occurrences of a reaction rule in the reaction dataset.
    multicenter_rules                  Determines whether a single rule is extracted for all centers in multicenter reactions (True) or if separate rules are generated for each center (False).
    include_rings                      Includes ring structures in the reaction rules connected to the reaction center atoms if set to True.
    include_func_groups                If True, specific functional groups are included in the reaction rule in addition to the reaction center and its environment.
    func_groups_list                   Specifies a list of functional groups to be included when include_func_groups is True.
    keep_leaving_groups                Keeps the leaving groups in the extracted reaction rule when set to True.
    keep_incoming_groups               Retains incoming groups in the extracted reaction rule if set to True.
    keep_reagents                      Includes reagents in the extracted reaction rule when True.
    atom_info_retention                Dictates the level of detail retained about atoms in the reaction center and their environment.
    ================================== =================================================================================
