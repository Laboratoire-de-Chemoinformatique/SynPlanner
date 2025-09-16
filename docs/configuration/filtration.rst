.. _filtration_config:

================================
Reaction filtration
================================

``SynPlanner`` includes a variety of reaction filters.
The list and order of application of filters can be specified in the configuration file.

Download example configuration
------------------------------

- GitHub: `configs/filtration.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/filtration.yaml>`_

Quickstart (CLI)
----------------

Run reaction filtration using the repository configuration in ``configs/filtration.yaml``:

.. code-block:: bash

   synplan reaction_filtering \
     --config configs/filtration.yaml \
     --input reaction_data_standardized.smi \
     --output reaction_data_filtered.smi

**Configuration file**

.. code-block:: yaml

    multi_center_config:
    no_reaction_config:
    dynamic_bonds_config:
      min_bonds_number: 1
      max_bonds_number: 6
    small_molecules_config:
      limit: 6

**Configuration parameters**

.. table::
    :widths: 35 50

    ================================== =================================================================================
    Reaction filter                    Description
    ================================== =================================================================================
    compete_products_config            Checks if there are compete reactions
    dynamic_bonds_config               Checks if there is an unacceptable number of dynamic bonds in CGR
    small_molecules_config             Checks if there are only small molecules in the reaction
    cgr_connected_components_config    Checks if CGR contains unrelated components (without reagents)
    rings_change_config                Checks if there is changing rings number in the reaction
    strange_carbons_config             Checks if there are 'strange' carbons in the reaction
    no_reaction_config                 Checks if there is no reaction in the provided reaction container
    multi_center_config                Checks if there is a multicenter reaction
    wrong_ch_breaking_config           Checks for incorrect C-C bond formation from breaking a C-H bond
    cc_sp3_breaking_config             Checks if there is C(sp3)-C bond breaking
    cc_ring_breaking_config            Checks if a reaction involves ring C-C bond breaking
    ================================== =================================================================================

.. note::
    1. If the reaction filter name is listed in the configuration file, it means that this filter will be activated.
    2. The order of filters listed in the configuration file defines the order of their application to the input reactions.
    3. ASome filters requires additional parameters (e.g. ``small_molecules_config``).