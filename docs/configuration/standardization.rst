.. _standardization_config:

================================
Reaction standardization
================================

``SynPlanner`` includes a variety of reaction standardizers.
The list and order of application of standardizers can be specified in the configuration file.

Download example configuration
------------------------------

- GitHub: `configs/standardization.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/standardization.yaml>`_

Quickstart (CLI)
----------------

Run reaction standardization using the repository configuration in ``configs/standardization.yaml``:

.. code-block:: bash

   synplan reaction_standardizing \
     --config configs/standardization.yaml \
     --input reaction_data_original.smi \
     --output reaction_data_standardized.smi

**Configuration file**

.. code-block:: yaml

    reaction_mapping_config:
    functional_groups_config:
    kekule_form_config:
    check_valence_config:
    implicify_hydrogens_config:
    check_isotopes_config:
    aromatic_form_config:
    mapping_fix_config:
    unchanged_parts_config:
    duplicate_reaction_config:

**Configuration parameters**

.. table::
    :widths: 30 50

    ================================== =================================================================================
    Reaction standardizer              Description
    ================================== =================================================================================
    reaction_mapping_config            Maps atoms of the reaction using chython (chytorch)
    functional_groups_config           Standardization of functional groups
    kekule_form_config                 Transform molecules to Kekule form when possible
    check_valence_config               Check atom valences
    implicify_hydrogens_config         Remove hydrogen atoms
    check_isotopes_config              Check and clean isotope atoms when possible
    split_ions_config                  Split ions in reaction when possible
    aromatic_form_config               Transform molecules to aromatic form when possible
    mapping_fix_config                 Fix atom-to-atom mapping in reaction when needed and possible
    unchanged_parts_config             Remove unchanged parts in reaction
    small_molecules_config             Remove small molecule from reaction
    remove_reagents_config             Remove reagents from reaction
    rebalance_reaction_config          Rebalance reaction
    duplicate_reaction_config          Remove duplicate reactions
    ================================== =================================================================================

.. note::
    1. If the reaction standardizer name is listed in the configuration file (see above), it means that this standardizer will be applied.
    2. The order of standardizers listed in the configuration file defines the order of their application to the input reactions.