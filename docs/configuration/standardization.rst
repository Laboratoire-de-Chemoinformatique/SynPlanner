.. _standardization_config:

================================
Reaction standardization
================================

``SynPlanner`` includes several reaction standardizers.
The list and order of application of standardizers can be specified in the configuration file.

Download example configuration
------------------------------

- GitHub: `configs/reactions_standardization.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/reactions_standardization.yaml>`_

Quickstart (CLI)
----------------

Run reaction standardization using the repository configuration in ``configs/reactions_standardization.yaml``:

.. code-block:: bash

   synplan reaction_standardizing \
     --config configs/reactions_standardization.yaml \
     --input reaction_data_original.smi \
     --output reaction_data_standardized.smi

**Configuration file**

.. code-block:: yaml

    kekule_form_config:
    functional_groups_config:
    remove_reagents_config:
    check_valence_config:
    implicify_hydrogens_config:
    check_isotopes_config:
    aromatic_form_config:
    unchanged_parts_config:
    deduplicate: true

**Configuration parameters**

.. table::
    :widths: 25 45 30

    ================================== ============================================================== ================================
    Reaction standardizer              Description                                                    Sub-parameters (bare-key value)
    ================================== ============================================================== ================================
    functional_groups_config           Standardization of functional groups                           none
    kekule_form_config                 Transform molecules to Kekule form when possible               none
    check_valence_config               Check atom valences                                            none
    implicify_hydrogens_config         Remove hydrogen atoms                                          none
    check_isotopes_config              Check and clean isotope atoms when possible                    none
    split_ions_config                  Split ions in reaction when possible                           none
    aromatic_form_config               Transform molecules to aromatic form when possible             none
    mapping_fix_config                 Fix atom-to-atom mapping in reaction when needed and possible   none
    unchanged_parts_config             Remove unchanged parts in reaction                             none
    small_molecules_config             Remove small molecule from reaction                            ``mol_max_size: 6``
    remove_reagents_config             Remove reagents from reaction                                  ``reagent_max_size: 7``
    rebalance_reaction_config          Add the molecules an unbalanced reaction is missing            ``add_redox_agents: false``, ``min_confidence: 0.0``, ``drop_competing_products: false``, ``use_mapping: true``, ``refuse_unsupported_redox: false``
    deduplicate                        Deduplicate reactions by CGR hash                              ``true`` (plain boolean, not a nested config)
    ================================== ============================================================== ================================


.. note::
    1. If the reaction standardizer name is listed in the configuration file (see above), it means that this standardizer will be applied.
    2. The configuration file enables standardization steps and sets their parameters; it does not define execution order.
    3. SynPlanner applies enabled standardizers in a canonical chemistry order. In the default pipeline this means Kekule conversion and functional-group normalization run before reagent removal and valence checking. Reagent removal runs before ``check_valence_config`` to avoid false valence errors from discarded reagent species, and aromatic conversion runs after valence-sensitive steps so duplicate detection sees a consistent final representation.
