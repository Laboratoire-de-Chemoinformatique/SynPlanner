.. _standardization:

================================
Reaction standardization
================================

Reaction standardization protocol can be adjusted configuration yaml file (these default parameters bellow are recommended).

.. code-block:: yaml

    reaction_mapping_config:
    functional_groups_config:
    kekule_form_config:
    check_valence_config:
    implicify_hydrogens_config:
    check_isotopes_config:
    split_ions_config:
    aromatic_form_config:
    mapping_fix_config:
    unchanged_parts_config:
    duplicate_reaction_config:

**Important-1:** if the reaction standardizer name is listed in the configuration file (see above), it means that this filter will be activated.

**Important-2:** the order of standardizers listed in the configuration file defines the order of their application to the input reactions.