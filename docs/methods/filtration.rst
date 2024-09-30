.. _filtration:

================================
Reaction filtration
================================

Reaction filtration is a crucial step in reaction data curation. It ensures the validity of reactions
used for reaction rule extraction. The current version of SynPlanner includes 11 reaction filters (see below).
In brackets, it is shown how this filter should be listed in the configuration file to be activated.

The current available reaction filters in SynPlanner:

.. table::
    :widths: 35 50

    ================================== =================================================================================
    Reaction filter                    Description
    ================================== =================================================================================
    compete_products_config            Checks if there are compete reactions
    dynamic_bonds_config               Checks if there is an unacceptable number of dynamic bonds in CGR
    small_molecules_config             Checks if there are only small molecules in the reaction or if there is only one small reactant or product
    cgr_connected_components_config    Checks if CGR contains unrelated components (without reagents)
    rings_change_config                Checks if there is changing rings number in the reaction
    strange_carbons_config             Checks if there are 'strange' carbons in the reaction
    no_reaction_config                 Checks if there is no reaction in the provided reaction container
    multi_center_config                Checks if there is a multicenter reaction
    wrong_ch_breaking_config           Checks for incorrect C-C bond formation from breaking a C-H bond
    cc_sp3_breaking_config             Checks if there is C(sp3)-C bond breaking
    cc_ring_breaking_config            Checks if a reaction involves ring C-C bond breaking
    ================================== =================================================================================