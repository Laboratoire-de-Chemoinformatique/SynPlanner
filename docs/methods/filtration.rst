.. _filtration:

================================
Reaction filtration
================================

Reaction filtration is a required step in reaction data curation. It ensures the validity of reactions
used for reaction rule extraction. The current version of ``SynPlanner`` includes 11 reaction filters (see below).
The name in ``code font`` is the key to list in the configuration file to activate that filter.

The current available reaction filters in ``SynPlanner``:

- ``no_reaction_config`` — No reaction filter: checks if there is no reaction in the provided reaction container
- ``compete_products_config`` — Compete products filter: checks if there are competing reactions
- ``dynamic_bonds_config`` — Dynamic bonds filter: checks if there is an unacceptable number of dynamic bonds in CGR
- ``small_molecules_config`` — Small molecules filter: checks if there are only small molecules in the reaction or if there is only one small reactant or product
- ``cgr_connected_components_config`` — CGR connected components filter: checks if CGR contains unrelated components (without reagents)
- ``rings_change_config`` — Rings change filter: checks if there is a changing ring number in the reaction
- ``strange_carbons_config`` — Strange carbons filter: checks if there are 'strange' carbons in the reaction
- ``multi_center_config`` — Multi-center filter: checks if there is a multicenter reaction
- ``wrong_ch_breaking_config`` — Wrong CH-breaking filter: checks for incorrect C-C bond formation from breaking a C-H bond
- ``cc_sp3_breaking_config`` — CC-sp3-breaking filter: checks if there is C(sp3)-C bond breaking
- ``cc_ring_breaking_config`` — CC-ring-breaking filter: checks if a reaction involves ring C-C bond breaking

Only ``no_reaction_config``, ``dynamic_bonds_config``, ``small_molecules_config`` and
``cc_sp3_breaking_config`` are enabled by ``configs/reactions_filtration.yaml``. A filter
whose key is absent from the configuration file is off; none of them are on by default.
See :doc:`/configuration/filtration` for each filter's parameters and their defaults.
