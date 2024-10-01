.. _filtration:

================================
Reaction filtration
================================

Reaction filtration is a crucial step in reaction data curation. It ensures the validity of reactions
used for reaction rule extraction. The current version of ``SynPlanner`` includes 11 reaction filters (see below).
In brackets, it is shown how this filter should be listed in the configuration file to be activated.

The current available reaction filters in ``SynPlanner``:

- No reaction filter (checks if there is no reaction in the provided reaction container)
- Compete products filter (checks if there are competing reactions)
- Dynamic bonds filter (checks if there is an unacceptable number of dynamic bonds in CGR)
- Small molecules filter (checks if there are only small molecules in the reaction or if there is only one small reactant or product)
- CGR connected components filter (checks if CGR contains unrelated components (without reagents))
- Rings change filter (checks if there is a changing ring number in the reaction)
- Strange carbons filter (checks if there are 'strange' carbons in the reaction)
- Multi-center filter (checks if there is a multicenter reaction)
- Wrong CH-breaking filter (checks for incorrect C-C bond formation from breaking a C-H bond)
- CC-sp3-breaking filter (checks if there is C(sp3)-C bond breaking)
- CC-ring-breaking filter (checks if a reaction involves ring C-C bond breaking)
