.. _extraction:

================================
Reaction rules
================================

Extraction protocol
-----------------------------

The protocol for reaction rule extraction in SynTool includes several steps:

**1. CGR creation**

The input reaction is converted to the Condensed Graph of Reaction (CGR), which is a single graph encoding an ensemble
of reactants and products. CGR results from the superposition of the atoms of products and reactants having the same numbers.
It contains both conventional chemical bonds (single, double, triple, aromatic, etc.) and so-called “dynamic” bonds describing
chemical transformations, i.e. breaking or forming a bond or changing bond order. Given CGRs it is possible to extract the
reaction center of the reaction.

**2. Reaction center extraction**

Some reactions can have several reaction centers and if the ``multicenter_rules`` parameter is ``True``, these centers will be
included in a single reaction rule. Otherwise, each reaction center will be included in different reaction rules.

**3. Reaction center extension**

The extracted reaction center can be extended by the inclusion of neighboring atoms (environmental atoms) of radius N
(can be specified by ``environment_atom_count`` parameter). Usually, the environment of radius 1 is included.

**4. Ring structures inclusion**

If the reaction center atoms are part of ring structures or functional groups, they can be included in the reaction center.
Ring structures are identified by the Smallest Set of Smallest Rings (SSSR) algorithm implemented in CGRTools.
If the ``include_rings`` parameter is ``True`` and any atom of the reaction center is part of the identified ring structure,
the whole ring is included in the reaction center.

**5. Functional group inclusion**

Functional groups to be included in the reaction center should be specified manually as a list of their SMILES in
``func_groups_list`` in the configuration file. If ``include_func_groups`` is ``True``, any atom of the reaction center is part of
any specified functional group, and the whole functional group is included in the reaction center.

**6. Leaving/incoming groups inclusion**

Leaving and incoming groups can be included in the reaction center. These groups are identified by comparison of
the atoms in reactants and products. If some atoms in reactants are not observed in products, these atoms are identified
as a leaving group. Likewise, if some atoms in products are not observed in reactants, these atoms are identified
as an incoming group. This functionality is supposed to handle protection/deprotection reactions and identify protective agents.
Leaving groups are added to the reaction center if the ``keep_leaving_groups`` parameter is ``True`` and incoming groups are added
to the reaction center if ``keep_incoming_groups`` parameter is ``True``.

**7. Reagents inclusion**

Reaction rule can be further specified by the inclusion of reagents if the ``keep_reagents`` parameter is ``True``.
It means that structurally identical reaction rules with identical atom properties will be considered different
if they are associated with different reagents.

**8. Atom properties specification**

Each atom in the extended and specified reaction center and its environment atoms (neighbor atoms) is described by four properties:
the number of neighbors, hybridization type, the number of hydrogens, and the size of the ring (if the atom belongs to any ring).
These properties determine the level of atom and reaction rule specification and can be disabled if needed in the
``atom_info_retention`` section in the configuration file. If some property in ``atom_info_retention`` is ``True``, it means that this property of
the atom will be included in the final reaction rule.

**9. Reaction rule creation**

After all settings, the final reaction rule is assembled, reversed (to be used in retrosynthesis mode),
and validated by application of the final reaction rule to the original reaction, from which it was extracted.

**10. Reaction rule validation**

Finally, the extracted rules are filtered by popularity, which is defined by the ``min_popularity`` parameter.
For example, ``min_popularity:3`` means, that only rules observed in not less than 3 reactions from the reaction dataset are remained.


Functional groups
-----------------------------

If reaction center atoms and their neighboring atoms are part of some specific substructural motif,
they should be also included in the reaction rule for a better description of the chemical context or reaction.
These motifs can be some “functional groups” with specific electronic and steric properties that influence
the reactivity of reaction center atoms and may define the reaction performance.

The list of functional groups can be specified in the configuration file, where each group is represented by ``chython`` SMARTS.

.. tip::
    The ``chython`` SMARTS definition is slightly different from the popular `Daylight SMARTS definition <https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html>`_, please consult the official ``chython`` documentation `here <https://chython.readthedocs.io/en/latest/>`_.

In ``SynPlanner``, roughly 25 functional groups from ``Coley, Connor W., JCIM., 59.6 (2019): 2529-2537`` are available in the default configuration file.

**Important:** currently in ``SynPlanner`` there is no exact specification of atom in the functional group, which must
intersect with reaction center atoms or their neighbors for inclusion of the whole functional group to the reaction rule.
It means that if any atom functional group intersects with reaction center atoms or their neighbors, the functional group
will be included in the reaction rule. It sometimes leads to multiple inclusions of the same functional group in the reaction,
which makes the final reaction rule more specific.


