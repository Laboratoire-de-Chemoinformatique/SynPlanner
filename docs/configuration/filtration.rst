.. _filtration_config:

================================
Reaction filtration
================================

``SynPlanner`` includes several reaction filters.
The list and order of application of filters can be specified in the configuration file.

Download example configuration
------------------------------

- GitHub: `configs/reactions_filtration.yaml <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/configs/reactions_filtration.yaml>`_

Quickstart (CLI)
----------------

Run reaction filtration using the repository configuration in ``configs/reactions_filtration.yaml``:

.. code-block:: bash

   synplan reaction_filtering \
     --config configs/reactions_filtration.yaml \
     --input reaction_data_standardized.smi \
     --output reaction_data_filtered.smi

**Configuration file**

.. code-block:: yaml

    no_reaction_config:
    dynamic_bonds_config:
      min_bonds_number: 1
      max_bonds_number: 12
    small_molecules_config:
      mol_max_size: 6
    cc_sp3_breaking_config:

**Configuration parameters**

.. table::
    :widths: 25 40 35

    ================================== ================================================================= ================================================================
    Reaction filter                    Description                                                       Sub-parameters (value used when the key is left bare)
    ================================== ================================================================= ================================================================
    compete_products_config            Checks if there are compete reactions                             ``fingerprint_tanimoto_threshold: 0.3``, ``mcs_tanimoto_threshold: 0.6``
    dynamic_bonds_config               Checks if there is an unacceptable number of dynamic bonds in CGR ``min_bonds_number: 1``, ``max_bonds_number: 6`` (the shipped config raises this to ``12`` — see the warning below)
    small_molecules_config             Checks for small molecules in the reaction (see note 5)           ``mol_max_size: 6``
    cgr_connected_components_config    Checks if CGR contains unrelated components (without reagents)    none
    rings_change_config                Checks if there is changing rings number in the reaction          none
    strange_carbons_config             Checks if there are 'strange' carbons in the reaction             none
    no_reaction_config                 Checks if there is no reaction in the provided reaction container none
    multi_center_config                Checks if there is a multicenter reaction                         none
    wrong_ch_breaking_config           Checks for incorrect C-C bond formation from breaking a C-H bond  none
    cc_sp3_breaking_config             Checks if there is C(sp3)-C bond breaking                         ``decoration_depth: 1``, ``ring_max_size: 7``
    cc_ring_breaking_config            Checks if a reaction involves ring C-C bond breaking              none
    ================================== ================================================================= ================================================================

.. warning::
    The ``DynamicBondsConfig`` model default is ``max_bonds_number: 6``, but
    ``configs/reactions_filtration.yaml`` deliberately sets ``12``. Writing a bare
    ``dynamic_bonds_config:`` therefore silently rejects every reaction that changes
    7–12 bonds — a large and chemically legitimate population in commercial datasets
    (multicomponent condensations, tandem/cascade steps, protection + coupling in one
    record). Nothing crashes and no warning is printed; the reactions simply disappear.
    Always set ``min_bonds_number`` / ``max_bonds_number`` explicitly.

    The same applies to the whole table: a bare key means "the model default", which is
    not always the value the shipped config uses. Start from
    ``configs/reactions_filtration.yaml`` rather than from this list.

.. note::
    1. If the reaction filter name is listed in the configuration file, it means that this filter will be activated.
    2. The configuration file enables filters and sets their parameters; it does not define execution order. Filters run in the fixed order of :meth:`~synplan.chem.data.filtering.ReactionFilterConfig.create_filters`.
    3. To disable a filter, omit its key entirely. A bare ``key:`` enables the filter with the defaults in the third column.
    4. The shipped configuration enables only four of these eleven filters. The other seven are opt-in; enabling all of them is a much more aggressive curation than the SynPlanner default and will discard a substantially larger fraction of any dataset.
    5. ``small_molecules_config`` rejects a reaction in three cases, not one: every reactant *and* every product is at most ``mol_max_size`` heavy atoms; **or** there is exactly one reactant and it is small; **or** there is exactly one product and it is small. The last two cases fire even when the other side of the reaction is large — a 10-heavy-atom reactant giving methanol as its only product is rejected at the default ``mol_max_size: 6``.
    6. Filtration re-applies kekulization and aromatization before the filters run, so a reaction that passes may be written out with a SMILES string that differs from the input.