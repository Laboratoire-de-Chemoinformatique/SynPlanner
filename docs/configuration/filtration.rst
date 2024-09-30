.. _filtration:

================================
Reaction filtration
================================

The current recommendation is to divide these filters into two groups (standard filters and standard + special filters),
which can be set up by configuration yaml files (these default parameters are recommended).

Standard filters (4 filters):

.. code-block:: yaml

    multi_center_config:
    no_reaction_config:
    dynamic_bonds_config:
      min_bonds_number: 1
      max_bonds_number: 6
    small_molecules_config:
      limit: 6

Standard and special filters (4 + 3 filters):

.. code-block:: yaml

    multi_center_config:
    no_reaction_config:
    dynamic_bonds_config:
      min_bonds_number: 1
      max_bonds_number: 6
    small_molecules_config:
      limit: 6
    cc_ring_breaking_config:
    wrong_ch_breaking_config:
    cc_sp3_breaking_config:

**Important-1:** if the reaction filter name is listed in the configuration file (see above), it means that this filter will be activated. Also, some filters requires additional parameters (e.g. ``small_molecules_config``).

**Important-2:** the order of filters listed in the configuration file defines the order of their application to the input reactions.