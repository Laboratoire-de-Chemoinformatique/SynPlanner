Data download
--------------

Use the built-in downloader to fetch pre-trained models, reaction rules, and building blocks from HuggingFace.

Preset download (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download a ready-to-use preset with all components needed for retrosynthetic planning:

.. code-block:: bash

   synplan download_preset --preset synplanner-article --save_to synplan_data

This downloads the ``synplanner-article`` preset, which includes:

- Reaction rules (TSV): ``policy/supervised_gcn/v1/reaction_rules.tsv``
- Ranking policy weights: ``policy/supervised_gcn/v1/v1/ranking_policy.ckpt``
- Filtering policy weights: ``policy/supervised_gcn/v1/v1/filtering_policy.ckpt``
- Value network weights: ``value/supervised_gcn/v1/value_network.ckpt``
- Building blocks: ``building_blocks/emolecules-salt-ln/building_blocks.tsv``

Python API:

.. code-block:: python

   from synplan.utils.loading import download_preset

   paths = download_preset("synplanner-article", save_to="synplan_data")
   rules_path = paths["reaction_rules"]
   policy_path = paths["ranking_policy"]
   bb_path = paths["building_blocks"]

Details
~~~~~~~

For a full list of datasets and descriptions, see :doc:`/user_guide/data`.


