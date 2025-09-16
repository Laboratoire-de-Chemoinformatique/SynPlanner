Data download
--------------

Use the built-in downloader to fetch all example data (rules, weights, building blocks, benchmarks) from Hugging Face.

Everything
~~~~~~~~~~~

.. code-block:: bash

   synplan download_all_data --save_to tutorials/synplan_data

This creates subfolders like ``uspto/``, ``weights/``, and ``building_blocks/`` under ``tutorials/synplan_data``.

Minimal set for planning
~~~~~~~~~~~~~~~~~~~~~~~~

You typically need:

- Reaction rules: ``uspto/uspto_reaction_rules.pickle``
- Policy weights: ``uspto/weights/ranking_policy_network.ckpt``
- Building blocks: ``building_blocks/building_blocks_em_sa_ln.smi`` (unzipped)
- Targets file: a text file with SMILES, e.g. ``tutorials/synplan_data/benchmarks/sascore/targets_with_sascore_1.5_2.5.smi``

Details
~~~~~~~

For a full list of datasets and descriptions, see :doc:`/user_guide/data`.


