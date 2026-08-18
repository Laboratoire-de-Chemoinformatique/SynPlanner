Data download
--------------

Use the built-in downloader to fetch pre-trained models, reaction rules, and building blocks from HuggingFace.

Preset download (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download a ready-to-use preset with all components needed for retrosynthetic planning:

.. code-block:: bash

   synplan download_preset --preset synplanner-gps --save_to synplan_data

This downloads the ``synplanner-gps`` preset, which includes:

- Reaction rules (TSV): ``policy/supervised_gps/v1/reaction_rules.tsv``
- Ranking policy weights: ``policy/supervised_gps/v1/v1/ranking_policy.ckpt``
- Filtering policy weights: ``policy/supervised_gcn/v1/v1/filtering_policy.ckpt``
- Value network weights: ``value/supervised_gcn/v1/value_network.ckpt``
- Building blocks: ``building_blocks/emolecules-salt-ln/building_blocks.tsv``

The command prints the local path of every downloaded file, so use that output
rather than retyping paths.

.. warning::
    The rules and the ranking head of ``synplanner-gps`` come from
    ``supervised_gps`` (11235 rules); its filtering head comes from
    ``supervised_gcn`` (24094 rules). The two heads therefore cannot be combined
    (see :doc:`/configuration/planning`). If you need a matched filtering +
    ranking pair, download ``--preset synplanner-article`` instead: all of its
    components come from ``supervised_gcn/v1``.

Python API:

.. code-block:: python

   from synplan.utils.loading import download_preset

   paths = download_preset("synplanner-gps", save_to="synplan_data")
   rules_path = paths["reaction_rules"]
   policy_path = paths["ranking_policy"]
   bb_path = paths["building_blocks"]

SAScore benchmark
~~~~~~~~~~~~~~~~~

Download the published 100-target subsets used by ``sascore-benchmark``:

.. code-block:: bash

   synplan download_sascore_benchmark --save_to synplan_data

The files are saved under ``synplan_data/benchmarks/sascore/subset_100``. The
equivalent Python API is:

.. code-block:: python

   from synplan.utils.loading import download_sascore_benchmark

   benchmark_files = download_sascore_benchmark(save_to="synplan_data")

Details
~~~~~~~

For a full list of datasets and descriptions, see :doc:`/user_guide/data`.


