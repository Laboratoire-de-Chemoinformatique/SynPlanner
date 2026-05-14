.. _api:

================
API
================


This page gives an overview of all public ``SynPlanner`` objects, functions and
methods. All classes and functions exposed in ``synplan.*`` namespace are public.

The following subpackages are public.

- ``synplan.chem``: The module for curation (standardization and filtration) of reaction and molecular data.
- ``synplan.interfaces``: Functions for building interfaces, currently command line interface added. For the graphical user interface, see the `HuggingFace repository <https://huggingface.co/spaces/Laboratoire-De-Chemoinformatique/SynPlanner/tree/main>`_.
- ``synplan.mcts``: Functions and classes responsible for Monte-Carlo Tree Search.
- ``synplan.ml``: Functions that are used to train policy and value networks.
- ``synplan.route_quality``: Post-search route analysis for competing functional group detection and route re-ranking by selectivity. Inspired by Westerlund et al., *ChemRxiv*, 2025 (`doi:10.26434/chemrxiv-2025-gdrr8 <https://doi.org/10.26434/chemrxiv-2025-gdrr8>`_).
- ``synplan.utils``: Functions used for configuring ``SynPlanner``, loading all the data, logging and visualisation of synthetic routes.

.. toctree::
    :hidden:
    :titlesonly:
    :maxdepth: 4

    api_reference/modules.rst



