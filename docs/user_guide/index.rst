User Guide
==========

This guide is a practical, task-oriented path through SynPlanner.

What's inside
-------------

- Concepts and algorithms → see :doc:`/methods/methods`.
- Configuration of pipelines (YAML and Python) → see :doc:`/configuration/configuration`.
- CLI usage → see :doc:`cli_interface`.

Run in Google Colab
-------------------

Open selected tutorials directly in Google Colab (no local install):

- Retrosynthetic planning |Colab-Retro|
- SynPlanner benchmarking |Colab-Bench|
- Route clustering by strategic bonds |Colab-Cluster|

.. |Colab-Retro| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/retrosynthetic_planning.ipynb
   :alt: Open in Colab

.. |Colab-Bench| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/planning_benchmarking.ipynb
   :alt: Open in Colab

.. |Colab-Cluster| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/routes_clustering.ipynb
   :alt: Open in Colab

Tutorials
-------------------

``SynPlanner`` is supplemented by several tutorials explaining different aspects of tool usage. These tutorials included
some advanced explanations and configurations, that can be used not only for demonstrative and educational purposes but
also for creating flexible pipelines.

**Introductory tutorials:**

- `Welcome to Chython`_ - an introduction to the chython library and its core concepts.
- `Coming from RDKit`_ - a side-by-side comparison for users familiar with RDKit.
- `10 minutes to SynPlanner`_ - a quickstart guide to run planning via Python or CLI.

**Pipeline tutorials:**

- `Data Curation`_ - demonstrates how to prepare data (reaction standardization and filtration) before reaction rules extraction and retrosynthetic model training.
- `Rules Extraction`_ - demonstrates how to extract reaction rules from reaction data.
- `Policy Training`_ - demonstrates how to train ranking and filtering policy networks.
- `Retrosynthetic Planning`_ - demonstrates how retrosynthetic planning can be performed for target molecules.
- `Clustering`_ - demonstrates how to cluster predicted retrosynthetic routes.
- `Protection Scoring`_ - demonstrates how to detect competing functional groups and score routes for selectivity issues, inspired by `Westerlund et al. (2025) <https://doi.org/10.26434/chemrxiv-2025-gdrr8>`_.

**Advanced tutorials:**

- `Combined Ranking and Filtering Policy`_ - demonstrates how to combine ranking and filtering policy networks.
- `NMCS Algorithms`_ - demonstrates Nested Monte Carlo Search algorithms for retrosynthetic planning.
- `Tree Analysis`_ - demonstrates how to analyze tree search results: policy performance, winning rule ranks, branching profile, and route details.
- `Planning with RDKit`_ - demonstrates how to use SynPlanner with RDKit Mol objects for input and output.
- `Rule Analysis`_ - demonstrates how to analyze and visualize reaction rules.
- `Priority Rules`_ - demonstrates how to create custom retrosynthetic planner with user defined retrosynthetic SMARTS.

.. _Welcome to Chython: 00_Welcome_to_Chython.ipynb
.. _Coming from RDKit: 01_Coming_from_RDKit.ipynb
.. _10 minutes to SynPlanner: ten_minutes.rst
.. _Data Curation: 02_Data_Curation.ipynb
.. _Rules Extraction: 03_Rules_Extraction.ipynb
.. _Policy Training: 04_Policy_Training.ipynb
.. _Retrosynthetic Planning: 05_Retrosynthetic_Planning.ipynb
.. _Clustering: 06_Clustering.ipynb
.. _Protection Scoring: 07_Protection_Scoring.ipynb
.. _Combined Ranking and Filtering Policy: 08_Combined_Ranking_Filtering_Policy.ipynb
.. _NMCS Algorithms: 09_NMCS_Algorithms.ipynb
.. _Tree Analysis: 10_Tree_Analysis.ipynb
.. _Planning with RDKit: 11_Planning_with_RDKit.ipynb
.. _Rule Analysis: 12_Rule_Analysis.ipynb
.. _Priority Rules: 13_Priority_Rules.ipynb

.. toctree::
    :hidden:
    :titlesonly:

    ten_minutes
    cli_interface
    data
    00_Welcome_to_Chython
    01_Coming_from_RDKit
    02_Data_Curation
    03_Rules_Extraction
    04_Policy_Training
    05_Retrosynthetic_Planning
    06_Clustering
    07_Protection_Scoring
    08_Combined_Ranking_Filtering_Policy
    09_NMCS_Algorithms
    10_Tree_Analysis
    11_Planning_with_RDKit
    12_Rule_Analysis
    13_Priority_Rules
    ../configuration/configuration
    ../methods/methods
   migration