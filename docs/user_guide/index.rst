User Guide
==========

This guide is a practical, task-oriented path through SynPlanner.

What's inside
-------------

- Concepts and algorithms: see :doc:`/methods/methods`.
- Configuration of pipelines (YAML and Python): see :doc:`/configuration/configuration`.
- CLI usage: see :doc:`cli_interface`.

Run in Google Colab
-------------------

Every tutorial runs in Google Colab with no local install. Open the tutorial you
want and click the |Colab| badge at the top of it: the setup cell installs
SynPlanner and downloads the public data it needs, and does nothing when the
notebook is run locally. Tutorial 20 additionally requires a prepared
vendor-aware InChIKey JSON catalogue.

.. |Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Open In Colab

Tutorials
-------------------

``SynPlanner`` is supplemented by several tutorials explaining different aspects of tool usage. These tutorials included
some advanced explanations and configurations, that can be used for both demonstration and building custom pipelines.

**Introductory tutorials:**

- `Welcome to Chython`_ - an introduction to the chython library and its core concepts.
- `Coming from RDKit`_ - a side-by-side comparison for users familiar with RDKit.
- `10 minutes to SynPlanner`_ - a quickstart guide to run planning via Python or CLI.

**Pipeline tutorials:**

- `Data Curation`_ - demonstrates how to prepare data (reaction standardization and filtration) before reaction rules extraction and retrosynthetic model training.
- `Rules Extraction`_ - demonstrates how to extract reaction rules from reaction data.
- `Policy Training`_ - demonstrates how to train ranking and filtering policy networks.
- `Retrosynthetic Planning`_ - demonstrates how retrosynthetic planning can be performed for target molecules.
- `Tree Analysis`_ - demonstrates how to analyze tree search results: policy performance, winning rule ranks, branching profile, and route details.
- `Clustering`_ - demonstrates how to cluster predicted retrosynthetic routes.
- `Protection Scoring`_ - demonstrates how to detect competing functional groups and score routes for selectivity issues, inspired by `Westerlund et al. (2026) <https://doi.org/10.1021/acs.jcim.6c01147>`_.

**Advanced tutorials:**

- `Combined Ranking and Filtering Policy`_ - demonstrates how to combine ranking and filtering policy networks.
- `NMCS Algorithms`_ - demonstrates Nested Monte Carlo Search algorithms for retrosynthetic planning.
- `Planning with RDKit`_ - demonstrates how to use SynPlanner with RDKit Mol objects for input and output.
- `Rule Analysis`_ - demonstrates how to analyze and visualize reaction rules.
- `Priority Rules`_ - demonstrates how to create custom retrosynthetic planner with user defined retrosynthetic SMARTS.
- `MHN Ranking Training`_ - demonstrates how to train and fine-tune the MHN ranking policy architecture.
- `Routes Compare`_ - demonstrates how to compare route sets with RouteCGRs.
- `Building Block Search`_ - demonstrates building-block search utilities.
- `Synthon-Based Library Design`_ - demonstrates the synthon subsystem: building-block classification and synthonisation, target fragmentation, availability against a synthon stock, library and analogue enumeration, positional analogue scanning, and ring closure.
- `Retrosynthesis with Synthon Priority Rules`_ - demonstrates how the shipped synthon disconnections steer an MCTS search as a curated priority-rule set.
- `InChIKey Building-Block Catalogue`_ - uses a prepared vendor-aware JSON catalogue for stereo-aware Boceprevir MCTS and detached route costing.

.. _Welcome to Chython: 00_Welcome_to_Chython.ipynb
.. _Coming from RDKit: 01_Coming_from_RDKit.ipynb
.. _10 minutes to SynPlanner: ten_minutes.rst
.. _Data Curation: 02_Data_Curation.ipynb
.. _Rules Extraction: 03_Rules_Extraction.ipynb
.. _Policy Training: 04_Policy_Training.ipynb
.. _Retrosynthetic Planning: 05_Retrosynthetic_Planning.ipynb
.. _Tree Analysis: 06_Tree_Analysis.ipynb
.. _Clustering: 07_Clustering.ipynb
.. _Protection Scoring: 08_Protection_Scoring.ipynb
.. _Combined Ranking and Filtering Policy: 09_Combined_Ranking_Filtering_Policy.ipynb
.. _NMCS Algorithms: 10_NMCS_Algorithms.ipynb
.. _Planning with RDKit: 11_Planning_with_RDKit.ipynb
.. _Rule Analysis: 12_Rule_Analysis.ipynb
.. _Priority Rules: 13_Priority_Rules.ipynb
.. _MHN Ranking Training: 14_MHN_Ranking_Training.ipynb
.. _Routes Compare: 15_Routes_compare.ipynb
.. _Building Block Search: 16_Building_block_search.ipynb
.. _Synthon-Based Library Design: 17_Synthon_Based_Design.ipynb
.. _Retrosynthesis with Synthon Priority Rules: 18_Retrosynthesis_With_Synthon_Priority_Rules.ipynb
.. _InChIKey Building-Block Catalogue: 20_InChIKey_Building_Block_Catalogue.ipynb

.. toctree::
   :hidden:
   :titlesonly:

   ten_minutes
   cli_interface
   data
   tables
   migration
   00_Welcome_to_Chython
   01_Coming_from_RDKit
   02_Data_Curation
   03_Rules_Extraction
   04_Policy_Training
   05_Retrosynthetic_Planning
   06_Tree_Analysis
   07_Clustering
   08_Protection_Scoring
   09_Combined_Ranking_Filtering_Policy
   10_NMCS_Algorithms
   11_Planning_with_RDKit
   12_Rule_Analysis
   13_Priority_Rules
   14_MHN_Ranking_Training
   15_Routes_compare
   16_Building_block_search
   17_Synthon_Based_Design
   18_Retrosynthesis_With_Synthon_Priority_Rules
   20_InChIKey_Building_Block_Catalogue
   ../configuration/configuration
   ../methods/methods
