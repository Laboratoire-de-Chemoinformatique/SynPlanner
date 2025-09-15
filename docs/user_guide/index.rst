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

Currently, there are two groups of tutorials available – general and advanced. General tutorials aimed to provide a quick review
of the ``SynPlanner`` pipeline and its application to the custom data. The advanced tutorial provides more explanations about
each ``SynPlanner`` pipeline step and allows for more sophisticated functionalities and configurations for each pipeline step.

**General tutorials:**

- `10 minutes to SynPlanner`_ - a quickstart guide to run planning via Python or CLI.
- `SynPlanner Pipeline`_ - demonstrates the whole ``SynPlanner`` pipeline of reaction data curation, reaction rules extraction, retrosynthetic models training, retrosynthetic planning, and predicted retrosynthetic routes visualization. This tutorial can be used for routine creating ready-to-use planners for custom data and creating custom pipeline configurations.

**Advanced tutorials:**

- `Step-1. Data curation`_ - demonstrates how to prepare data (reaction standardization and filtration) before reaction rules extraction and retrosynthetic model training in ``SynPlanner``
- `Step-2. Reaction rules extraction`_ - demonstrates how to extract reaction rules from reaction data in ``SynPlanner``.
- `Step-3. Policy network training`_ - demonstrates how to train ranking and filtering policy network in ``SynPlanner``.
- `Step-4. Retrosynthetic planning`_ - demonstrates how retrosynthetic planning can be performed for target molecules in ``SynPlanner``.
- `Step-5. Clustering`_ - demonstrates how to cluster predicted retrosynthetic routes in ``SynPlanner``.

.. _SynPlanner Pipeline: SynPlanner_Pipeline.ipynb
.. _10 minutes to SynPlanner: ten_minutes.rst
.. _Step-1. Data curation: Step-1_Data_Curation.ipynb
.. _Step-2. Reaction rules extraction: Step-2_Rules_Extraction.ipynb
.. _Step-3. Policy network training: Step-3_Policy_Training.ipynb
.. _Step-4. Retrosynthetic planning: Step-4_Retrosynthetic_Planning.ipynb
.. _Step-5. Clustering: Step-5_Clustering.ipynb

.. toctree::
    :hidden:
    :titlesonly:

    ten_minutes
    cli_interface
    data
    SynPlanner_Pipeline
    Step-1_Data_Curation
    Step-2_Rules_Extraction
    Step-3_Policy_Training
    Step-4_Retrosynthetic_Planning
    Step-5_Clustering
    ../configuration/configuration
    ../methods/methods