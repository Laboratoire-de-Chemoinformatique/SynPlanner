.. _tutorials:

============
Tutorials
============

``SynPlanner`` is supplemented by several tutorials explaining different aspects of tool usage. These tutorials included
some advanced explanations and configurations, that can be used not only for demonstrative and educational purposes but
also for creating flexible pipelines.

Currently, there are two groups of tutorials available – general and advanced. General tutorials aimed to provide a quick review
of the ``SynPlanner`` pipeline and its application to the custom data. The advanced tutorial provides more explanations about
each ``SynPlanner`` pipeline step and allows for more sophisticated functionalities and configurations for each pipeline step.

**General tutorials:**

- `SynPlanner Pipeline`_ - demonstrates the whole ``SynPlanner`` pipeline of reaction data curation, reaction rules extraction, retrosynthetic models training, retrosynthetic planning, and predicted retrosynthetic routes visualization. This tutorial can be used for routine creating ready-to-use planners for custom data and creating custom pipeline configurations.

**Advanced tutorials:**

- `Step-1. Data curation`_ - demonstrates how to prepare data (reaction standardization and filtration) before reaction rules extraction and retrosynthetic model training in ``SynPlanner``
- `Step-2. Reaction rules extraction`_ - demonstrates how to extract reaction rules from reaction data in ``SynPlanner``.
- `Step-3. Policy network training`_ - demonstrates how to train ranking and filtering policy network in ``SynPlanner``.
- `Step-4. Retrosynthetic planning`_ - demonstrates how retrosynthetic planning can be performed for target molecules in ``SynPlanner``.

.. _SynPlanner Pipeline: SynPlanner_Pipeline.ipynb
.. _Step-1. Data curation: Step-1_Data_Curation.ipynb
.. _Step-2. Reaction rules extraction: Step-2_Rules_Extraction.ipynb
.. _Step-3. Policy network training: Step-3_Policy_Training.ipynb
.. _Step-4. Retrosynthetic planning: Step-4_Retrosynthetic_Planning.ipynb

.. toctree::
    :hidden:
    :titlesonly:

    SynPlanner_Pipeline
    Step-1_Data_Curation
    Step-2_Rules_Extraction
    Step-3_Policy_Training
    Step-4_Retrosynthetic_Planning
