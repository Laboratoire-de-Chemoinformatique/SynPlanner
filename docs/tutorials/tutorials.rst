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

- `General tutorial`_ - demonstrates the whole ``SynPlanner`` pipeline of reaction data curation, reaction rules extraction, retrosynthetic models training, retrosynthetic planning, and predicted retrosynthetic routes visualization. This tutorial can be used for routine creating ready-to-use planners for custom data and creating custom pipeline configurations.

**Advanced tutorials:**

- `Data curation`_ - demonstrates how to prepare data (reaction standardization and filtration) before reaction rules extraction and retrosynthetic model training in ``SynPlanner``
- `Rules extraction`_ - demonstrates how to extract reaction rules from reaction data in ``SynPlanner``.
- `Policy training`_ - demonstrates how to train ranking and filtering policy network in ``SynPlanner``.
- `Retrosynthetic planning`_ - demonstrates how retrosynthetic planning can be performed for target molecules in ``SynPlanner``.

.. _General tutorial: general_tutorial.ipynb
.. _Data curation: data_curation.ipynb
.. _Rules extraction: rules_extraction.ipynb
.. _Policy training: policy_training.ipynb
.. _Retrosynthetic planning: retrosynthetic_planning.ipynb

.. toctree::
    :hidden:
    :titlesonly:

    general_tutorial
    data_curation
    rules_extraction
    policy_training
    retrosynthetic_planning
