.. _configuration:

================
Configuration
================

``SynPlanner`` pipeline is designed to be flexible for serving different needs related to reaction data curation,
reaction rules extraction, and retrosynthetic planning. ``SynPlanner`` can be configured by configuration
`yaml files <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/tree/main/configs>`_. or configuration Python classes (mainly used in Python interface in tutorials).

What this section is for
------------------------

This section shows how to configure SynPlanner: which YAML fields exist, how they affect the pipelines, and how to
use the corresponding Python configuration classes. Use this when you want to change behavior without modifying code.

For the conceptual background of each stage, see :doc:`/methods/methods`. For how to run via CLI, see :doc:`/user_guide/cli_interface`. For Python, see :doc:`/api`.

.. toctree::
    :hidden:
    :titlesonly:

    standardization
    filtration
    extraction
    policy
    value
    planning