.. _tutorials:

============
Tutorials
============

The tutorials provide basic information on how to use SynPlanner.

We recommend that you start with `retrosynthetic planning`_,
which provides a minimal example of how to use SynPlanner for retrosynthetic planning.


The main pipeline of SynPlanner training from the raw reaction data includes:

- `Data curation`_, which presents the workflow for reaction standardization and reaction filtration.
- `Rules extraction`_, which provides a workflow for extracting rules from curated reaction data.
- `Ranking policy training`_, which shows the workflow for extracting rules from curated reaction data.


.. _Data curation: data_curation.ipynb
.. _Rules extraction: rules_extraction.ipynb
.. _Ranking policy training: ranking_policy_training.ipynb
.. _Retrosynthetic planning: retrosynthetic_planning.ipynb

.. toctree::
    :hidden:
    :titlesonly:

    data_curation
    rules_extraction
    policy_training
    retrosynthetic_planning
