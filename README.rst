.. image:: docs/images/banner.png

.. raw:: html

    <div align="center">
        <h1>SynPlanner – a tool for synthesis planning</h1>
    </div>

    <h3>
        <p align="center">
            <a href="https://synplanner.readthedocs.io/">Docs</a> •
            <a href="https://synplanner.readthedocs.io/en/latest/tutorials.html">Tutorials</a> •
            <a href="https://doi.org/10.26434/chemrxiv-2024-bzpnd">Paper</a> •
            <a href="https://huggingface.co/spaces/Laboratoire-De-Chemoinformatique/SynPlanner">GUI demo</a>
        </p>
    </h3>

    <div align="center">
        <a href="https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/SynPlanner">
            <img src="https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/SynPlanner" alt="License Badge">
        </a>
    </div>

``SynPlanner`` is an open-source tool for retrosynthetic planning,
designed to increase flexibility in training and developing
customized retrosynthetic planning solutions from raw chemical data.
It integrates Monte Carlo Tree Search (MCTS) with graph neural networks
to evaluate applicable reaction rules (policy network) and
the synthesizability of intermediate products (value network).


Overview
--------------------

``SynPlanner`` can be used for:

- ⚒️ Standardizing and filtering reaction data
- 📑 Extracting reaction rules (templates) with various options
- 🧠 Training policy and value networks using supervised and reinforcement learning
- 🔍 Performing retrosynthetic planning with different MCTS-based search strategies
- 🖼️ Visualising found synthetic paths and working with graphical user interface


Installation
--------------------

Conda (Linux)
--------------------

``SynPlanner`` can also be installed using conda/mamba package managers.
For more information on conda installation please refer to the
`official documentation <https://github.com/conda-forge/miniforge>`_.

To install ``SynPlanner``, first clone the repository and move the package directory:

.. code-block:: bash

    git clone https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner.git
    cd SynPlanner/

Next, create ``SynPlanner`` environment with `synplan_env_linux.yaml` file:

.. code-block:: bash

    conda env create -f conda/synplan_env_linux.yaml
    conda activate synplan_env
    pip install .


After installation, one can add the ``SynPlanner`` environment in their Jupyter platform:

.. code-block:: bash

    python -m ipykernel install --user --name synplan_env --display-name "synplan"

Colab Tutorials
--------------------

Colab tutorials don’t require the local installation of ``SynPlanner`` but are limited by available computational resources in Google Colab.

Currently, two tutorials are available:

- General tutorial - presents the full pipeline of SynPlanner starting from raw reaction data and resulting in ready-to-use retrosynthetic planning. *This tutorial can be used for training retrosynthetic models on custom data from scratch.*

- Planning tutorial – presents the ready-to-use retrosynthetic planning in SynPlanner. *This tutorial can be used for retrosynthetic planning for custom target molecules with pretrained retrosynthetic models that can downloaded from SynPlanner.*

Jupyter Tutorials
--------------------

Jupyter Tutorials requires the local installation of ``SynPlanner`` but can be executed with advanced computational resources on local servers.

Currently, four tutorials are available:

- `Data curation <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/docs/tutorial/data_curation.ipynb>`_ presents the workflow for reaction standardization and reaction filtration.
- `Rules extraction <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/docs/tutorial/rules_extraction.ipynb>`_  provides a workflow for extracting rules from curated reaction data.
- `Ranking policy training <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/docs/tutorial/retrosynthetic_planning.ipynb>`_ shows the workflow for extracting rules from curated reaction data.
- `Retrosynthetic planning <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/docs/tutorial/data_curation.ipynb>`_ provides a minimal example of how to use SynPlanner for retrosynthetic planning.


Contributing
--------------------

Contributions are welcome, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.

To contribute with code to the project, follow these steps:

1. Fork this repository.
2. Create a branch: ``git checkout -b <branch_name>``.
3. Make your changes and commit them: ``git commit -m '<commit_message>'``
4. Push to the remote branch: ``git push``
5. Create the pull request.


Maintainers
--------------------

* `Tagir Akhmetshin <https://github.com/tagirshin>`_
* `Dmitry Zankov <https://github.com/dzankov>`_


Contributors
--------------------

* `Timur Madzhidov <tmadzhidov@gmail.com>`_
* `Alexandre Varnek <varnek@unistra.fr>`_
* `Philippe Gantzer <https://github.com/PGantzer>`_
* `Dmitry Babadeev <https://github.com/prog420>`_
* `Anna Pinigina <anna.10081048@gmail.com>`_
* `Mikhail Volkov <https://github.com/mbvolkoff>`_

