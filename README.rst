.. image:: docs/images/banner.png

.. raw:: html

    <div align="center">
        <h1>SynPlanner – a tool for synthesis planning</h1>
    </div>

    <h3>
        <p align="center">
            <a href="https://synplanner.readthedocs.io/">Docs</a> •
            <a href="https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/tree/main/tutorials">Tutorials</a> •
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
-----------------------------

``SynPlanner`` can be used for:

- ⚒️ Standardizing and filtering reaction data
- 📑 Extracting reaction rules (templates) with various options
- 🧠 Training policy and value networks using supervised and reinforcement learning
- 🔍 Performing retrosynthetic planning with different MCTS-based search strategies
- 🖼️ Visualising found synthetic paths and working with graphical user interface


Installation
-----------------------------

Conda (Linux)
=============================

``SynPlanner`` can be installed using conda/mamba package managers.
For more information on conda installation please refer to the
`official documentation <https://github.com/conda-forge/miniforge>`_.

To install ``SynPlanner``, first clone the repository and move the package directory:

.. code-block:: bash

    git clone https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner.git
    cd SynPlanner/

Next, create ``SynPlanner`` environment with ``synplan_env_linux.yaml`` file:

.. code-block:: bash

    conda env create -f conda/synplan_env_linux.yaml
    conda activate synplan_env
    pip install .


After installation, ``SynPlanner`` can be added to Jupyter platform:

.. code-block:: bash

    conda install ipykernel
    python -m ipykernel install --user --name synplan_env --display-name "synplan"

Tutorials
-----------------------------

Colab
=============================

    Colab tutorials do not require the local installation of ``SynPlanner`` but their performance is limited by available computational resources in Google Colab

Currently, two tutorials are available:

- `Retrosynthetic planning <https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/retrosynthetic_planning.ipynb>`_ can be used for retrosynthetic planning of any target molecule with pre-trained retrosynthetic models and advanced analysis of the search tree.
- `SynPlanner benchmarking <https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/planning_benchmarking.ipynb>`_ can be used for retrosynthetic planning of many target molecules for benchmarking or comparison analysis.

Jupyter
=============================

    Jupyter Tutorials requires the local installation of ``SynPlanner`` but can be executed with advanced computational resources on local servers

Currently, five tutorials are available:

**Quick-start tutorials.** These tutorials can be used for easy execution of the default ``SynPlanner`` pipeline:

- `General tutorial <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/general_tutorial.ipynb>`_ presents the full pipeline of SynPlanner starting from raw reaction data and resulting in ready-to-use retrosynthetic planning.

**Advanced tutorials.** These tutorials provide advanced explanations and options for each step in the ``SynPlanner`` pipeline:

- `Reaction data curation <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/data_curation.ipynb>`_ presents the workflow for reaction standardization and reaction filtration.
- `Reaction rules extraction <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/rules_extraction.ipynb>`_  provides a workflow for extracting reaction rules from curated reaction data.
- `Policy network training <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/policy_training.ipynb>`_ shows the workflow for policy network training.
- `Retrosynthetic planning <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/retrosynthetic_planning.ipynb>`_ provides an example of how to use ``SynPlanner`` for retrosynthetic planning.

Command-line interface
-----------------------------

``SynPlanner`` pipeline can be accessed by neat command-line interface (CLI). For example, retrosynthetic planning of several target molecules  with pre-trained models can performed with the following commands:

.. code-block:: bash

    synplan download_all_data --save_to synplan_data
    synplan planning --config configs/planning.yaml --targets synplan_data/benchmarks/sascore/targets_with_sascore_1.5_2.5.smi --reaction_rules synplan_data/uspto/uspto_reaction_rules.pickle --building_blocks synplan_data/building_blocks/building_blocks_em_sa_ln.smi --policy_network synplan_data/uspto/weights/ranking_policy_network.ckpt --results_dir planning_results

More details about CLI can be found in `SynPlanner Documentaion <https://synplanner.readthedocs.io/en/latest/interfaces/cli.html>`_

Contributing
-----------------------------

Contributions are welcome, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.

Maintainers
-----------------------------

* `Tagir Akhmetshin <https://github.com/tagirshin>`_
* `Dmitry Zankov <https://github.com/dzankov>`_

Contributors
-----------------------------

* `Timur Madzhidov <tmadzhidov@gmail.com>`_
* `Alexandre Varnek <varnek@unistra.fr>`_
* `Philippe Gantzer <https://github.com/PGantzer>`_
* `Dmitry Babadeev <https://github.com/prog420>`_
* `Anna Pinigina <anna.10081048@gmail.com>`_
* `Mikhail Volkov <https://github.com/mbvolkoff>`_

