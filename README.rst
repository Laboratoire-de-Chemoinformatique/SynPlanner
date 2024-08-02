.. image:: docs/images/banner.png

.. raw:: html

    <div align="center">
        <h1>synplanner – a tool for synthesis planning</h1>
    </div>

    <h3>
        <p align="center">
            <a href="https://synplanner.readthedocs.io/">Docs</a> •
            <a href="https://github.com/Laboratoire-de-Chemoinformatique/synplanner/tree/main/tutorials">Tutorials</a> •
            <a href="https://github.com/Laboratoire-de-Chemoinformatique/synplanner/tutorials">Paper</a> •
            <a href="https://huggingface.co/spaces/Laboratoire-De-Chemoinformatique/synplanner">GUI demo</a>
        </p>
    </h3>

    <div align="center">
        <a href="https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/synplanner">
            <img src="https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/synplanner" alt="License Badge">
        </a>
    </div>


``synplanner`` is an open-source tool for retrosynthesis planning,
designed to increase flexibility in training and developing
customized retrosynthetic planning solutions from raw chemical data.
It integrates Monte Carlo Tree Search (MCTS) with graph neural networks
to evaluate applicable reaction rules (policy network) and
the synthesizability of intermediate products (value network).


Overview
--------------------

``synplanner`` can be used for:

- ⚒️ Standardizing and filtering reaction data
- 📑 Extracting reaction rules (templates) with various options
- 🧠 Training policy and value networks using supervised and reinforcement learning
- 🔍 Performing retrosynthetic planning with different MCTS-based search strategies
- 🖼️ Visualising found synthetic paths and working with graphical user interface


Installation
--------------------

``synplanner`` can be installed by the following steps:

.. code-block:: bash

    # install miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

    # create a new environment and poetry
    conda create -n synplanner -c conda-forge poetry "python=3.11" -y
    conda activate synplanner

    # clone synplanner
    git clone https://github.com/Laboratoire-de-Chemoinformatique/synplanner.git

    # navigate to the synplanner folder and install all the dependencies
    cd synplanner/
    poetry install

After installation, ``synplanner`` environment can be used with Jupyter platform:

.. code-block:: bash

    conda install jupyter ipykernel
    python -m ipykernel install --user --name synplanner --display-name "synplanner"

Quick start
--------------------

Each command in ``synplanner`` has a description that can be called with ``synplanner --help`` and ``synplanner command --help``

To run a retrosynthesis planning in ``synplanner`` the reaction rules, trained retrosynthetic models (policy network and value network),
and building block molecules are needed.

The planning command takes the file with the SMILES of target molecules listed one by one.
Also, the target molecule can be provided in the SDF format.

If you use your custom building blocks, be sure to canonicalize them before planning.

.. code-block:: bash

    # download planning data
    synplanner download_planning_data

    # canonicalize building blocks
    synplanner building_blocks_canonicalizing --input building_blocks_custom.smi --output synplanner_planning_data/building_blocks.smi

    # planning with rollout evaluation
    synplanner planning --config configs/planning.yaml --targets benchmark/targets_with_sascore_1.5_2.5.smi --reaction_rules synplanner_planning_data/uspto_reaction_rules.pickle --building_blocks synplanner_planning_data/building_blocks.smi --policy_network synplanner_planning_data/ranking_policy_network.ckpt --results_dir planning_results

    # planning with value network evaluation
    synplanner planning --config configs/planning.yaml --targets benchmark/targets_with_sascore_1.5_2.5.smi --reaction_rules synplanner_planning_data/uspto_reaction_rules.pickle --building_blocks synplanner_planning_data/building_blocks.smi --policy_network synplanner_planning_data/ranking_policy_network.ckpt --value_network synplanner_planning_data/value_network.ckpt --results_dir planning_results

After retrosynthesis planning is finished, the visualized retrosynthesis routes can be fund in the results folder (``planning_results/extracted_routes_html``).

``synplanner`` includes the full pipeline of reaction data curation, reaction rules extraction, and retrosynthetic models training.
For more details consult the corresponding sections in the documentation `here <https://synplanner.readthedocs.io/>`_.

Tutorials
--------------------

``synplanner`` can be accessed via the Python interface. For a better understanding of ``synplanner`` and its functionalities consult
the tutorials in `synplanner/tutorials`. Currently, two tutorials are available:

``tutorials/general_tutorial.ipynb`` – explains how to do a reaction rules extraction, policy network training, and retrosynthesis planning in synplanner.

``tutorials/planning_tutorial.ipynb`` – explains how to do a retrosynthesis planning with various configurations of planning algorithms (various expansion/evaluation functions and search strategies).

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

