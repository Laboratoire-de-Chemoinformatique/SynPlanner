.. image:: docs/images/banner.png

SynPlanner – a tool for synthesis planning
===========================================

.. centered:: Docs_ | Tutorials_ | Paper_ | GUI demo_

.. _Docs: https://synplanner.readthedocs.io/
.. _Tutorials: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/tree/main/tutorials
.. _Paper: https://doi.org/10.26434/chemrxiv-2024-bzpnd
.. _GUI demo: https://huggingface.co/spaces/Laboratoire-De-Chemoinformatique/SynPlanner

.. centered:: |License Badge|

.. |License Badge| image:: https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/SynPlanner
   :target: https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/SynPlanner
   :alt: License Badge

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

Next, create ``SynPlanner`` environment with ``synplan_linux.yaml`` file:

.. code-block:: bash

    conda env create -f conda/synplan_linux.yaml
    conda activate synplan
    pip install .


After installation, ``SynPlanner`` can be added to Jupyter platform:

.. code-block:: bash

    conda install ipykernel
    python -m ipykernel install --user --name synplan --display-name "synplan"

Docker (CLI)
=============================

You can run the SynPlanner command-line interface inside a Docker container. Follow these steps to build, name, and test the image.

1. Build the image

   Use the provided ``cli.Dockerfile`` to build a Linux AMD64 image. Name (tag) the image using the convention:

   ``<semver>-<interface>-<platform>``

   For example, to build version 1.1.0 with the CLI interface on AMD64::

.. code-block:: bash

       docker build \
         --platform linux/amd64 \
         -t synplan:1.1.0-cli-amd64 \
         -f cli.Dockerfile .

2. Verify the image

   List your local images to confirm the tag::

.. code-block:: bash

       docker images | grep synplan

You should see an entry similar to::

       synplan   1.1.0-cli-amd64   ...

3. Run and test the CLI

   Launch a container to execute the ``--help`` command and confirm the CLI is working::

.. code-block:: bash

       docker run --rm --platform linux/amd64 -it synplan:1.1.0-cli-amd64 --help

4. Example: planning with Docker

   You can also mount a local directory for data persistence. For example::

.. code-block:: bash

    docker run --rm \
      --platform linux/amd64 \
      -v "$(pwd)":/app \
      -w /app \
      synplan:1.1.0-cli-amd64 \
      planning \
        --config configs/planning.yaml \
        --targets tutorials/synplan_data/benchmarks/sascore/targets_with_sascore_1.5_2.5.smi \
        --reaction_rules tutorials/synplan_data/uspto/uspto_reaction_rules.pickle \
        --building_blocks tutorials/synplan_data/building_blocks/building_blocks_em_sa_ln.smi \
        --policy_network tutorials/synplan_data/uspto/weights/ranking_policy_network.ckpt \
        --results_dir tutorials/planning_results

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

- `SynPlanner pipeline <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/SynPlanner_Pipeline.ipynb>`_ presents the full pipeline of SynPlanner starting from raw reaction data and resulting in ready-to-use retrosynthetic planning.

**Advanced tutorials.** These tutorials provide advanced explanations and options for each step in the ``SynPlanner`` pipeline:

- `Step 1: Reaction data curation <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/Step-1_Data_Curation.ipynb>`_ can be used for reaction standardization and reaction filtration.
- `Step 2: Reaction rules extraction <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/Step-2_Rules_Extraction.ipynb>`_  can be used for extracting reaction rules from curated reaction data.
- `Step 3: Policy network training <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/Step-3_Policy_Training.ipynb>`_ can be used for policy network training.
- `Step 4: Retrosynthetic planning <https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/tutorials/Step-4_Retrosynthetic_Planning.ipynb>`_ can be used for retrosynthetic planning.

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

