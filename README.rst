.. image:: docs/images/banner.png

SynPlanner – a tool for synthesis planning
===========================================

Docs_  |  Tutorials_  |  Preprint_  |  Paper_  |  `GUI demo`_

.. _Docs: https://synplanner.readthedocs.io/
.. _Tutorials: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/tree/main/tutorials
.. _Preprint: https://doi.org/10.26434/chemrxiv-2024-bzpnd
.. _Paper: https://doi.org/10.1021/acs.jcim.4c02004
.. _GUI demo: https://huggingface.co/spaces/Laboratoire-De-Chemoinformatique/SynPlanner

|License Badge| |PyPI Version Badge| |Python Versions Badge|

.. |License Badge| image:: https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/SynPlanner
   :target: https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/SynPlanner
   :alt: License Badge
.. |PyPI Version Badge| image:: https://img.shields.io/pypi/v/SynPlanner.svg
   :target: https://pypi.org/project/SynPlanner/
   :alt: PyPI Version
.. |Python Versions Badge| image:: https://img.shields.io/pypi/pyversions/SynPlanner.svg
   :target: https://pypi.org/project/SynPlanner/
   :alt: Supported Python Versions

``SynPlanner`` is an open-source tool for retrosynthetic planning,
designed to increase flexibility in training and developing
customized retrosynthetic planning solutions from raw chemical data.
It integrates Monte Carlo Tree Search (MCTS) with graph neural networks
to evaluate applicable reaction rules (policy network) and
the synthesizability of intermediate products (value network).


Overview
=============================

Unlock the power of ``SynPlanner`` for your chemical synthesis projects:

- ✅ **Ensure Data Quality:** Effortlessly standardize and filter raw chemical reaction data.
- 🧪 **Customize Reaction Templates:** Extract versatile reaction rules (templates) with a wide array of options.
- 🧠 **Advanced Model Training:** Train robust policy and value networks using both supervised and reinforcement learning techniques.
- 🗺️ **Flexible Retrosynthesis:** Perform in-depth retrosynthetic planning with diverse MCTS-based search strategies.
- 📊 **Intuitive Visualization:** Clearly visualize discovered synthetic paths and interact with an easy-to-use graphical user interface.

🚀 Quick Start
=============================

Get started with ``SynPlanner`` in a flash!

1.  **Download Essential Data:**
    Fetch the necessary pre-trained models and example data to begin your journey.

    .. code-block:: bash

        synplan download_all_data --save_to synplan_data

2.  **Explore Planning:**
    Once the data is downloaded, you can try running a planning example. For more detailed instructions, see the `Command-line interface`_ or `Tutorials`_ sections.

    .. code-block:: bash

        synplan planning --config configs/planning.yaml --targets synplan_data/benchmarks/sascore/targets_with_sascore_1.5_2.5.smi --reaction_rules synplan_data/uspto/uspto_reaction_rules.pickle --building_blocks synplan_data/building_blocks/building_blocks_em_sa_ln.smi --policy_network synplan_data/uspto/weights/ranking_policy_network.ckpt --results_dir planning_results_quickstart

    (Note: Ensure ``configs/planning.yaml`` exists or adjust the path accordingly. You might need to create a basic one or use one from the cloned repository if you haven't installed all package data globally.)

Installation
=============================

PyPI / pip
-----------------------------

``SynPlanner`` can also be installed directly using pip:

.. code-block:: bash

    pip install SynPlanner

You can find the package on PyPI: `SynPlanner on PyPI <https://pypi.org/project/SynPlanner/>`_.

After installation, ``SynPlanner`` can be added to Jupyter platform:

.. code-block:: bash

    conda install ipykernel
    python -m ipykernel install --user --name synplan --display-name "synplan"

Docker (CLI)
-----------------------------

You can run the SynPlanner command-line interface inside a Docker container. Follow these steps to build, name, and test the image.

1. Build the image

   Use the provided ``cli.Dockerfile`` to build a Linux AMD64 image. Name (tag) the image using the convention:

   ``<semver>-<interface>-<platform>``

   For example, to build version 1.1.0 with the CLI interface on AMD64:

.. code-block:: bash

       docker build \
         --platform linux/amd64 \
         -t synplan:1.1.0-cli-amd64 \
         -f cli.Dockerfile .

2. Verify the image

   List your local images to confirm the tag:

.. code-block:: bash

       docker images | grep synplan

You should see an entry similar to:

       synplan   1.1.0-cli-amd64   ...

3. Run and test the CLI

   Launch a container to execute the ``--help`` command and confirm the CLI is working:

.. code-block:: bash

       docker run --rm --platform linux/amd64 -it synplan:1.1.0-cli-amd64 --help

4. Example: planning with Docker

   You can also mount a local directory for data persistence. For example:

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
=============================

Colab
-----------------------------

    Colab tutorials do not require the local installation of ``SynPlanner`` but their performance is limited by available computational resources in Google Colab

Currently, two tutorials are available:

- `Retrosynthetic planning <https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/retrosynthetic_planning.ipynb>`_ can be used for retrosynthetic planning of any target molecule with pre-trained retrosynthetic models and advanced analysis of the search tree.
- `SynPlanner benchmarking <https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/planning_benchmarking.ipynb>`_ can be used for retrosynthetic planning of many target molecules for benchmarking or comparison analysis.

Jupyter
-----------------------------

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
=============================

``SynPlanner`` pipeline can be accessed by neat command-line interface (CLI). For example, retrosynthetic planning of several target molecules  with pre-trained models can performed with the following commands:

.. code-block:: bash

    synplan download_all_data --save_to synplan_data
    synplan planning --config configs/planning.yaml --targets synplan_data/benchmarks/sascore/targets_with_sascore_1.5_2.5.smi --reaction_rules synplan_data/uspto/uspto_reaction_rules.pickle --building_blocks synplan_data/building_blocks/building_blocks_em_sa_ln.smi --policy_network synplan_data/uspto/weights/ranking_policy_network.ckpt --results_dir planning_results

More details about CLI can be found in `SynPlanner Documentaion <https://synplanner.readthedocs.io/en/latest/interfaces/cli.html>`_

Contributing
=============================

Contributions are welcome, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.

Maintainers
=============================

* `Tagir Akhmetshin <https://github.com/tagirshin>`_
* `Dmitry Zankov <https://github.com/dzankov>`_

Contributors
=============================

* `Timur Madzhidov <tmadzhidov@gmail.com>`_
* `Alexandre Varnek <varnek@unistra.fr>`_
* `Philippe Gantzer <https://github.com/PGantzer>`_
* `Dmitry Babadeev <https://github.com/prog420>`_
* `Anna Pinigina <anna.10081048@gmail.com>`_
* `Mikhail Volkov <https://github.com/mbvolkoff>`_

📜 How to Cite
=============================

If you use ``SynPlanner`` in your research, please cite our work:

Akhmetshin, T.; Zankov, D.; Gantzer, P.; Babadeev, D.; Pinigina, A.; Madzhidov, T.; Varnek, A. SynPlanner: An End-to-End Tool for Synthesis Planning. *J. Chem. Inf. Model.* **2025**, *65* (1), 15–21. DOI: 10.1021/acs.jcim.4c02004

.. code-block:: bibtex

    @article{akhmetshin2025synplanner,
        title = {SynPlanner: An End-to-End Tool for Synthesis Planning},
        author = {Akhmetshin, Tagir and Zankov, Dmitry and Gantzer, Philippe and Babadeev, Dmitry and Pinigina, Anna and Madzhidov, Timur and Varnek, Alexandre},
        journal = {Journal of Chemical Information and Modeling},
        volume = {65},
        number = {1},
        pages = {15--21},
        year = {2025},
        doi = {10.1021/acs.jcim.4c02004},
        note = {PMID: 39739735},
        url = {https://doi.org/10.1021/acs.jcim.4c02004}
    }

