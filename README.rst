.. image:: https://raw.githubusercontent.com/Laboratoire-de-Chemoinformatique/SynPlanner/main/docs/images/banner.png
   :alt: SynPlanner banner
   :align: center

SynPlanner – a tool for synthesis planning
===========================================

Docs_  |  Tutorials_  |  Preprint_  |  Paper_  |  `GUI demo`_

.. _Docs: https://synplanner.readthedocs.io/
.. _Tutorials: https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/tree/main/tutorials
.. _Preprint: https://doi.org/10.26434/chemrxiv-2024-bzpnd
.. _Paper: https://doi.org/10.1021/acs.jcim.4c02004
.. _GUI demo: https://huggingface.co/spaces/Laboratoire-De-Chemoinformatique/SynPlanner


|License Badge| |PyPI Version Badge| |Python Versions Badge| |Open In Colab|

.. |License Badge| image:: https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/SynPlanner
   :target: https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/SynPlanner
   :alt: License Badge
.. |PyPI Version Badge| image:: https://img.shields.io/pypi/v/SynPlanner.svg
   :target: https://pypi.org/project/SynPlanner/
   :alt: PyPI Version
.. |Python Versions Badge| image:: https://img.shields.io/pypi/pyversions/SynPlanner.svg
   :target: https://pypi.org/project/SynPlanner/
   :alt: Supported Python Versions

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/routes_clustering.ipynb
   :alt: Open In Colab

``SynPlanner`` is an open-source tool for retrosynthetic planning,
designed to increase flexibility in training and developing
customized retrosynthetic planning solutions from raw chemical data.
It integrates Monte Carlo Tree Search (MCTS) with graph neural networks
to evaluate applicable reaction rules (policy network) and
the synthesizability of intermediate products (value network).

✨ Overview
=============================

``SynPlanner`` offers comprehensive capabilities for chemical synthesis planning:

- ✅ **Ensure Data Quality:** Effortlessly standardize and filter raw chemical reaction data.
- 🧪 **Customize Reaction Templates:** Extract versatile reaction rules (templates) with a wide array of options.
- 🧠 **Advanced Model Training:** Train robust policy and value networks using both supervised and reinforcement learning techniques.
- 🗺️ **Flexible Retrosynthesis:** Perform in-depth retrosynthetic planning with diverse MCTS-based search strategies.
- 📊 **Intuitive Visualization:** Clearly visualize discovered synthetic paths and interact with an easy-to-use graphical user interface.

📦 Installation
=============================

PyPI / pip
-----------------------------

``SynPlanner`` can also be installed directly using pip:

.. code-block:: bash

    pip install SynPlanner

You can find the package on PyPI: `SynPlanner on PyPI <https://pypi.org/project/SynPlanner/>`_.

After installation, ``SynPlanner`` can be added to Jupyter platform:

.. code-block:: bash

    python -m ipykernel install --user --name synplan --display-name "synplan"

🐳 Docker (prebuilt images)
-----------------------------

Prebuilt images are published to GHCR for Linux/AMD64.

.. code-block:: bash

    VERSION=1.2.1

    # CLI: pull and show help
    docker pull ghcr.io/laboratoire-de-chemoinformatique/synplanner:${VERSION}-cli-amd64
    docker run --rm --platform linux/amd64 \
      ghcr.io/laboratoire-de-chemoinformatique/synplanner:${VERSION}-cli-amd64 --help

    # GUI: pull and run on http://localhost:8501
    docker pull ghcr.io/laboratoire-de-chemoinformatique/synplanner:${VERSION}-gui-amd64
    docker run --rm --platform linux/amd64 -p 8501:8501 \
      ghcr.io/laboratoire-de-chemoinformatique/synplanner:${VERSION}-gui-amd64

For a mounted-data planning example and more details, see the Get started docs: https://synplanner.readthedocs.io/en/latest/get_started/index.html

🚀 Quick Start
=============================

Get started with ``SynPlanner`` in a few steps:

1.  **Download Essential Data:**
    Fetch the necessary pre-trained models and example data to begin your journey.

    .. code-block:: bash

        synplan download_all_data --save_to synplan_data

2.  **Explore Planning:**
    Once the data is downloaded, you can try running a planning example. For more detailed instructions, see the `Tutorials`_ sections.

    .. code-block:: bash

        synplan planning --config configs/planning.yaml --targets synplan_data/benchmarks/sascore/targets_with_sascore_1.5_2.5.smi --reaction_rules synplan_data/uspto/uspto_reaction_rules.pickle --building_blocks synplan_data/building_blocks/building_blocks_em_sa_ln.smi --policy_network synplan_data/uspto/weights/ranking_policy_network.ckpt --results_dir planning_results_quickstart

    (Note: Ensure ``configs/planning.yaml`` exists or adjust the path accordingly. You might need to create a basic one or use one from the cloned repository if you haven't installed all package data globally.)

📓 Colab tutorials
-----------------------------

Currently, three tutorials are available that can run with Google Colab:

- `Retrosynthetic planning <https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/retrosynthetic_planning.ipynb>`_: plan routes for one target and inspect the search tree.
- `SynPlanner benchmarking <https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/planning_benchmarking.ipynb>`_: run many targets and compare results.
- `Route clustering by strategic bonds <https://colab.research.google.com/github/Laboratoire-de-Chemoinformatique/SynPlanner/blob/main/colab/routes_clustering.ipynb>`_: cluster planned routes by strategic bonds and view concise HTML reports.

🤝 Contributing
=============================

Contributions are welcome, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.

👥 Maintainers
=============================

* `Tagir Akhmetshin <https://github.com/tagirshin>`_
* `Dmitry Zankov <https://github.com/dzankov>`_

👥 Contributors
=============================

* `Timur Madzhidov <tmadzhidov@gmail.com>`_
* `Alexandre Varnek <varnek@unistra.fr>`_
* `Almaz Gilmullin <https://github.com/Protolaw>`_
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

