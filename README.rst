.. image:: docs/images/banner.png

.. raw:: html

    <div align="center">
        <h1>SynPlanner – a tool for synthesis planning</h1>
    </div>

    <h3>
        <p align="center">
            <a href="https://synplanner.readthedocs.io/">Docs</a> •
            <a href="https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/tree/main/tutorials">Tutorials</a> •
            <a href="https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner/tutorials">Paper</a> •
            <a href="https://huggingface.co/spaces/Laboratoire-De-Chemoinformatique/SynPlanner">GUI demo</a>
        </p>
    </h3>

    <div align="center">
        <a href="https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/SynPlanner">
            <img src="https://img.shields.io/github/license/Laboratoire-de-Chemoinformatique/SynPlanner" alt="License Badge">
        </a>
    </div>

``SynPlanner`` is an open-source tool for retrosynthesis planning,
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
Pip
--------------------

The easiest way to install SynPlanner is through PYPI:

.. code-block:: bash

    pip install synplan

.. tip::

    In case your organisation have additional protection rules you can try to install it through adding additional
    flags:

    .. code-block:: bash

        pip install [--trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org]
        --use-pep517 synplan


Conda
--------------------

SynPlanner can also be installed using conda/mamba package managers.
For more information on conda installation please refer to the
`official documentation <https://github.com/conda-forge/miniforge>`_.

To install SynPlanner, first clone the repository and move the package directory:

.. code-block:: bash

    git clone https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner.git
    cd SynPlanner/

Next, create SynPlanner environment with `.yaml` file, where `$OS` can be `linux`, `macos`, `win`:

.. code-block:: bash

    conda env create -f conda/synplan_env_$OS.yaml
    conda activate synplan_env
    pip install .

.. tip::

    After installation, one can add the SynPlanner environment in their Jupyter platform:

    .. code-block:: bash

        python -m ipykernel install --user --name synplan_env --display-name "synplan"

Tutorials
--------------------

``SynPlanner`` can be accessed via the Python interface. For a better understanding of ``SynPlanner`` and its functionalities consult
the tutorials in `SynPlanner/tutorials`. Currently, two tutorials are available:

``tutorials/general_tutorial.ipynb`` – explains how to do a reaction rules extraction,
policy network training, and retrosynthesis planning in SynPlanner.

``tutorials/planning_tutorial.ipynb`` – explains how to do a retrosynthesis
planning with various configurations of planning algorithms
(various expansion/evaluation functions and search strategies).

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

