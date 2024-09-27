.. _installation:

Installation
===========================

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


Poetry
--------------------

Poetry is useful for developers to reproduce exact environment. After installation of Poetry (described here)
you can follow these steps:

.. code-block:: bash

    # create a new environment and poetry
    conda create -n synplan_env -c conda-forge "poetry=1.3.2" "python=3.11" -y
    conda activate synplan_env

    # clone SynPlanner
    git clone https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner.git
    cd SynPlanner/

    # install SynPlanner with poetry
    poetry install

If Poetry fails with error, a possible solution is to update the bashrc file with the following command:

.. code-block:: bash

    echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >> ~/.bashrc
    exec "bash"

Manual installation
--------------------

If you want to install SynPlanner in an already existing environment with minimal breaking of dependencies,
or if you want to install different GPU drivers, we recommend to install all dependencies manually.
We also recommend using conda environments for proper installation of GPU drivers. In this example, the code
provided for the linux machine:

1. Create conda environment with Python less than 3.12:

.. code-block:: bash

    conda create -n synplan_env "python<3.12"
    conda activate synplan_env

2. Install PyTorch (for GPU drivers version and OS-specific installation please consult
`PyTorch documentation <https://pytorch.org/get-started/locally/>`_):

.. code-block:: bash

    conda install "pytorch<=2.3" pytorch-cuda=12.1 -c pytorch -c nvidia

3. Install Pytorch Geometric (for GPU drivers version and OS-specific installation please consult
`PyTorch Geometric documentation <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_):

.. code-block:: bash

    conda install pyg -c pyg

4. Install other dependencies available in conda:

.. code-block:: bash

    conda install "numpy<2" pytorch-lightning pandas ipykernel ipywidgets click "ray-default" -c conda-forge

5. Finalise installation by installing pip dependencies:

.. code-block:: bash

    git clone https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner.git
    cd SynPlanner/
    pip install .
