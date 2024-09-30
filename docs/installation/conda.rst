.. _conda:

================================
Conda (Linux)
================================

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