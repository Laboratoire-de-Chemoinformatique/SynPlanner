.. _manual:

===========================
Manual installation
===========================

If you want to install ``SynPlanner`` in an already existing environment with minimal breaking of dependencies,
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