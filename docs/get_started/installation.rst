Installation
-------------

This page explains the supported environments and shows three ways to install SynPlanner.

Supported environments
~~~~~~~~~~~~~~~~~~~~~~

- Python: ``>=3.10,<3.13`` (CPython). The publishing workflow builds with Python ``3.12``.
- OS/arch: developed and CI-tested on Linux x86_64 and MacOS arm64. Other platforms may work, but Docker is
  recommended for maximum portability.

What you get after install
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- CLI entrypoint: ``synplan``
- Python API: ``import synplan``
- Data and weights are not bundled. Fetch them with
  ``synplan download_all_data --save_to tutorials/synplan_data`` or follow :doc:`data_download`.

Install with pip (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a virtual environment.

.. code-block:: bash

   pip install SynPlanner

Verify:

.. code-block:: bash

   synplan --version
   synplan --help
   python -c "import synplan, sys; print('synplan', synplan.__version__)"


Install with Docker (portable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build and run the CLI inside a container. The provided Dockerfile targets Linux/AMD64.

.. code-block:: bash

   docker build --platform linux/amd64 -t synplan:latest-cli-amd64 -f cli.Dockerfile .
   docker run --rm --platform linux/amd64 -it synplan:latest-cli-amd64 --help

From source with Poetry (dev)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner.git
   cd SynPlanner/
   poetry env use $(which python)
   poetry install   # add "--with docs,dev" if you need docs or dev extras
   poetry shell
   synplan --help

Limitations and notes
~~~~~~~~~~~~~~~~~~~~~

- Wheels are published from Linux. If you experience platform issues on macOS/Windows,
  prefer Docker or a Linux environment (e.g., WSL2).
- Example data and model weights are not included; download them with the CLI
  (see :doc:`data_download`).
- To run the full planning quickstart in 10 minutes, continue to :doc:`/user_guide/ten_minutes`.


