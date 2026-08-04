Installation
-------------

This page explains the supported environments and shows three ways to install SynPlanner.

Supported environments
~~~~~~~~~~~~~~~~~~~~~~

- Python: ``>=3.10,<3.15`` (CPython). The publishing workflow builds with Python ``3.12``.
- OS/arch: CI runs the test suite on Linux, macOS, and Windows across every supported Python version.
  Other platforms may work, but Docker is recommended for maximum portability.

What you get after install
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- CLI entrypoint: ``synplan``
- Python API: ``import synplan``
- Data and weights are not bundled. Fetch them with
  ``synplan download_preset --preset synplanner-gps --save_to synplan_data`` or follow :doc:`data_download`.
- The ``configs/*.yaml`` files used by every CLI example are **not** installed by
  ``pip`` either — they live only in the git repository. Clone it, or fetch the
  one you need:

  .. code-block:: bash

     curl -O https://raw.githubusercontent.com/Laboratoire-de-Chemoinformatique/SynPlanner/main/configs/planning_standard.yaml

  ``synplan planning --config`` is required and has no built-in default.

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

From source with uv (dev)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner.git
   cd SynPlanner/
   uv sync --extra cpu   # add "--group docs --group dev" if you need docs or dev extras
   uv run synplan --help

Selecting a PyTorch build
~~~~~~~~~~~~~~~~~~~~~~~~~

Most users do not need to think about this: the default PyTorch wheel for your
platform is installed automatically and works.

If you need a specific build (CPU-only on a GPU machine, or a particular CUDA
version), how you select it depends on the installer.

**With uv**, three mutually exclusive extras are available:

.. code-block:: bash

   uv sync --extra cpu      # CPU only
   uv sync --extra cu126    # CUDA 12.6
   uv sync --extra cu128    # CUDA 12.8

**With pip**, these extras change nothing. ``pip install "SynPlanner[cpu]"``
succeeds — pip accepts the extra — but all it adds is ``torch>=2.0``, which is
already satisfied by the transitive requirement from ``chytorch-synplan``,
``torch-geometric`` and ``pytorch-lightning``. The index mapping that gives the
extras their meaning lives in ``[tool.uv.sources]`` and ``[[tool.uv.index]]``,
which pip does not read, so torch is resolved from PyPI either way. The result is
identical to a plain ``pip install SynPlanner``. To control the build with pip, install ``torch`` from the
`PyTorch install selector <https://pytorch.org/get-started/locally/>`_ first, then
install SynPlanner.

To check what you ended up with:

.. code-block:: bash

   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

If ``torch.cuda.is_available()`` is ``False`` on a machine with a GPU, the build
does not match the hardware — reinstall torch from the matching index rather than
patching the existing install.

Limitations and notes
~~~~~~~~~~~~~~~~~~~~~

- If you experience platform issues, prefer Docker or a Linux environment (e.g., WSL2).
- Example data and model weights are not included; download them with the CLI
  (see :doc:`data_download`).
- To run the full planning quickstart in 10 minutes, continue to :doc:`/user_guide/ten_minutes`.


