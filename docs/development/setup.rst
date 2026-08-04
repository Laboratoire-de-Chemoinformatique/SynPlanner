Local development setup
=======================

This page covers the local environment and building Docker images. For pull
request acceptance criteria see :doc:`pr_review`; for cutting a release see
:doc:`release`.

uv setup
--------

.. code-block:: bash

   # Install uv (see https://docs.astral.sh/uv/getting-started/installation/)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Clone and install with extras for docs/dev
   git clone https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner.git
   cd SynPlanner
   uv sync --group docs --group dev --extra cpu

   # Run tests
   uv run pytest -q

Build CLI Docker image
----------------------

.. code-block:: bash

   docker build --platform linux/amd64 -t synplan:dev-cli-amd64 -f cli.Dockerfile .
   docker run --rm --platform linux/amd64 synplan:dev-cli-amd64 --help

Build GUI Docker image
----------------------

.. code-block:: bash

   docker build --platform linux/amd64 -t synplan:dev-gui-amd64 -f gui.Dockerfile .
   docker run --rm --platform linux/amd64 -p 8501:8501 synplan:dev-gui-amd64

Build the documentation
-----------------------

.. code-block:: bash

   uv run sphinx-build -b html docs /tmp/synplanner-docs-html

Do not edit generated ``docs/_build`` output by hand — rebuild instead.
