Development
================

This page covers local development setup, uv usage, and building Docker images.

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

Publish to GHCR (maintainers)
-----------------------------

Images are published automatically by CI on pushes to ``main`` and manual dispatches.
To push locally (requires write permissions to the repo's packages):

.. code-block:: bash

   VERSION=$(python -c 'import tomllib,sys;print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')
   docker login ghcr.io -u USERNAME -p TOKEN
   REPO=ghcr.io/laboratoire-de-chemoinformatique/synplanner
   docker tag synplan:dev-cli-amd64 ${REPO}:${VERSION}-cli-amd64
   docker tag synplan:dev-gui-amd64 ${REPO}:${VERSION}-gui-amd64
   docker push ${REPO}:${VERSION}-cli-amd64
   docker push ${REPO}:${VERSION}-gui-amd64


