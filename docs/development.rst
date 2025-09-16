Development
================

This page covers local development setup, Poetry usage, and building Docker images.

Poetry setup
------------

.. code-block:: bash

   # Install Poetry (see https://python-poetry.org/docs/)
   curl -sSL https://install.python-poetry.org | python3 -
   export PATH="$HOME/.local/bin:$PATH"

   # Clone and install with extras for docs/dev
   git clone https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner.git
   cd SynPlanner
   poetry env use $(which python)
   poetry install --with docs,dev
   poetry shell

   # Run tests
   pytest -q

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

   VERSION=$(python -c 'import tomllib,sys;print(tomllib.load(open("pyproject.toml","rb"))["tool"]["poetry"]["version"])')
   docker login ghcr.io -u USERNAME -p TOKEN
   REPO=ghcr.io/laboratoire-de-chemoinformatique/synplanner
   docker tag synplan:dev-cli-amd64 ${REPO}:${VERSION}-cli-amd64
   docker tag synplan:dev-gui-amd64 ${REPO}:${VERSION}-gui-amd64
   docker push ${REPO}:${VERSION}-cli-amd64
   docker push ${REPO}:${VERSION}-gui-amd64


