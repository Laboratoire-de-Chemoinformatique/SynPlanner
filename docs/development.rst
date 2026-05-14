Development
================

This page covers local development setup, uv usage, and building Docker images.
For pull request acceptance criteria, see :doc:`pr_review`.

.. toctree::
   :hidden:
   :maxdepth: 1

   pr_review

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

Bump version
------------

Version is managed from ``pyproject.toml`` with ``uv version``.

.. code-block:: bash

   # patch: 1.4.3 -> 1.4.4
   uv version --bump patch --no-sync

   # minor: 1.4.4 -> 1.5.0
   uv version --bump minor --no-sync

   # major: 1.5.0 -> 2.0.0
   uv version --bump major --no-sync

This updates ``pyproject.toml`` and relocks the project without syncing the
local environment.

**Manual steps after bumping:**

1. Update ``docs/_static/switcher.json``: add the old version to the list and
   rename ``(stable)`` to the new version.
2. Update ``CHANGELOG.md``: move items from ``[Unreleased]`` into a new
   section and add footer links.
3. Update ``docs/get_started/docker_images.rst`` so the documented GHCR
   ``VERSION`` matches the new release.

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
