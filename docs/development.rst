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

The version lives in one place — ``pyproject.toml`` (``[project] version``).
``synplan.__version__`` and the Sphinx docs derive from it automatically.

Recommended: one command does the bump *and* the dependent edits.

.. code-block:: bash

   uv run release patch              # or: minor, major
   uv run release patch --dry-run    # preview, writes nothing

This runs ``uv version --bump`` (updating ``pyproject.toml`` and ``uv.lock``),
stamps the ``CHANGELOG.md`` ``[Unreleased]`` section with the new version and
date plus footer links, and rotates ``docs/_static/switcher.json``. The GHCR
``VERSION`` in ``docs/get_started/docker_images.rst`` renders from Sphinx
``|release|`` and needs no edit. Review the diff, then commit and tag
``vX.Y.Z``.

Manual path
~~~~~~~~~~~

To bump only ``pyproject.toml`` + ``uv.lock`` without the dependent edits:

.. code-block:: bash

   uv version                          # show current, e.g. synplanner 1.5.1
   uv version --short                  # 1.5.1
   uv version --bump patch --no-sync   # 1.5.1 -> 1.5.2

``--no-sync`` writes ``pyproject.toml`` and relocks ``uv.lock`` without
reinstalling the local environment. Use ``--dry-run`` to preview, or
``--frozen`` to skip the lockfile update entirely.

.. note::

   ``uv version`` only touches ``[project] version``. After a manual bump,
   update ``CHANGELOG.md`` and ``docs/_static/switcher.json`` yourself — or
   just use ``uv run release``, which does both.

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
