Cutting a release
=================

Work through the checklist below, then bump the version and tag. Pushing the
``vX.Y.Z`` tag is what triggers the PyPI and TestPyPI publish workflows.

Pre-release checklist
---------------------

Refresh the agent skill
~~~~~~~~~~~~~~~~~~~~~~~

The agent skill in ``skills/synplanner-usage/`` is hand-written, because
generating it would strip the ordering, defaults and judgement that make it
useful. It therefore needs a deliberate look before each release.

``tests/test_skill.py`` catches the mechanical half — a renamed module, a moved
symbol, a dropped CLI command, a deleted config or docs page — and fails the
build. It cannot tell you that a *new* capability is missing, or that a
recommendation has gone stale.

1. Run ``uv run pytest tests/test_skill.py``. Fix anything it reports by editing
   the skill text, not by loosening the test.
2. **Update the use cases.** If the release adds or changes a workflow, add or
   amend the matching entry in
   ``skills/synplanner-usage/references/tasks.md`` — task title, the API pieces
   in order, and links to the tutorial and docs pages. A new tutorial, a new CLI
   command, or a new recommended default each mean a new or edited entry.
3. **Update ``SKILL.md`` if the guidance itself changed** — a new default, a
   changed chython/RDKit boundary, a workflow that now wants a GPU, advice that
   no longer holds. Leave it alone if only the task list moved.
4. If either file changed, confirm the rendered pages still build:
   :doc:`/agent_skill` and :doc:`/tasks` include them directly.

Both files are published on the docs site and served through ``llms.txt``, so a
stale skill misleads users and their agents until the next release.

Run every gate CI runs
~~~~~~~~~~~~~~~~~~~~~~

These four are what ``ci.yml`` enforces. Run them exactly as written — over the
whole tree, not over the files you touched, or a clean local run can still fail
CI on breakage someone else introduced:

.. code-block:: bash

   uv run --no-sync pytest --cov=synplan --cov-report=xml
   uv run --no-sync ruff format --check synplan tests
   uv run --no-sync ruff check synplan tests
   uvx ty check synplan/

``ruff check`` and ``ruff format --check`` are separate gates. ``check`` says
nothing about line wrapping, so it passes on a file the formatter would rewrite.

Other checks
~~~~~~~~~~~~

- ``CHANGELOG.md`` has an ``[Unreleased]`` section describing the user-visible
  changes.
- Documentation touched by the release is updated — see the checklist in
  :doc:`pr_review`.
- The docs build clean, with no Sphinx warnings:

  .. code-block:: bash

     uv run --group docs sphinx-build -b html docs /tmp/synplanner-docs

  ``--group docs`` is required and is what ``readthedocs.yaml`` uses. Without it
  ``uv run`` falls through to whatever ``sphinx-build`` is on ``PATH``, which on a
  machine with a system Sphinx fails on the missing ``myst_parser`` rather than
  building. CI does not build the docs, so this check is manual.

Dependencies
~~~~~~~~~~~~

Check open Dependabot alerts before tagging:

.. code-block:: bash

   gh api repos/Laboratoire-de-Chemoinformatique/SynPlanner/dependabot/alerts \
     --paginate -q '.[] | select(.state=="open") | [.security_vulnerability.package.name,
     .security_vulnerability.severity,
     (.security_vulnerability.first_patched_version.identifier // "none")] | @tsv' | sort -u

Most close by raising a floor in ``[tool.uv] constraint-dependencies``, which
also records which advisory each floor is for. Relock with a cooldown, because a
freshly published release is the one worth distrusting, and exempt our own
package so its newest version still resolves:

.. code-block:: bash

   # dates as YYYY-MM-DD: cutoff seven days back, exemption a day ahead
   uv lock --exclude-newer 2026-07-28 \
           --exclude-newer-package chython-synplan=2026-08-05

Leave an alert open rather than take a major-version jump on the eve of a
release; note the reason beside the constraint block and handle it separately.

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

The compare links and the switcher entry are derived from the newest ``vX.Y.Z``
git tag, not from ``pyproject.toml``. If a version was bumped but never tagged,
the script says so and you must fold that CHANGELOG section into the new release
by hand.

Patch releases are appropriate for bug fixes, dependency constraints,
documentation repair, and robustness improvements that do not intentionally
change the public feature set.

Manual path
~~~~~~~~~~~

To bump only ``pyproject.toml`` + ``uv.lock`` without the dependent edits:

.. code-block:: bash

   uv version                          # show current, e.g. synplanner 1.5.1
   uv version --short                  # 1.5.1
   uv version --bump patch --no-sync   # 1.5.1 -> 1.5.2
   uv lock --check

``--no-sync`` writes ``pyproject.toml`` and relocks ``uv.lock`` without
reinstalling the local environment. Use ``--dry-run`` to preview, or
``--frozen`` to skip the lockfile update entirely.

.. note::

   ``uv version`` only touches ``[project] version``. After a manual bump,
   update ``CHANGELOG.md`` and ``docs/_static/switcher.json`` yourself — or
   just use ``uv run release``, which does both.

Publish to GHCR (maintainers)
-----------------------------

Docker images are tagged from ``project.version`` in ``pyproject.toml`` by the
Docker GitHub Actions workflow. For version ``1.4.4``, the expected GHCR tags
are:

- ``ghcr.io/laboratoire-de-chemoinformatique/synplanner:1.4.4-cli-amd64``
- ``ghcr.io/laboratoire-de-chemoinformatique/synplanner:1.4.4-gui-amd64``

Images are published automatically by CI on pushes to ``main`` and manual
dispatches. To push locally (requires write permissions to the repository's
packages):

.. code-block:: bash

   VERSION=$(python -c 'import tomllib,sys;print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')
   docker login ghcr.io -u USERNAME -p TOKEN
   REPO=ghcr.io/laboratoire-de-chemoinformatique/synplanner
   docker tag synplan:dev-cli-amd64 ${REPO}:${VERSION}-cli-amd64
   docker tag synplan:dev-gui-amd64 ${REPO}:${VERSION}-gui-amd64
   docker push ${REPO}:${VERSION}-cli-amd64
   docker push ${REPO}:${VERSION}-gui-amd64
