Pull request review checklist
=============================

Use this checklist before approving a pull request. It is intended for human
reviewers and coding agents that need enough project context to avoid approving
working code that still leaves release, documentation, dependency, or operational
gaps.

Review standard
---------------

Approve a pull request when it improves SynPlanner's overall code health and the
remaining issues are not important enough to block forward progress. Do not hold
an improvement for cosmetic perfection, but do block changes that reduce
correctness, maintainability, reproducibility, or release safety.

Separate comments by severity:

- ``Blocking``: correctness, data loss, security, reproducibility, stale process
  cleanup, broken public API/CLI, dependency lock inconsistency, missing tests for
  risky behavior, or documentation that would make copy-pasted commands fail.
- ``Non-blocking``: small naming, local style, minor cleanup, or future
  refactors that are not needed for this pull request.
- ``Question``: unclear intent or tradeoff where the answer may change the
  review decision.

Understand the change first
---------------------------

Before line-by-line review:

- Read the pull request description, linked issues, and any experiment notes.
- Identify the user-facing behavior, public CLI/API surface, and release impact.
- Check whether the change is a bug fix, performance change, dependency change,
  documentation-only change, or release preparation.
- Review each changed file intentionally. Generated files, lock files, notebooks,
  and Docker files deserve different scrutiny than handwritten implementation
  code, but none should be ignored.

Implementation checklist
------------------------

Check code against the existing project shape:

- Prefer existing SynPlanner patterns over new abstractions unless the new
  abstraction removes real duplication or risk.
- Keep chemistry-specific logic in ``synplan.chem`` modules rather than generic
  orchestration modules.
- Keep CLI behavior and Python API behavior aligned. If a CLI flag is added or
  renamed, update the user guide and any copy-paste commands.
- Avoid silent fallbacks for data-processing failures. Pipeline crashes,
  ``BrokenProcessPool``, missing dedup keys, parse failures, and cache loading
  errors should either be explicit or documented as intentionally recoverable.
- For parallel chemistry code, check worker cleanup, timeout semantics, batch
  sizing, file descriptor use, and whether exceptions are surfaced from worker
  futures.
- When reviewing reaction standardization, filtering, and rule extraction —
  verify progress reporting, ``ignore_errors`` behavior, ``error_file`` output,
  deduplication keys, and summary counts.
- Verify ML training changes by checking dataset cache invalidation,
  train/validation leakage, long-tail class behavior, logger configuration,
  result directory creation, checkpointing, and GPU/CPU assumptions.
- For logging integrations, keep optional remote services out of core
  dependencies. Use extras such as ``SynPlanner[litlogger]``,
  ``SynPlanner[wandb]``, ``SynPlanner[mlflow]``, or ``SynPlanner[loggers]``.
- Do not introduce private-library workarounds without a short comment explaining
  the dependency version or limitation being worked around.

Tests and local checks
----------------------

Run the smallest useful check set first, then broaden based on risk.

For Python changes:

.. code-block:: bash

   uv run ruff format <changed-python-files>
   uv run ruff check <changed-python-files>
   uvx ty check <changed-python-files>

For targeted behavior:

.. code-block:: bash

   uv run pytest <targeted-test-files-or-test-names> -q

For broad changes touching shared pipeline, CLI, training, or dependency
behavior:

.. code-block:: bash

   uv run pytest -q

For dependency metadata changes:

.. code-block:: bash

   uv lock --check

If a check is not run, the review should say why. Do not approve a high-risk
change based only on reasoning if a practical targeted test exists.

Dependency and security review
------------------------------

When ``pyproject.toml`` or ``uv.lock`` changes:

- Confirm ``uv.lock`` is consistent with ``pyproject.toml``.
- Check that optional integrations are extras instead of mandatory dependencies.
- Inspect lock-file changes for unexpected transitive upgrades or downgrades.
- For PyTorch, PyTorch Lightning, RDKit, chython, and remote logger packages,
  check whether the selected version is compatible with supported Python versions
  and whether the change was intentional.
- Review dependency diffs in GitHub when available, and do not rely only on the
  manifest diff.

Documentation and release checklist
-----------------------------------

Any change that affects users, developers, installation, commands, configuration,
or release artifacts should update documentation in the same pull request.

Check the relevant files:

- ``docs/user_guide/cli_interface.rst`` for CLI commands and flags.
- ``docs/configuration/*.rst`` for YAML/config fields.
- ``docs/api_reference/*.rst`` for new public modules that should appear in API
  docs.
- ``tutorials/*.ipynb`` when the notebook workflow would otherwise show stale API calls.
- ``CHANGELOG.md`` for user-visible changes, release notes, dependency changes,
  and developer workflow changes.
- ``docs/_static/switcher.json`` when cutting a new stable version.
- ``docs/get_started/docker_images.rst`` when the documented GHCR image version
  changes.
- Docker build files and ``.github/workflows/build-docker.yml`` when image tags,
  build args, Python versions, or extras change.

Do not manually edit generated ``docs/_build`` output. Rebuild docs instead, or
leave generated output untouched.

For docs-only or docs-heavy changes, validate with a temporary build directory:

.. code-block:: bash

   uv run sphinx-build -b html docs /tmp/synplanner-docs-html

Version and Docker notes
------------------------

Patch releases are appropriate for bug fixes, dependency constraints,
documentation repair, and robustness improvements that do not intentionally
change the public feature set.

Use ``uv`` for version bumps:

.. code-block:: bash

   uv version --bump patch --no-sync
   uv lock --check

Docker images are tagged from ``project.version`` in ``pyproject.toml`` by the
Docker GitHub Actions workflow. For version ``1.4.4``, the expected GHCR tags are:

- ``ghcr.io/laboratoire-de-chemoinformatique/synplanner:1.4.4-cli-amd64``
- ``ghcr.io/laboratoire-de-chemoinformatique/synplanner:1.4.4-gui-amd64``

Before accepting a release-preparation pull request, check that
``docs/get_started/docker_images.rst`` uses the same version.

Final review decision
---------------------

Approve only when:

- The pull request purpose is clear and the implementation matches it.
- Blocking review threads are resolved or explicitly deferred with agreement.
- Required checks pass, or skipped checks are justified in the review.
- Dependency, docs, changelog, and Docker/version updates are complete for the
  scope of the change.
- No unrelated refactors, generated-output churn, or local experiment files are
  included accidentally.

References
----------

- GitHub Docs: `Reviewing proposed changes in a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/reviewing-proposed-changes-in-a-pull-request>`_
- Google Engineering Practices: `The Standard of Code Review <https://google.github.io/eng-practices/review/reviewer/standard.html>`_
- Google Engineering Practices: `What to look for in a code review <https://google.github.io/eng-practices/review/reviewer/looking-for.html>`_
- Keep a Changelog: `Keep a Changelog 1.1.0 <https://keepachangelog.com/en/1.1.0/>`_
