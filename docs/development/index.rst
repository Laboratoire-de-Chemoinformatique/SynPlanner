Development
===========

Maintainer-facing documentation: setting up a local environment, reviewing pull
requests, and cutting a release.

.. toctree::
   :maxdepth: 1

   setup
   package_layout
   pr_review
   release
   chemist_review

- :doc:`setup` — uv environment, running tests, building Docker images.
- :doc:`package_layout` — which package a new module belongs in, and how it
  is named.
- :doc:`pr_review` — the checklist to work through before approving a pull
  request, including which documentation a change must update.
- :doc:`release` — version bumping, the pre-release checklist, and publishing
  images to GHCR.
- :doc:`chemist_review` — the open chemistry questions on the ``llm``-provenance
  disconnection rules, and the rules held out of ``rules.json`` until they are answered.
