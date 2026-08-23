Package layout
==============

Where a new module goes, and what it is called. These rules exist so that the
answer to "where do I put this?" does not depend on who is asking.

Rule 1 — four layers, imports point down
----------------------------------------

.. code-block:: text

   interfaces/     entry points: CLI, GUI          may import anything
   mcts/  ml/      engines: search, learning       chem, utils
   chem/           chemistry domain                utils
   utils/          infrastructure                  nothing from synplan

``utils`` is a leaf: it may not import from ``chem``, ``mcts`` or ``ml``. A
module that needs one of them is not infrastructure — it is either domain code
(``chem``) or assembly code (``interfaces``).

``mcts`` and ``ml`` share a tier and may import each other: reinforcement
training drives a search, and the search evaluates with a network.

The layer graph is asserted in ``tests/unit/test_package_layers.py``.

Rule 2 — configuration lives beside its domain
----------------------------------------------

There is no central configuration module. ``TreeConfig`` lives in
``mcts/config.py``, ``PolicyNetworkConfig`` in ``ml/config.py``,
``SynthonConfig`` in ``chem/synthon/config.py``. ``utils/config.py`` holds only
the base machinery every config inherits — ``BaseConfigModel`` and
``NestedConfigContainer``.

Rule 3 — verbs for stages, nouns for things
-------------------------------------------

A module that *is a step* in a pipeline is named with a verb and exposes the
worker that runs it, plus a module-level convenience function:

.. code-block:: text

   classify.py      Classifier
   synthonise.py    Synthoniser
   fragment.py      Fragmenter
   enumerate.py     Enumerator

A module that *is a thing* is named with a noun and exposes the class of the
same name:

.. code-block:: text

   stock.py         SynthonStock
   coverage.py      Coverage
   transformer.py   SynthonTransformer
   config.py        SynthonConfig

Reading the module name should tell you whether you get a function or a type.

Rule 4 — no package named ``data``
-----------------------------------

Name a package for what it holds. A package of shipped rule files is ``rules``.
A package of pipeline stages that *process* data is named for the processing —
``curation``, not ``data``. The word "data" describes almost every package in a
cheminformatics library and therefore distinguishes none of them.

Rule 5 — adapters sit beside their target, named after their source
--------------------------------------------------------------------

When capability *X* produces artefacts consumed by engine *Y*, the converter
lives next to the artefact's abstraction and is named after *X*:

.. code-block:: text

   chem/reaction/rules/synthon.py     synthon disconnections as priority rules
   mcts/policy/synthon.py             a synthon-driven policy, if one is added

Not ``synthon/priority.py`` and not ``synthon/policy.py``. This keeps the
abstraction in one place when a second implementation arrives: the shared type
stays at the target, and each source is one file named after itself.

Rule 6 — entry points only in ``interfaces``
---------------------------------------------

Domain packages ship no command line. Command implementations, argument
parsing, output contracts and run bookkeeping live in ``interfaces/``; the
domain package exposes the Python API those commands call.

Rule 7 — flat until it hurts
-----------------------------

A package stays flat up to roughly ten modules. Past that, split into
sub-packages by role — never by file type. ``routes/io/``, ``routes/quality/``
and ``routes/representation/`` are roles; a ``models/`` or ``helpers/``
directory is not.

Where does a new use case go?
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Adding
     - Goes to
   * - A chemistry capability
     - ``chem/<domain>/``
   * - A search strategy or objective (forward synthesis)
     - ``mcts/`` — the search owns its direction, not the chemistry
   * - A learned model
     - ``ml/networks/`` and ``ml/training/``
   * - A new source of an existing artefact
     - one file named after the source, beside the artefact's abstraction
       (rule 5)
   * - Virtual library generation
     - ``chem/<generator>/``, sharing the library container in ``chem/``
   * - A command
     - ``interfaces/``
   * - A configuration model
     - beside its domain (rule 2)

Moving a module
---------------

Public import paths are kept working with a shim module: call
:func:`synplan._compat.deprecated_module`, then re-export. Resolve the
re-exports lazily through a module-level ``__getattr__`` when the shim would
otherwise import across a layer boundary — a shim must not reintroduce an edge
rule 1 forbids. See ``chem/reaction_rules/`` for the eager form and
``chem/reaction/routes/route_cgr.py`` for the lazy one.
