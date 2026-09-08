.. _migration:

================================
Migration guide
================================

This page collects breaking changes across releases and the minimal
edits needed to update calling code. New entries are added at the top.
For the full per-release log, see :doc:`/release_notes`.

.. contents::
   :local:
   :depth: 2

Unreleased
==========

Tree pickle persistence removed
--------------------------------

``Tree.save_pickle()`` was removed. Save and read finished searches as JSON or
JSON.GZ with the existing search-record API:

.. code-block:: python

   from synplan.mcts.record import read_search_record, write_search_record

   write_search_record(tree, "tree.json.gz")
   record = read_search_record("tree.json.gz")
   routes = list(record.routes())

The record preserves nodes, statistics and selected vendor offers without
serializing the catalogue, its SQLite connections or the policy network. It
supports analysis and route reconstruction, but cannot resume the search.
Loading pickled trees is unsupported.

1.6.0
=====

chython 1.100 changes what your SMARTS mean
--------------------------------------------

SynPlanner now requires ``chython-synplan`` 1.100 exactly. Its rewritten SMARTS
parser follows Daylight on three primitives that chython used to read
differently. Hand-written patterns — priority rules, ``func_groups_list``,
protection group definitions — that use them silently change meaning:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Primitive
     - Old chython reading
     - chython 1.100 (Daylight)
   * - ``A``
     - any atom
     - any *aliphatic* atom; use ``*`` for any atom
   * - ``x``
     - number of heteroatom neighbours
     - ring connectivity; the heteroatom count is now ``y``
   * - unmarked charge
     - charge 0
     - unconstrained; ``[N]`` also matches a cationic N, ``[N+0]`` does not

Rewrite ``A`` as ``*`` and ``x`` as ``y`` wherever the old meaning was intended,
and add ``+0`` where a neutral atom was assumed. Rules extracted by SynPlanner
itself are unaffected — they are generated from CGRs, not hand-written.

Route post-processing moved to ``synplan.chem.reaction.routes``
----------------------------------------------------------------

Route-level post-processing now lives in ``synplan.chem.reaction.routes``. This includes
RouteCGR construction and hashing, route IO, route clustering, route analysis,
depiction, route-quality scoring, and related route analysis helpers. Notebook plotting helpers now live in ``synplan.chem.reaction.routes.notebook_plots``.

The interim ``synplan.routes`` package and the older ``synplan.route_quality``
compatibility namespace were removed. The ``synplan.chem.reaction_routes`` namespace
remains as a deprecated compatibility layer for the ``main``-branch module paths
``io``, ``route_cgr``, ``clustering``, ``leaving_groups``, and ``visualisation``.
New code should use the canonical paths below.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Removed path
     - New path
   * - ``synplan.routes.analysis``
     - ``synplan.chem.reaction.routes.analysis``
   * - ``synplan.routes.clustering``
     - ``synplan.chem.reaction.routes.clustering``
   * - ``synplan.routes.depiction``
     - ``synplan.chem.reaction.routes.representation.depiction``
   * - ``synplan.routes.io``
     - ``synplan.chem.reaction.routes.io``
   * - ``synplan.routes.notebook_plots``
     - ``synplan.chem.reaction.routes.notebook_plots``
   * - ``synplan.routes.route_cgr``
     - ``synplan.chem.reaction.routes.representation``
   * - ``synplan.routes.route_cgr.hash``
     - ``synplan.chem.reaction.routes.representation.hash``
   * - ``synplan.routes.quality``
     - ``synplan.chem.reaction.routes.quality``
   * - ``synplan.routes.quality.protection``
     - ``synplan.chem.reaction.routes.quality.protection``
   * - ``synplan.route_quality``
     - ``synplan.chem.reaction.routes.quality``

Reaction rules moved to ``synplan.chem.reaction.rules``
-------------------------------------------------------

Rule analysis, extraction, priority-rule parsing, and the QueryCGR/Morgan rule
representations now live under ``synplan.chem.reaction.rules`` (and
``synplan.chem.reaction.rules.representation``). ``synplan.chem.reaction_rules``
remains as a deprecated shim for ``analysis``, ``extraction``, and ``priority``;
importing from it emits a ``DeprecationWarning``.

MCTS expansion wrappers moved to ``synplan.mcts.policy``. The old
``synplan.mcts.expansion`` module is gone.

Tree persistence and route exports
----------------------------------

``TreeWrapper`` was removed. Use the search-record API above for finished-search
persistence. Route JSON and CSV exports live in
``synplan.chem.reaction.routes.io``. Update imports from
``synplan.mcts.tree.export_tree_to_json`` and
``synplan.mcts.tree.export_tree_to_csv`` to the canonical route-I/O module.

1.5.0
=====

Per-node state moved off ``Tree``
---------------------------------

All nine ``Tree.nodes_*`` parallel dicts were removed. The values now
live directly on each :class:`~synplan.mcts.node.Node`. Any read of the
old ``Tree`` attributes raises ``AttributeError`` with a hint pointing
at the new location.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Removed
     - New location
   * - ``tree.nodes_visit[nid]``
     - ``tree.nodes[nid].visit``
   * - ``tree.nodes_depth[nid]``
     - ``tree.nodes[nid].depth``
   * - ``tree.nodes_prob[nid]``
     - ``tree.nodes[nid].prob``
   * - ``tree.nodes_init_value[nid]``
     - ``tree.nodes[nid].init_value``
   * - ``tree.nodes_total_value[nid]``
     - ``tree.nodes[nid].total_value``
   * - ``tree.nodes_rules[nid]``
     - ``tree.nodes[nid].rule_id``

Note that ``nodes_rules`` renamed to ``rule_id`` on the new attribute.
All the others keep their suffix as the attribute name.

``Tree.stats`` is now a typed dataclass
---------------------------------------

``Tree.stats`` returns a :class:`~synplan.mcts.tree.TreeStats` dataclass
instead of a plain ``dict``. Use attribute access:

.. code-block:: python

   # before
   tree.stats["expansion_calls"]
   tree.stats.get("expansion_calls", 0)

   # after
   tree.stats.expansion_calls

Subscripting on a known field raises ``TypeError`` with a migration hint;
unknown keys still raise ``KeyError``. The defaults on every
``TreeStats`` field are static, so the ``.get(..., default)`` form is
obsolete. Drop the default.

``Tree.to_stats_dict()`` is unchanged: it still returns a flat
``dict[str, Any]`` with the same keys, so CSV/JSON consumers downstream
are unaffected.

``EvaluationStrategy.evaluate_node`` signature
----------------------------------------------

The legacy ``(node, node_id, nodes_depth, nodes_prob)`` parameters
collapse into a single ``nodes: dict[int, Node]`` mapping. Custom
evaluator subclasses must be updated:

.. code-block:: diff

   -def evaluate_node(self, node, node_id, nodes_depth, nodes_prob):
   -    depth = nodes_depth[node_id]
   -    prob = nodes_prob.get(node_id, 0.0)
   +def evaluate_node(self, node, node_id, nodes):
   +    depth = nodes[node_id].depth
   +    prob = nodes[node_id].prob

Pickled trees from 1.4.x
------------------------

Legacy pickled trees are unsupported. Re-run the search and save a JSON search
record; the per-node rule provenance missing from 1.4.x cannot be reconstructed.

YAML ``key:`` (null) for nested standardization / filtering configs
-------------------------------------------------------------------

In ``ReactionStandardizationConfig`` and ``ReactionFilterConfig``, an
empty YAML value (``functional_groups_config:``) parses to Python
``None`` and previously left the field as ``None`` — silently disabling
the step.

The new behaviour treats ``key:`` and ``key: {}`` as equivalent: both
instantiate the nested config with defaults; explicit dicts pass
overrides through.

**To disable a step you must now omit the key entirely.** The field
default of ``None`` is preserved when the key is absent. If your YAML
used ``key:`` to disable a step, replace those lines with omission.

``apply_reaction_rule`` default ``top_reactions_num``
-----------------------------------------------------

The default is raised from 3 to 5. Rationale: with priority rules
enabled, multi-fragment disconnects (e.g. Ugi 4CR) frequently produce
more than three valid product sets per rule application, and the old
cap silently truncated valid disconnects. Five matches the typical
priority-rule fan-out without inflating policy expansion noticeably.

This is a global change. It affects MCTS rollouts
(``synplan/mcts/evaluation.py``) and per-node expansion
(``synplan/mcts/tree.py``) — every call site that did not pass the
kwarg explicitly. Existing planning runs may produce more child nodes
per rule application and consequently larger trees, different timings,
and different routes.

Pin the old behaviour explicitly:

.. code-block:: python

   apply_reaction_rule(..., top_reactions_num=3)
