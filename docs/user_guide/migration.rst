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

Pickled ``Tree`` instances from 1.4.x are *partially* compatible with
1.5.0. Direct unpickling (via
:meth:`~synplan.mcts.tree.TreeWrapper.load_tree_from_id` or equivalent)
still succeeds because ``TreeWrapper.__setstate__`` uses
``Tree.__new__(Tree)`` plus ``__dict__.update``, so legacy attributes
survive verbatim.

Code paths that only read ``tree.synthesis_route``,
``tree.route_to_node``, or ``tree.nodes[id].precursors_to_expand``
continue to work. Code paths that touch the migrated surfaces fail:

- ``tree.stats.<anything>`` raises ``AttributeError`` (the legacy
  ``stats`` is still a ``dict``).
- ``tree.nodes[id].rule_source`` / ``.rule_key`` / ``.policy_rank`` /
  ``.depth`` etc. raise ``AttributeError`` because ``Node.__dict__`` from
  a 1.4.x pickle lacks the new fields.

No automatic migration is provided — there is no way to reconstruct the
per-node rule provenance that 1.4.x never recorded. The supported
workaround is to re-run the search.

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
