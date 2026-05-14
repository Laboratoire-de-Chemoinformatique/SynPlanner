.. _priority_rules:

================================
Priority rules
================================

A *priority rule* is a curated retrosynthetic SMARTS pattern that you want
the planner to try ahead of the learned policy on every node. Typical
use cases:

- A few hand-picked Ugi 4CR templates when synthesising peptide-like targets.
- A deprotection rule set for fully protected intermediates.
- A small in-house ring-formation library you trust more than the policy on
  a specific scaffold class.

Priority rules sit *outside* the policy network: they bypass policy
ranking, enter sibling selection with a strong UCB prior, and are tracked
under their own counters in :class:`~synplan.mcts.tree.TreeStats`.

Passing priority rules to ``Tree``
----------------------------------

Priority rules are passed as a mapping of named sets:

.. code-block:: python

    from synplan.mcts.tree import Tree
    from synplan.utils.config import TreeConfig

    Tree(
        target=...,
        config=TreeConfig(use_priority=True),
        priority_rules={
            "ugi": ugi_rules,
            "boc_deprotection": boc_rules,
        },
        ...
    )

Each set's name becomes the ``rule_source`` label on every child it
produces and gets its own counter pair under
``tree.stats.per_priority_source[<name>]``.

Validation rules:

- The reserved name ``"policy"`` is rejected (raises ``ValueError``).
- Empty rule lists are rejected: either populate the set or remove
  the key.
- Non-empty string keys are required.
- Setting ``config.use_priority=True`` without supplying ``priority_rules``
  raises ``ValueError`` rather than running silently with no priority
  effect.

Mechanism
---------

Priority rules match by chython substructure isomorphism
(``pattern < molecule``) and enter expansion with ``prob=1.0``. The
existing per-product fragment-count multiplier in
:meth:`~synplan.mcts.tree.Tree._add_child_if_new` then applies the
relation

.. math::

   \mathrm{scaled\_prob} = \mathrm{prob} \times n_{\text{qualifying fragments}}

so a 4-fragment priority disconnect (e.g. an Ugi 4CR-style template that
produces four valid precursor fragments) enters UCB with prior 4. This
is intentional: curated multi-fragment disconnects are designed to
dominate sibling selection over single-fragment policy children.

If you observe ``Node.prob`` values larger than 1.0, this is expected
behaviour when a priority set introduces non-policy rules; see the
priority semantics section above.

Iterated application
--------------------

``TreeConfig.priority_rule_multiapplication`` (alias kwarg
``apply_reaction_rule(multirule=True, rm_dup=True)``) repeatedly
applies the same rule to its own product set until no new tuple is
produced. The flag fires for *every* priority source in the mapping,
not for the policy.

The intended use is bulk transformations such as stripping every
Boc/Cbz/etc. protective group from a fully protected substrate in one
expansion step.

SMARTS dialect note
-------------------

Priority rules are loaded through ``Reactor.from_smarts``, which parses
patterns with chython's aromaticity perception. This differs from
RDKit's. Patterns authored against RDKit may match unexpectedly under
chython.

The :func:`synplan.chem.reaction_rules.parse_priority_rules` helper
reports a SMARTS as broken via chython *and* runs an RDKit fallback
parse so you can distinguish "broken pattern" from "dialect mismatch".
Validate priority rules on a known target before scaling up.

Per-source statistics
---------------------

``Tree.stats`` (a :class:`~synplan.mcts.tree.TreeStats` dataclass)
exposes:

- ``policy_rules_tried`` / ``policy_rules_succeeded``: policy-only
  counters.
- ``priority_rules_tried`` / ``priority_rules_succeeded``: aggregate
  across all priority sources.
- ``per_priority_source[<set_name>].tried`` and ``.succeeded``: the
  per-set breakdown.

``Tree.to_stats_dict()`` flattens ``per_priority_source`` into the
output dict for CSV/JSON export. Per-route priority usage stats
(``n_routes_with_priority``, ``fraction_routes_with_priority``) treat
*any* non-policy step as priority.

Rule provenance on nodes and routes
-----------------------------------

Every child carries:

- ``rule_source``: either the priority set name or
  :data:`~synplan.mcts.tree.POLICY_SOURCE_NAME` (``"policy"``).
- ``rule_key``: collision-safe identifier formatted as
  ``<source>:<id>`` so priority and policy IDs never collide in
  serialization.
- ``policy_rank``: exact 1-indexed Top-N position from the expansion
  function, or ``None`` for priority children.

Route SVG, JSON, and RDKit exports propagate this metadata, and route
SVGs annotate rule keys and policy ranks alongside molecules.

Tutorial
--------

Tutorial 13 (`Priority Rules`_) walks through curating a small Ugi
priority set, running planning with and without it, comparing
``per_priority_source`` counters, and rendering route SVGs that surface
priority hits.

.. _Priority Rules: ../user_guide/13_Priority_Rules.ipynb
