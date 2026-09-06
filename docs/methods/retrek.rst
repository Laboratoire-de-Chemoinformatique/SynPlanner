.. _retrek_method:

=======================
ReTReK reaction scoring
=======================

ReTReK (Retrosynthesis Tree Reaction EvaluatoR with Knowledge) defines four
per-reaction heuristics for the chemical quality of a disconnection. SynPlanner
calculates them after a search from detached
:class:`~synplan.chem.reaction.routes.route.Route` objects. The search tree is
not needed and its scores are not changed.

Reaction scores
===============

Each reaction scorer consumes a
:class:`~synplan.chem.reaction.scoring.ReactionScoreContext`. The route adapter
constructs that context from each :class:`~synplan.chem.reaction.routes.route.Step`:

* the disconnected product is ``step.product``;
* the forward reaction's reactants are ``step.reaction.reactants``;
* availability comes from ``route.leaves()`` and ``route.unresolved``;
* ``step.origin.rule_id`` indexes the supplied reaction-rule collection.

This distinction matters for reactions with more than one product: the route
records which product the step actually disconnects instead of assuming the
first reaction product.

``CDScore``
-----------

The convergent-disconnection score compares every precursor's heavy-atom count
with an equal fraction of the product's heavy atoms:

.. math::

    \operatorname{CD} = \frac{1}{1 + \frac{1}{n}\sum_i
    \left|\frac{a(P)}{n} - a(R_i)\right|}

This original calculation remains the default. Passing
``normalized_atom_contributions=True`` to ``calculate_cdscore`` instead counts
the mapped heavy atoms each precursor contributes to the selected product,
ignores non-contributing components, and calculates a size-independent balance:

.. math::

    p_i = \frac{c_i}{\sum_j c_j}, \qquad
    \operatorname{CD}_{\mathrm{normalized}} = 1 -
    \frac{\sum_i |p_i - 1/n|}{2(1 - 1/n)}

The original calculation returns zero for a one-precursor reaction. The
normalized calculation returns zero when fewer than two precursors contribute
and requires atom mappings shared between the precursors and the selected
product.

``ASScore``
-----------

The available-substances score is the fraction of a step's precursors that are
purchasable route leaves. One available precursor therefore scores ``1 / 1 =
1``. It reads the verdict stored by ``Route`` and does not query the original
tree or repeat a building-block catalogue lookup.

``RDScore``
-----------

The ring-disconnection score is one when the product has more rings than all
precursors together, and zero otherwise.

``STScore``
-----------

The selectivity-transformation score counts the chemically distinct placements
of each rule-product reaction center in its corresponding precursor. Full
substructure mappings that differ only outside the reaction center are counted
once, as are center placements related by an automorphism of the precursor.
Rule products are matched one-to-one with precursors, independently of component
order. For distinct-site counts :math:`m_i`, the score is:

.. math::

    \operatorname{ST} = \frac{1}{\prod_i m_i}

Enabling STScore requires the ordered reaction-rule collection whose indices
match each route ``Step``'s ``origin.rule_id``.

The low-level ``calculate_stscore`` function uses distinct reactive sites by
default. Pass ``distinct_reactive_sites=False`` to reproduce the original raw
substructure-embedding calculation.

Normalized aggregation
======================

For the scores available on one step, SynPlanner calculates a normalized
weighted mean:

.. math::

    K_{step} = \frac{\sum_{i \in A} w_i s_i}{\sum_{i \in A} w_i}

``A`` contains the non-zero-weight scores that did not return ``UNAVAILABLE``.
Removing an unavailable score from both numerator and denominator keeps the
result in ``[0, 1]`` and preserves the relative meaning of the remaining
weights. If no score is available, the step is unavailable rather than zero.

The route score is the arithmetic mean of its step scores. It is a route-quality
number independent of ``route.provenance.search_score``.

Future work: MCTS-time evaluation
=================================

Using ReTReK inside MCTS remains future work. It needs a design for placing the
chemistry term in selection, retaining the rule and availability information at
the right time, and defining its interaction with back-propagated node values.
The post-search scorer does not stand in for that algorithm.
