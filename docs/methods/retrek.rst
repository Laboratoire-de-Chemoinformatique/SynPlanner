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
* rule provenance comes from ``step.origin`` through an injected resolver.

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

``ASScore``
-----------

The available-substances score is the fraction of a step's precursors that are
purchasable route leaves. It reads the verdict stored by ``Route`` and does not
query the original tree or repeat a building-block catalogue lookup.

``RDScore``
-----------

The ring-disconnection score is one when the product has more rings than all
precursors together, and zero otherwise.

``STScore`` -- provisional
-------------------------

The existing STScore matching formula is known to be incorrect. It is retained
locally as work in progress but is disabled by default and must be corrected
before this feature is published. Enabling it requires a resolver from a route
``Step`` to its canonical retro reactor.

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
