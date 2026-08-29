.. _routes_methods:

======
Routes
======

.. _route_as_a_graph:

A route as a graph
==================

A reaction condenses into a **CGR**: one graph holding reactants and products at
once, where every bond carries what it was and what it became. A whole route
condenses the same way, into a **RouteCGR** — a single graph of the entire
synthesis, with each bond additionally carrying which step changed it.

Writing a route as one graph is what makes it *comparable*. A graph can be
hashed, so routes are deduplicated through a dictionary rather than compared
pairwise, and near-identical ones share a bucket. Reducing the RouteCGR to the
bonds the synthesis actually forms or breaks gives the **SB-CGR**, which groups
routes by strategy rather than by spelling.

Step order and identity
-----------------------

Step order is part of a route's identity, and ``route_cgr_hash()`` keeps it.
That is the key to deduplicate on: the sequence of a synthesis is usually part
of what the synthesis is, so two routes making the same disconnections in a
different order are two routes to run, not one written twice.

``route_cgr_hash_without_route_order()`` drops the order and answers a narrower
question — whether two routes are the same *set* of disconnections however they
were sequenced. Use it only once you have decided the sequence does not matter
for the question you are asking. ``route_order_variant_sets()`` names the groups
that differ solely in step order, so you can check whether the distinction
arises in your route set before committing to an answer.

A single search does not hand you duplicates. Measured on a celecoxib search,
all 330 winning routes were distinct — by ordered steps, by step set, and by
RouteCGR hash alike — and none differed from another only in step order.
Deduplication earns its keep where route sets meet: two searches on one target,
a search against a literature set, the output of two tools. Composition costs
milliseconds per route; hashing costs over a second, so budget for the set size
you actually have.

.. _route_cgr_deconvolution:

RouteCGR Deconvolution
======================

``compose_route_cgr()`` builds a RouteCGR that carries the native per-step
atom and bond labels required to reconstruct the mapped reaction sequence. No
post-processing metadata attachment is required.

.. code-block:: python

    from synplan.chem.reaction.routes.representation import (
        compose_route_cgr,
        routes_dict_from_route_cgrs,
    )

    composed = compose_route_cgr(tree, route_id, preserve_transient_bonds=True)
    restored_reactions = routes_dict_from_route_cgrs({route_id: composed["cgr"]})

RouteCGR composition assumes route-level atom mapping. The same atom-map number
must refer to the same chemical atom throughout the route, and distinct atoms
must not reuse the same map number. Independently mapped reaction steps with
local atom numbering are not supported as direct input and may produce undefined
composition or deconvolution behavior.

Deconvolution reconstructs ``ReactionContainer`` steps from labels stored on
RouteCGR atoms and bonds. It does not promise exact planning-tree JSON, sibling
ordering, UI metadata, or the original formatted mapped reaction SMILES. Those
are serialization concerns outside the bare RouteCGR representation.

For tests and debugging, ``compose_route_cgr(..., return_reactions_dict=True)``
can also return the composed per-step reactions eagerly, but this is deliberately
not the default hot path.

For generated planning routes, ``scripts/route_cgr_roundtrip.py`` can be used as
a developer check. It composes winning routes with ``preserve_transient_bonds``
enabled and verifies side-wise atom-map preservation for the reconstructed
reaction steps. Generated benchmark data and round-trip result folders are
local artifacts and should not be committed.


Typed Route APIs
================

``build_route_cgr()`` is the typed counterpart to ``compose_route_cgr()``. It
returns a ``RouteCGRBuildResult`` with either the RouteCGR or a diagnostic; the
older mapping-or-``None`` API remains available for compatibility.

For route serialization, ``build_route_trees()`` returns a ``RouteExportResult``
with all skipped-route diagnostics. ``make_json()`` retains the existing v1
dict/list output, while ``strict=True`` on ``build_route_trees()`` or
``write_routes_json()`` raises before writing an incomplete export.
