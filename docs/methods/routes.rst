.. _route_cgr_deconvolution:

======================
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
----------------

``build_route_cgr()`` is the typed counterpart to ``compose_route_cgr()``. It
returns a ``RouteCGRBuildResult`` with either the RouteCGR or a diagnostic; the
older mapping-or-``None`` API remains available for compatibility.

For route serialization, ``build_route_trees()`` returns a ``RouteExportResult``
with all skipped-route diagnostics. ``make_json()`` retains the existing v1
dict/list output, while ``strict=True`` on ``build_route_trees()`` or
``write_routes_json()`` raises before writing an incomplete export.
