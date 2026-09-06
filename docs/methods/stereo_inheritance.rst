Stereo inherited from stock: route audit
=======================================

``audit_stereo_inheritance`` is an opt-in route-level prototype. It traces
specified target configurations through unchanged mapped environments and checks
every leaf against an explicit compatible catalogue record. Existing search,
canonicalisation, stock lookup and route export defaults are unchanged.

.. code-block:: python

   from chython import smiles
   from synplan.chem.reaction.routes.stereo import audit_stereo_inheritance
   from synplan.chem.reaction.routes.stereo_io import read_stereo_route

   # Preserve the original target and source tree before any stereo removal.
   route, mapping_sources = read_stereo_route(saved_route_tree)
   result = audit_stereo_inheritance(
       route, smiles(original_target_smiles), catalogue,
       mapping_sources=mapping_sources,
   )
   evidence = result.to_dict()
   if result.supported:
       reconstructed_route = result.route

Callers already holding a detached ``Route`` may pass it directly, with the source
of its current reaction atom maps for every zero-based step. Local numbers
invented by parsing unmapped SMILES are not mapping evidence. The strict import
adapter checks molecule/reaction node agreement before linking objects, rebases
local maps, and preserves original reaction strings and source identifiers. Its
``strip_stereo=True`` option is for explicit recovery experiments only.

The audit supports carbon tetrahedra and ordinary double bonds. It compares
orientations in mapped neighbour frames, so changing CIP priority does not imply
inversion. New centres, changed local environments, contradictory configurations,
unsupported stereo types and uncertain mappings remain unresolved proposals.
Symmetry enumeration is bounded (256 mappings by default); reaching the bound
abstains. The import adapter conservatively checks possible carbon requirements
before it has target-specific obligations and may abstain unnecessarily.

Connectivity keys only identify catalogue candidates. A compatible full record
must specify every required configuration; wrong and unspecified stereoisomers
cannot satisfy a requirement. All other leaves also need actual records, with
no small-molecule shortcut. When offers are present, the reported price belongs
to the selected compatible record. Input molecules, route provenance, conditions
and catalogue objects remain unchanged.

Only complete reconstructions have ``supported=True`` and a returned ``Route``.
The forward check preserves shared adjacent molecule objects and reproduces all
target-linked configurations from assigned stock. The machine-readable ledger
identifies original target atoms, mapped paths, selected records and the first
responsible step/leaf. Inferred labels are explicitly marked as reconstructed.
Use ``to_dict()['reconstructed_steps']`` to retain mapped stereo; legacy export
normalisation is outside this prototype's contract.

This is graph inheritance, not a reaction selectivity or laboratory certificate.
Conditions remain unassessed. Alpha-carbonyl stereocentres receive a structural
review flag rather than a predicted epimerization outcome. Protection edits must
be audited again with valid updated maps. Search-state requirements, caches,
deduplication, exact stock termination and standard export integration require a
separate change.
