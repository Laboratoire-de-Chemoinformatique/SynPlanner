Native reactor atom mapping
===========================

Native route-consistent mapping is opt-in. Load a complete rule set with one
configuration and use it for the whole tree::

   from synplan.chem.reaction import ReactorConfig
   from synplan.utils.loading import load_reaction_rules

   rules = load_reaction_rules(
       "reaction_rules.tsv",
       ReactorConfig(reconcile_mapping=True),
   )

Do not mix baseline and native reactors, including priority-rule sets. A
native tree is marked with the ``route-origin-v1`` contract. Each node carries
only two allocation ceilings; every molecule carries a map-to-origin table.
Alternative children may reuse the same allocation interval because only one
alternative can occur in a route lineage.

Atom identity
-------------

Matched atoms keep their map and origin when element and isotope are unchanged.
Charge, radical, hydrogen, bond and stereo edits therefore preserve identity.
Product-only atoms and element/isotope substitutions receive consecutive map
and origin identifiers above the parent node's ceilings. Allocation occurs
after Chython patching and before canonicalization. A sparse private legacy-map
view keeps production policy tensors identical to baseline tensors.

RouteCGR modes
--------------

``build_route_cgr`` and the composition adapters accept the keyword-only
``reconcile_atom_mapping`` argument:

* ``True`` forces legacy structural reconciliation;
* ``False`` requires complete origin metadata and performs direct folding;
* ``None`` selects direct folding for native provenance and legacy folding for
  old or imported routes. An automatic direct failure warns and falls back once.

Strict direct folding verifies a global bijection between map IDs and origins
before composing anything. This detects same-element collisions that CGR
composition alone cannot see. Public route JSON remains unchanged and is
therefore treated as legacy input.

Pruning compatibility is intentionally deferred. Validate native mapping with
``enable_pruning=False``.
