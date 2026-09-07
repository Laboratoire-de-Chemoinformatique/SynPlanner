Stereo constraints in search and routes
=======================================

SynPlanner preserves specified stereo during preparation, rule extraction,
reaction application, search, stock selection and route export. Chython supplies
all production stereo operations. Existing policy weights receive a separate
connectivity projection; their proposals are checked against the authoritative
stereo-bearing molecules before acceptance.

Supported scope
---------------

.. list-table:: Structural support and its limits
   :header-rows: 1
   :widths: 25 40 35

   * - Stereo
     - Supported
     - Requires further evidence or implementation
   * - Carbon tetrahedra (R/S)
     - Mapped orientation, inheritance, explicit query constraints, stock checks
     - Selectivity of a newly formed center; epimerization under a procedure
   * - Ordinary double bonds (E/Z)
     - The same constraints, stock selection and coherent forward reconstruction
     - E:Z ratio and stability under a procedure
   * - Allene axial chirality
     - Both termini and their references, stock selection and route reconstruction
     - Experimental selectivity and stability; general cumulenes
   * - Enhanced ABS / OR / AND groups
     - Preservation in CXSMILES and V3000; absolute assignments can be checked
     - Fulfillment of relative or mixture semantics requires assessment
   * - Atropisomers, planar/helical and other non-tetrahedral stereo
     - Explicit unsupported-input diagnostics for recognized encodings
     - Deferred; no silent conversion to an absolute tetrahedron

Mapped orientation is the reference, not a CIP letter comparison. R can become S
when substituent priorities change while spatial configuration is retained.
Unspecified stereo is unknown. OR represents relative stereo with unresolved
absolute identity; AND represents a mixture and does not imply a 50:50 ratio.
A structural label never establishes ee, er, dr, E:Z or isolated purity.

Planning behavior
-----------------

The default ``TreeConfig(stereo_mode="proposal")`` keeps useful paths that still
need a stereo strategy. ``stereo_mode="strict"`` excludes outcomes with unresolved
stereo obligations. Finding every starting material makes a path
``connectivity_solved``; it becomes a winning route only after its stereo
requirements and coherent forward reconstruction also pass.

Creating a required stereocenter, double bond or allene axis produces an explicit
obligation. SynPlanner does not choose an arbitrary configuration or enumerate all
stereoisomers. The route identifies the required geometry, responsible step and
next action: a compatible chiral precursor, supported transformation or documented
separation. Known contradictory configurations are rejected. Mapping exhaustion,
relative/mixture requirements and unsupported stereo remain unassessed.

.. code-block:: python

   tree.run()
   fulfilled = tree.routes()
   proposals = [Route.from_tree(tree, n) for n in tree.proposal_nodes]
   proposal = proposals[0]
   assert proposal.connectivity_solved and not proposal.solved
   print(proposal.stereo_status)
   print(proposal.to_json()["stereo"]["first_responsible"])

Import ``Route`` from ``synplan.chem.reaction.routes.route``. CLI runs write
``stereo_proposals_<target index>.json`` and ``.html`` beside normal route reports,
including when no stereo-fulfilled route was found. HTML reports place unresolved
stereo requirements beside the responsible steps. Route JSON and search records
retain structural and selectivity-evidence status separately.

Exact materials and evidence
----------------------------

The existing InChIKey catalogue stays indexed by connectivity prefix. Each
candidate retains its full key, stereo-bearing SMILES and vendor offers. Chython
checks the explicit required geometry before a record is selected. Opposite and
unspecified isomers cannot satisfy a specified request. Partially specified
requests may accept an explicit material satisfying all specified requirements.
Every leaf needs actual stock, including small reagents. Cost uses the selected
compatible record; an opposite isomer's price cannot be substituted.

Evidence is attached to the particular reactants, products, agents and procedure.
It is not a patent lookup that universally assigns a template's stereochemical
outcome. ``attach_stereo_evidence`` accepts source observations and keeps their
measurement stage, mixture composition, yield basis and other supplied fields.
Unreviewed observations cannot discharge strategy obligations.

.. code-block:: python

   from synplan.chem.stereo_evidence import review_stereo_route

   reviewed = review_stereo_route(
       proposal, step_index=0,
       source="Procedure and analytical record identifier",
       observation={"er": "98:2", "stage": "isolated", "yield_basis": "desired isomer"},
       reviewer="Responsible chemist",
       reason="Reviewed this exact substrate, procedure and desired configuration",
       catalogue=building_blocks,
   )

Review returns a copy and reruns whole-route inheritance, mapping and stock checks.
It can accept a specific transformation; it cannot approve away missing stock,
contradictory geometry or broken mapping. Without a catalogue, only the explicit
stock records pinned to the route are available for rechecking. Editing the
structures, maps, procedure, evidence or selected materials reopens the assessment.
``TreeConfig.stereo_assessments`` can carry the scoped records into another search.
The default pipeline does not predict catalysts or extract patent prose.

Extraction, persistence and compatibility
-----------------------------------------

``RuleExtractionConfig.ignore_stereo`` now defaults to False. Query conversion
preserves parity in the retained reference frame and includes the needed stereo
references. Stereo-only changes survive extraction. Rule identities distinguish
opposite stereo while tolerating atom renumbering. Source-event records distinguish
creation/destruction, retention/inversion, changed reference environments and
annotation gain/loss. Untrusted atom maps produce an unassessed event.

Extraction writes ``<rules>.stereo.jsonl`` with one source record per parsed
reaction and ``<rules>.manifest.json`` with input/rule hashes and schema versions.
The existing failure/audit files account for failed parsing. New rule vocabularies
must be paired with their trained fixed-output policy; matching rule counts are
insufficient. New ranking/filtering checkpoints retain the vocabulary digest.
MHN policies bind runtime rule representations separately. Legacy assets lacking
a manifest retain the existing compatibility checks; their provenance cannot be
recovered from a weight tensor. Rebuild cached training datasets when rules change.

Search records use ``synplan-tree/3`` and still read schemas 1 and 2. Public route
artifacts use ``synplan-routes/2``: target keys are now Chython canonical SMILES,
including stereo. External evaluation adapters must normalize these keys in their
own boundary code. Core search no longer uses RDKit to create an export key.
SMILES retains CX groups and SDF/RDF writers use V3000. RouteCGR keeps checked
source stereo snapshots beside its connectivity graph; changed graphs require
reassessment before those snapshots can be restored.

Runtime and backend
-------------------

Stock retrieval and finalized-precursor caching use indexed lookups. Opposite
stereo requirements can reuse policy proposals while their search constraints
remain separate. Search obligations use persistent local records, so expanding a
node does not copy its complete stereo history. Export and context checks traverse
the route records. No automatic stereoisomer enumeration is performed.

``max_mapping_work`` bounds correspondence work (default 100,000 candidate/check
units), including failed and recursive matches. Stock buckets and route alignment
also have explicit caps. Exhaustion reports an incomplete assessment. These are
bounds on added correspondence work, not a claim that Chython perception,
canonicalization or arbitrary graph isomorphism is linear.

The pinned Chython backend supplies native shared work limits, strict SMILES/MDL stereo
parsing, checked SMARTS constraints, enhanced reaction CXSMILES and stereo
validation before atom-set deduplication. Its MDL writers prepare and check 2D
geometry, and mark unspecified double bonds explicitly. Drawing never changes
the caller's molecule. Unsupported group queries fail instead of silently
matching absolute configurations.

The compiled matcher enforces the shared budget for representable queries;
extended predicates retain the Python path. Recursive constraints share the
same budget. Published wheels include these compiled operations. SynPlanner uses
public graph APIs to preserve native storage during graph edits and rule application.

Chython 1.108 draws specified configurations with filled or aligned hashed wedges
and unspecified configurations with compact, tapered waves. Molecule, reaction
and route depictions share these symbols without changing stored stereochemistry.
Atropisomer planning and general conditions/selectivity prediction remain deferred.
