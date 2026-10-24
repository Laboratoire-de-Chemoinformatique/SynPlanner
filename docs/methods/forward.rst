.. _forward_prediction:

Forward reaction assessment
===========================

Forward prediction estimates candidate products from reactant SMILES.

Models and shared prediction policy
-----------------------------------

``OnnxChemformer`` lives in ``synplan.ml.inference.chemformer``. Its directory
contains ``encoder.onnx``, ``decoder.onnx``, ``vocabulary.json`` and ``model.json``.
The compact SynFwd-1 bundle is at ``synplan_data/forward/synfwd-1/v1/``.
``ForwardChimera`` lives in ``synplan.ml.inference.chimera``. Its directory is
``synplan_data/forward/forwardchimera-denovo/v1/`` and contains ``encoder.onnx``,
``decoder.onnx``, ``decoder_with_past.onnx``, ``model.json`` and
``source/vocab.txt``. Both directories are ignored by Git. A single graph is
insufficient. Original exports, ZIP, checkpoint and attribution are retained.

Both ONNX adapters use the existing base dependencies, without importing
PyTorch, RetroChimera, Lightning, Syntheseus or code from the asset directory.
Sessions load once. CPU defaults to four threads. ``device="cuda"`` requires an
independently installed ONNX Runtime CUDA provider and its CUDA/cuDNN libraries;
provider initialization failures raise instead of silently falling back.

Chimera additionally accepts explicit ``backend="ckpt"``. This needs optional
PyTorch (``SynPlanner[forward-checkpoint]``), but neither RetroChimera nor
Lightning. Export uses the same compact architecture with optional PyTorch and
ONNX tooling (``SynPlanner[forward-export]``). The default remains ONNX.
Chemformer's existing adapter supports ONNX only.

Each input is reactants-only SMILES, with dot-separated components, not a
``reactants>>product`` string. There is no separate reagent channel. Chython
formatting removes mapping; RDKit produces canonical isomeric reactant SMILES.
Stereo and isotopes remain in inputs; no tautomer standardization is performed.
For stereo-bearing mixtures, components are parsed separately and combined with
chython molecule union. This avoids chython 1.108 reversing a chiral center at
the root of a later component. The same handling applies to generated products.
Inputs exceeding a model's token limit raise without truncation.

Shared pipeline version 2 uses up to ten distinct rooted SMILES per input,
including the canonical representation. Components are sorted consistently.
A local RNG uses seed zero and a SHA-256-derived input-specific seed, so
collection order does not affect augmentation. Root combinations are sampled
with bounded work rather than enumerating an exponential Cartesian product.
Symmetric inputs may yield fewer than ten distinct spellings. These spellings
pass directly to each tokenizer, **without recanonicalization**.

Each spelling generates twenty raw beams. Candidates are parsed/canonicalized
by chython with ``fix_tautomers=False``; invalid candidates are discarded.
Within each augmentation, keep the first occurrence of each complete key from
``inchi_key(molecule, ignore_stereo=True)`` and assign consecutive ranks after
filtering. Across augmentations, identities are ranked by:

1. Best cleaned rank achieved in any augmentation (ascending).
2. Sum of reciprocal cleaned ranks across augmentations (descending).
3. First occurrence (stable tie resolution).

This follows `RetroChimera v1.3.0 ranking
<https://github.com/microsoft/retrochimera/blob/v1.3.0/retrochimera/utils/root_aligned_score.py>`_,
with intentional stereo-free InChIKey grouping instead of upstream canonical
isomeric-SMILES grouping. Full keys and all fragments are used; isotope changes
remain distinguishable. Standard InChI normalization applies, with no extra
tautomer standardization. The numeric ``ranking_scores`` encode upstream's
``reciprocal_support - 1e8 * (best_rank - 1)``; sorting uses the explicit rules.

The representative product comes from the best cleaned rank, then earliest
augmentation and beam position. Its stereochemistry is retained. Its cumulative
sequence ``log_likelihoods`` are separate from aggregate ``ranking_scores``.
Neither is an experimental success probability. Return at most ``num_results``
products, without padding. ``reactants`` records normalized inputs; results
preserve input order. Other fields are ``predictions``, ``canonical_products``,
``product_inchikeys`` and ``augmentation_count``.

.. code-block:: python

   from synplan.ml.inference.chemformer import OnnxChemformer
   from synplan.ml.inference.chimera import ForwardChimera

   chemformer = OnnxChemformer("synplan_data/forward/synfwd-1/v1", threads=4)
   chimera = ForwardChimera("synplan_data/forward/forwardchimera-denovo/v1")
   predictions = chimera.predict(["CC(=O)Cl.N"], num_results=10,
                                 beam_width=20, batch_size=1)
   checkpoint = ForwardChimera("synplan_data/forward/forwardchimera-denovo/v1",
                              backend="ckpt")

``beam_width`` controls decoding per augmentation, independently of the final
result count. The internal ``_predict_batch`` decoders remain available for raw
parity tests; they receive already prepared spellings and retain raw beams.

Chimera architecture and assets
------------------------------

The Figshare release is article **32994914**, file **66654872**,
``forwardchimera_denovo_pistachio.zip`` (MD5
``0eba4b70c9594fde311cd80e80d7b0af``). The source reference is RetroChimera
v1.3.0, commit ``bf5ec59eec9ef32911d5167bf7528e3648622be1``.
The actual checkpoint has eight encoder and eight decoder layers, hidden width
512, eight attention/KV heads, gated SiLU feed-forward width 2048, RMSNorm,
interleaved sinusoidal positions, and separate 343-token embeddings.
It has 76,169,047 trainable parameters. Source limit is 1024 tokens; generation
limit is 512. Stereo-bearing tokens are supported; the upstream tokenizer does
not strip stereochemistry.

Only this checkpoint's architecture is implemented in ``_chimera_torch.py``.
The ONNX encoder, initial decoder and decoder-with-past are float32/opset 17.
The cached decoder gathers self/cross attention states by beam parent indices;
ONNX I/O binding retains those states on the execution device. Beam search
retains finished hypotheses even if they are subsequently pruned from active
beams, following upstream's customised search.

Reproduce the download, Figshare MD5 check and export with:

.. code-block:: bash

   python -m scripts.export_forward_chimera --download \
       --model synplan_data/forward/forwardchimera-denovo/v1

The bundle contains release metadata, architecture/export settings, attribution,
validation results and a SHA-256 manifest excluding itself. RetroChimera is used
only in a separate verification environment. The compact CPU encoder matched
upstream exactly and incremental decoder log probabilities differed by at most
7.63e-6 on fixed inputs. Raw beams matched at widths 1, 10 and 20. ONNX/checkpoint
comparisons also cover CPU, CUDA, expanded/reordered KV state and raw beams.
Stereo-free grouping is tested independently because it intentionally differs.
The width-one upstream comparison uses three identical inputs to avoid its
single-active-item squeeze failure; standalone singleton decoding is tested
separately. Width-ten/twenty comparisons use different inputs.


Annotation persistence
----------------------

``Step.forward_prediction`` optionally stores ``product_inchikey``, ``rank``,
``score`` and ``alert`` under reaction-node ``meta.forward_prediction``.
Annotations belong to steps and do not modify shared reaction containers.
``Route.assessment`` is an optional root-level dictionary. Route JSON also
round-trips ``provenance``, including search score and policy log likelihood.
Legacy documents without these optional fields remain readable.

``Route.from_json(..., on_step=callback)`` calls
``callback(reaction, source_node)`` for each restored step, allowing collection
tools to preserve metadata on the original JSON nodes.


Matching and step scores
-----------------------

``ForwardReactionAssessor`` visits ``route.steps``, taking reactants from
``step.reaction.reactants`` and the expected product from ``step.product``.
The latter matters when a reaction has multiple products.

**One-reactant exception.** Reactions containing exactly one entry in
``step.reaction.reactants`` receive ``score=1.0``, ``rank=null`` and
``alert=false`` without preprocessing, cache lookup or model inference.
One-reactant reactions were intentionally excluded from SynFwd-1 training,
with the assumption that suitable conditions could yield the product.
The same selected convention applies to Chimera; it does not imply Chimera was
trained with the same exclusion. This score records that assumption; it is not a model top-one match or proof
of experimental feasibility. Reagents are not counted as reactants. Two
reactant entries still require inference even when they are identical.
The designated product must still have a valid stereo-free InChIKey.

For other reactions, up to ten final distinct products are generated. Each parseable candidate is
canonicalized with chython using ``fix_tautomers=False``, then identified with
``inchi_key(molecule, ignore_stereo=True)``. The expected product is identified
with the same stereo-free InChIKey operation. Full returned keys are compared;
they are not truncated to the connectivity block. Isotope differences remain
distinguishable. Standard InChI normalization can identify some tautomeric forms
as equivalent, but this is not a general tautomer enumeration procedure.

Stereochemistry is ignored for prediction deduplication and matching: neither route chemistry nor model
inputs lose their stereo labels. An opposite or unspecified stereoisomer can
therefore match. This measures **stereo-free product recovery**, not
stereochemical correctness or experimental feasibility.

The matching distinct product at position ``k`` receives ``rank=k``, ``score=1/k``,
and ``alert=false``. Positions start at one. If no top-ten candidate matches,
``rank=null``, ``score=0.0`` and ``alert=true``. Invalid candidates have been removed and duplicates collapsed before ranking. An alert means only that the expected product was not recovered in
the first ten positions. It does not identify why the model missed it.

Model execution errors and failures to identify the expected product propagate
to the caller. They are not converted into zero scores.

.. code-block:: python

   from synplan.chem.reaction.routes.quality import ForwardReactionAssessor

   assessor = ForwardReactionAssessor(
       model, cache_path="results/forward_cache.json", batch_size=16
   )
   assessed_routes = assessor.assess(routes, silent=False)
   print(assessed_routes[0].steps[0].forward_prediction)
   print(assessor.last_stats)

``Step.forward_prediction`` is optional. Each annotated step contains only
``product_inchikey``, ``rank``, ``score`` and ``alert``; there is no ``reason``
field. ``Route.to_json()`` and ``Route.from_json()`` round-trip this dictionary
under reaction-node ``meta.forward_prediction``. Annotation never writes into
shared reaction containers. Existing Python route provenance remains attached
to the returned copies.

Cache and progress
------------------

The cache is an in-memory dictionary backed by a named JSON file retained after
the run. The script defaults to ``forward_cache.json`` beside its output;
``--cache`` selects an explicit reusable path across collections. The Python
API defaults to that filename in the current directory.

Entries are grouped by a SHA-256 fingerprint of model/backend, runtime asset
hashes, augmentation count and seed, beam width, requested result count, decoder
chunk size, sequence limits, pipeline/matching versions, identity/ranking rules,
runtime package versions and execution providers. Within that group, normalized
reactants identify an entry. Expected products are excluded: identical
reactants, even with different expected products, need one model prediction.
Each entry stores final canonical predictions, stereo-free InChIKeys,
representative sequence log likelihoods, aggregate ranking scores and actual
augmentation count. Pipeline version 2 cannot reuse historical single-input
beam-10 entries. Old fingerprint groups remain intact in the cache. Final step scores are not cached:
the one-reactant exception leaves existing multi-reactant predictions reusable.
Previously cached one-reactant entries are retained but ignored.

The JSON file loads once. The assessor collects unique inputs across all routes,
then predicts only cache misses. After each completed batch it writes the updated
dictionary to a temporary sibling, closes that file, and atomically replaces the
named cache. A restart reuses committed batches; an interrupted batch may be
recomputed. An orphan temporary file is ignored. A malformed named cache raises
an error rather than silently losing saved results.

This sequential workflow needs neither a database nor worker locks. JSON keeps
the cache inspectable and portable with the results. It rewrites the whole file
per batch and retains its contents in memory, so it is intended for collections
that fit in memory. There is no concurrent-writer support: use separate cache
paths for simultaneous processes.

The first tqdm bar counts routes while collecting reactions. The second counts
uncached model inputs and advances after each completed batch. The summary and
``last_stats`` report route count, reaction occurrences, unique inputs, cache
hits/misses, model predictions, predictions saved, matched/unmatched occurrences,
one-reactant steps, total non-zero scores and elapsed assessment time.
Matched/unmatched counts describe only model-assessed occurrences; assumed
one-reactant scores are reported separately. Hits/misses count unique model
inputs; cache savings compare actual inference inputs with model-assessed
occurrences, excluding the one-reactant bypass. Python calls are
silent by default; scripts show progress unless ``--quiet`` is passed.

Assess an exported collection
----------------------------

From the repository root in ``synplan_env``:

.. code-block:: bash

   python scripts/assess_routes.py \
       --model synplan_data/forward/forwardchimera-denovo/v1 \
       --model-type chimera --backend onnx \
       --input results/routes.json.gz \
       --output results/routes_forward.json.gz \
       --cache results/forward_cache.json

The input is the existing target-keyed mapping whose values are route-ID
mappings or route lists. JSON and JSON.GZ are supported. The script restores
``Route`` objects and overlays annotations onto a copy of the original JSON,
preserving target/route/child order and arbitrary existing metadata. The JSON
reader's optional ``on_step(reaction, source_node)`` callback supplies the
correspondence without guessing reaction identity or overwriting metadata.
The output is written atomically to a separate file; input, output and cache
paths must differ.


Route assessment
----------------

``ForwardRouteScorer`` computes the geometric mean over **all** reaction steps:

.. math::

   F = \left(\prod_{i=1}^{N} s_i\right)^{1/N}

Each entry in ``route.steps`` contributes once, including all branches of a
convergent route. Shared downstream steps are not counted repeatedly along
different leaf-to-target paths. Any zero-score step gives ``F=0.0``. For positive
scores, the implementation uses ``exp(fsum(log(s_i)) / N)`` to avoid underflow.

Assumed one-reactant scores of 1.0 are included in the denominator. Consequently
``[0.25, 1.0]`` scores 0.5, and a route consisting entirely of assumed steps scores
1.0. Such steps can increase the mean; the assessment reports their count
separately. These values are ranking heuristics, not calibrated probabilities
of experimental success or measures of stereochemical correctness.

.. code-block:: python

   from synplan.chem.reaction.routes.quality import ForwardRouteScorer

   scorer = ForwardRouteScorer()
   assessed_routes = scorer.assess(annotated_routes, silent=False)
   forward_score = scorer.score(annotated_routes[0])
   ranked_routes = scorer.rank(assessed_routes)  # explicit, stable descending sort

   combined = ForwardRouteScorer(combine_search=True)
   combined_score = combined.score(annotated_routes[0])

``assess()`` preserves input order and returns route copies with an optional
``Route.assessment`` dictionary. It records ``aggregation="geometric_mean_all_steps"``,
``forward_score``, ``search_score``, ``total_steps``, ``zero_score_steps`` and
``assumed_one_reactant_steps``. It does not mutate steps or shared chemistry.
Missing step annotations raise an error, even when another step already scores
zero. Run reaction assessment first; route scoring never initializes a model.

Combining is **disabled by default**. When enabled, ``combined_score`` is added:

.. math::

   C = S_{\mathrm{search}} \times F

The original search scale is retained without collection normalization.
Combination requires a finite, nonnegative recorded search score. Forward-only
assessment works without search provenance and records ``search_score=null``.
``score()`` and inherited ``rank()`` use the forward score by default, or the
combined score when explicitly enabled. Policy log likelihood remains a separate,
unchanged provenance value and is never used in this combination.

``Route.to_json()`` and ``Route.from_json()`` round-trip optional root-level
``assessment`` and ``provenance``. Legacy files without these fields remain
readable. Re-running reaction assessment clears an earlier route assessment;
run the route scorer again to refresh it after reaction annotations change.
Explicitly editing route chemistry likewise requires reassessment.

Score a saved collection and optionally create an offline HTML report:

.. code-block:: bash

   python scripts/score_routes.py \
       --input results/routes_forward.json.gz \
       --output results/routes_assessed.json.gz \
       --html results/routes_assessed.html

Add ``--combine-search`` to include the combined score, or ``--quiet`` to hide
the route-based tqdm bar. Input, output and HTML paths must differ. JSON output
is written atomically; route IDs, collection order and existing metadata are
preserved. The input must already contain forward annotations.

The existing route renderer displays assessment tables automatically when routes
carry these annotations. The report shows forward/search scores, policy log
likelihoods, reaction scores and molecule drawings. Filters select all routes,
routes without zero-score steps, or routes containing at least one zero.
Assumed scores are blue, model matches green, and zero scores red. The combined
column appears only when a combined score is present. Zoom and SVG/PNG export
remain available; no network resources are required.


Iteration and time budgets
--------------------------

The ordinary default remains 600 seconds. Set ``tree.max_time: null`` to
disable the wall-clock limit and compare searches at a fixed iteration budget:

.. code-block:: yaml

   tree:
     max_iterations: 100
     max_time: null
     max_depth: 6

Tree-size, depth, exhaustion and stop-at-first conditions still apply. All
search algorithms support a null limit. Search statistics include the stopping
reason, and elapsed search time is retained when expansion exhausts the tree.
