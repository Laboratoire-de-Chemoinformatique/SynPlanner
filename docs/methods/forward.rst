.. _forward_prediction:

Forward reaction assessment
===========================

Forward assessment measures whether a model recovers the designated product of
each reaction in a completed route. It annotates copies of ``Route`` objects in
their original order. A separate ``ForwardRouteScorer`` aggregates these step
annotations without inference. Neither assessment changes the original search
score, policy likelihood, protection/competing-sites scores, or route order.

For a worked Chemformer example, see
:doc:`../user_guide/20_Forward_Prediction`: plan Lenvatinib routes, assess the saved
collection, aggregate scores, export JSON and HTML, and reopen the prediction cache.

Models and shared prediction policy
-----------------------------------

All three models use ``ForwardPredictor`` from ``synplan.ml.inference``, with
explicit ``model_type="chemformer"``, ``"chimera"`` or ``"reactiont5"`` selection.
The default is Chemformer with the ONNX backend. Existing bundle formats and
cache identities are unchanged by this unified interface.

For Chemformer, the directory contains ``encoder.onnx``, ``decoder.onnx``,
``vocabulary.json`` and ``model.json``.
The original SynFwd-1 bundle is at ``synplan_data/forward/synfwd-1/v1/``.
The cached ``v2/`` export additionally contains ``decoder_with_past.onnx``;
``kv_cache: true`` in its metadata selects incremental decoding automatically.
For Chimera, the directory is
``synplan_data/forward/forwardchimera-denovo/v1/`` and contains ``encoder.onnx``,
``decoder.onnx``, ``decoder_with_past.onnx``, ``model.json`` and
``source/vocab.txt``. Both directories are ignored by Git. A single graph is
insufficient. Original exports, ZIP, checkpoint and attribution are retained.

Chemformer and Chimera ONNX inference use the existing base dependencies,
without importing PyTorch, RetroChimera, Lightning, Syntheseus or code from the
asset directory.
Sessions load once. CPU defaults to four threads. ``device="cuda"`` requires an
independently installed ONNX Runtime CUDA provider and its CUDA/cuDNN libraries;
provider initialization failures raise instead of silently falling back.

Chimera additionally accepts explicit ``backend="ckpt"``. This needs optional
PyTorch (``SynPlanner[forward-checkpoint]``), but neither RetroChimera nor
Lightning. Export uses the same compact architecture with optional PyTorch and
ONNX tooling (``SynPlanner[forward-export]``). The default remains ONNX.
Chemformer supports ONNX only.

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

   from synplan.ml.inference import ForwardPredictor

   chemformer = ForwardPredictor("synplan_data/forward/synfwd-1/v2", threads=4)
   chimera = ForwardPredictor("synplan_data/forward/forwardchimera-denovo/v1",
                              model_type="chimera")
   predictions = chimera.predict(["CC(=O)Cl.N"], num_results=10,
                                 beam_width=20, batch_size=1)
   checkpoint = ForwardPredictor("synplan_data/forward/forwardchimera-denovo/v1",
                                 model_type="chimera", backend="ckpt")

``beam_width`` controls decoding per augmentation, independently of the final
result count. The internal ``_predict_batch`` decoders remain available for raw
parity tests; they receive already prepared spellings and retain raw beams.

Implementation and attribution
------------------------------

One ``Tokenizer(model_dir, model_type, max_length)`` class supplies batch
``encode()`` and ``decode()`` methods. Encoding returns integer token IDs and a
boolean mask where ``True`` means padding. The runtime converts this mask to
ReactionT5's integer attention mask. Chemformer adds encoder start/end tokens;
Chimera does not; ReactionT5 uses its exported tokenizer and blank-reagent
input template. Tokenization never canonicalizes or truncates augmented inputs.

Loading and ONNX execution are shared. Beam search retains the three models'
EOS, length normalization and termination rules. Changing those rules would
change predictions. The ``_..._torch.py`` modules are internal architecture/export
implementations; their leading underscore is a naming convention, not access
control. Conditional imports keep PyTorch and Transformers optional.

The Chemformer-derived code adapts MolecularAI/Chemformer,
MolecularAI/aizynthmodels and chemformer-workflows under Apache-2.0. Chimera
inference and augmentation adapt Microsoft RetroChimera v1.3.0 and OpenNMT-py
3.5.1 under MIT. ReactionT5v2 is by Tatsuya Sagawa et al. under MIT; its beam
semantics follow Transformers 4.40.2. The distributed ``LICENSE`` contains the
full third-party terms and attribution, separately scoped from SynPlanner's
MIT terms. Adapted source files retain short attribution and modification notices.

Chemformer KV caching on CPU
---------------------------

The original decoder recalculates the complete generated prefix at every token.
The cached export processes only the latest token, retaining each decoder
layer's self-attention keys/values and the static cross-attention projections of
the reactants. The encoder is unchanged. Initial decoding creates the cache;
``decoder_with_past.onnx`` gathers the selected beam parents and appends the
new token's state. Positional offsets, beam selection, finished-beam handling,
and final end-token scoring follow the original Chemformer sampler.

This changes execution, not the prediction policy: the model weights, up to ten
augmentations, twenty beams, maximum sequence length, stereo-free identity rules,
and final product ranking are unchanged. Floating-point kernel differences can
affect near-tied beams, so cached exports require numerical and ranked-prediction
parity checks. Existing two-graph bundles remain supported.

Create a separate bundle using the optional ``SynPlanner[forward-export]`` tools:

.. code-block:: console

   python -m scripts.export_chemformer \
       --source synplan_data/forward/synfwd-1/v1 \
       --checkpoint /path/to/best.ckpt \
       --output synplan_data/forward/synfwd-1/v2

The exporter verifies the checkpoint SHA-256 against the original metadata,
copies the encoder and attribution, and regenerates the manifest. The output
directory must be new. Production inference imports neither PyTorch nor the
Chemformer training checkout and requires no additional dependencies. Point the
existing assessment script's ``--model`` or planning ``forward_val_path`` at the
new directory to use it.

KV states live only for the current prediction. The retained JSON reaction cache
still avoids repeated predictions across routes. New graph hashes automatically
create a separate cache identity, preserving historical predictions. Keep the
original bundle and caches for reproducible comparisons.

``threads=4`` and ``decoder_batch_size=64`` retain their existing defaults. Cached
decoding respects the beam-row chunk limit; chunk outputs are joined between
steps. Single-chunk states remain ONNX Runtime values. CPU chunk joins allocate
arrays, so benchmark peak memory as well as latency: state size grows with beam
count and generated length. CPU benchmarks should use the same inputs, thread
count and prediction settings, exclude loading, disable the reaction-result
cache, and measure both implementations sequentially.

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
The same selected convention applies to Chimera and ReactionT5; it does not
imply either was trained with the same exclusion. This score records that
assumption; it is not a model top-one match or proof of experimental feasibility.
Reagents are not counted as reactants. Two
reactant entries still require inference even when they are identical.
The designated product must still have a valid stereo-free InChIKey.
Reports display these steps as ordinary positive scores of 1.0, using the same
styling as model-assessed positive steps. Their Top-k rank is shown as a dash
because no model prediction was run. There is no separate ``Assumed`` label,
legend or column in either the overview or per-route tables.

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
uncached model inputs and advances after each completed batch. The console summary
reports routes, reaction occurrences, unique inputs, cache hits/misses, positive
and zero scores, and elapsed assessment time. Positive counts include one-reactant
scores. ``last_stats`` additionally retains model predictions, predictions saved,
matched/unmatched model-assessed occurrences and one-reactant counts for auditing.
Hits/misses count unique model inputs; cache savings compare actual inference
inputs with model-assessed occurrences, excluding the one-reactant bypass.
Python calls are silent by default; scripts show progress unless ``--quiet``
is passed.

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
preserving target/route/child order and arbitrary existing metadata. The script
tracks source nodes through temporary metadata in a parsing copy and removes
that bookkeeping before assessment; the route and JSON reader APIs need no hooks.
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
Positive scores are green, including one-reactant steps, and zero scores red.
The combined column appears only when a combined score is present. Zoom and
SVG/PNG export remain available; no network resources are required.

Optional validation during planning
-----------------------------------

The final top-level entry of ``configs/planning_standard.yaml`` is optional:

.. code-block:: yaml

   # After tree, node_expansion, and other planning sections.
   forward_val_config: null

Omitting it or leaving it null runs ordinary planning, without loading a forward
model or opening a prediction cache. To enable it, set
``forward_val_config: forward_predictor.yaml``. This reference is relative to the
planning YAML, not the shell's working directory. The separate file contains:

.. code-block:: yaml

   forward_val_type: on-the-fly  # null, on-the-fly, post-processing
   forward_val_path: ../synplan_data/forward/synfwd-1/v2/
   top_k: 10
   device: cpu
   threads: 4
   batch_size: 16
   cache_path: null

``forward_val_type: null`` also disables assessment. An enabled configuration
requires a Chemformer ONNX bundle directory. Model and explicit cache paths are
relative to this forward YAML; a null cache path uses ``forward_cache.json`` in
the planning results directory. CUDA is explicit and never falls back silently.
An explicitly supplied unreadable or invalid configuration raises an error.

``top_k`` is an acceptance cutoff from 1 to 10. Prediction generation still uses
up to ten augmentations, twenty beams per augmentation, and up to ten distinct
stereo-free product identities. The complete prediction list is cached, so
changing Top-K does not invalidate predictions or require new inference. Missing
candidates are never padded. A product outside the selected Top-K receives the
same zero annotation as an absent product.

**On-the-fly** checks each otherwise eligible disconnection before creating its
child or registering a reusable search state. All newly generated precursors,
including stock molecules and small fragments, are supplied as forward reactants;
the expanded precursor is the expected product. Unrelated pending branches are
not part of that reaction. A missing Top-K match rejects the candidate. Passing
candidates retain their ordinary policy probabilities and search-value updates;
reciprocal rank is recorded separately. Single-reactant steps retain the assumed
1.0 convention. Filtering applies to policy and priority rules and to terminal
children. It does not validate temporary reactions inside the rollout evaluator.

Checks run sequentially, reusing the adapter's augmentation batching. One model
instance and JSON cache are shared across targets. Identical normalized reactants
reuse predictions even when their expected products differ. Model execution and
expected-product identification failures propagate as errors, never as chemical
rejections. Validation currently supports retrosynthetic search only.

**Post-processing** saves completed search output before assessing the completed
routes. It keeps all routes, including failures, and preserves their order. Both
modes calculate the existing geometric forward route score after search; combined
scoring remains opt-in through the route-scoring interface. Annotations and scores
survive JSON, HTML, mapped export, and search-record round trips. Search records
contain data only: opening one or calling ``routes()`` never loads a model.

Reaction annotations belong to incoming search edges; aggregate scores belong
only to ``Route.assessment``. Finalization retains assessed Route objects for
later exports. Search records store their aggregates in a separate
``route_assessments`` section, without duplicating route chemistry. Older
node-level aggregates are restored onto Route objects when reading legacy records.

The planning CLI handles loading and finalization. Low-level Python callers can
share an assessor explicitly:

.. code-block:: python

   from synplan.mcts.forward import load_forward_validation

   config, assessor = load_forward_validation("configs/forward_predictor.yaml", "results")
   # Include the usual target, rules, stock, policy and evaluator arguments.
   tree = Tree(..., forward_assessor=(
       assessor if config and config.forward_val_type == "on-the-fly" else None
   ))
   tree.run()
   routes = (tree.finalize_forward(assessor, config.forward_val_type, silent=False)
             if config else tree.routes())

The planning runner also accepts ``forward_val_config=None``, a YAML path, or a
``ForwardValidationConfig``. A path supplied directly through this Python API is
relative to the current working directory. The same disabled semantics apply.

Iteration and time budgets
~~~~~~~~~~~~~~~~~~~~~~~~~

The ordinary default remains 600 seconds. For an iteration-controlled comparison,
explicitly disable the wall-clock cutoff:

.. code-block:: yaml

   tree:
     max_iterations: 100
     max_time: null
     max_depth: 6

Tree-size, depth, exhaustion and stop-at-first conditions still apply. Filtering
changes the explored tree, so equal iteration counts do not imply equal numbers
of generated or accepted candidates. Search wall time includes on-the-fly
inference and cache work. Initialization and post-processing are reported
separately. Finite deadlines are checked between candidate predictions: an active
inference call may finish after the deadline, but no new prediction starts then.

``forward_validation.json`` records resolved settings, model identity,
initialization time, per-target search statistics, and forward statistics.
These include checks, model passes/rejections, assumed passes, unique inputs,
cache hits/misses, and inference time. A shared cache makes wall time dependent on
its initial contents; comparisons should record those contents and cache usage.

Routes surviving on-the-fly filtering pass this model by construction. Their pass
rate is not independent validation or experimental accuracy. Stereo-free product
recovery does not establish stereochemical correctness, conditions, or feasibility.

Chemformer checkpoint loading and competing-sites integration remain separate
subtasks.

ReactionT5v2 USPTO-MIT
---------------------

``ForwardPredictor(..., model_type="reactiont5")`` loads the ``sagawa/ReactionT5v2-forward-USPTO_MIT`` checkpoint as a
post-processing alternative. It uses the standard Hugging Face T5 architecture:
12 encoder and 12 decoder layers, width 768, 12 attention heads, gated GELU
feed-forward layers, and a 268-token vocabulary. It has **198,640,896 learned
parameters**, compared with **44,676,619** for the supplied Chemformer SynFwd-1
checkpoint and **76,169,047** for ForwardChimeraDeNovo. These counts exclude the
fixed sinusoidal position buffers in Chemformer and Chimera.

The release is pinned to ``260cf4aa7e6f55b1a9db1e328f48701cd3fd7cfc``. Its
provenance is T5 v1.1 base, CompoundT5 pretraining, ORD forward training, then
USPTO-MIT fine-tuning. The upstream model card's introduction says
``USPOT_50k``, whereas its fine-tuning section and repository name specify
USPTO-MIT; the bundle retains the original card and records this discrepancy.
The generic ``ReactionT5v2-forward`` checkpoint is not used.

Install the small tokenizer dependency for ONNX inference:

.. code-block:: bash

   uv pip install -e '.[reactiont5]'

Checkpoint inference additionally needs ``SynPlanner[reactiont5-checkpoint]``;
export needs ``SynPlanner[reactiont5-export]``. Transformers 4.40.2 is pinned to
match the upstream checkpoint and generation reference. ONNX inference imports
neither PyTorch nor Transformers. The other forward adapters do not require
these extras. The adapter does not depend on the ReactionT5 checkout, Lightning,
Syntheseus or RetroChimera.

Prepare the portable directory once (optional export dependencies required):

.. code-block:: bash

   python -m scripts.export_reactiont5 --download \
     --model synplan_data/forward/reactiont5v2-uspto-mit/v1

The bundle retains the original safetensors checkpoint, configuration, tokenizer,
release metadata and MIT attribution, alongside ``encoder.onnx``,
``decoder.onnx`` and ``decoder_with_past.onnx``. ``model.json`` lists the required
runtime graph files, including any external tensor data. ``manifest.json`` records
SHA-256 hashes; the download also verifies the released weight checksum. No
executable code is loaded from this directory. Large assets remain outside Git.

.. code-block:: python

   from synplan.ml.inference import ForwardPredictor

   model = ForwardPredictor("synplan_data/forward/reactiont5v2-uspto-mit/v1",
                            model_type="reactiont5")
   predictions = model.predict(["CC(=O)Cl.Nc1ccccc1"], num_results=10)
   # Explicit optional checkpoint inference:
   checkpoint = ForwardPredictor("synplan_data/forward/reactiont5v2-uspto-mit/v1",
                                 model_type="reactiont5", backend="ckpt")

Both backends default to CPU with four threads. CUDA must be requested explicitly
and must initialize successfully. The ONNX decoder retains K/V states on the
execution device, reordering them within its graph when beams change parents.

Every augmented spelling is formatted exactly as
``REACTANT:<reactant SMILES>REAGENT: `` with a trailing space and **blank
reagents**. Route reagent metadata is not supplied. Atom mapping is removed
through chython formatting; stereochemistry and isotopes are preserved in the
input. The released Unigram tokenizer is used directly, without recanonicalizing
augmented spellings. Decoded outputs follow the reference convention of removing
spaces and a trailing dot before shared chemical processing.

Generation uses the reference prediction script's explicit limits: 400 input
tokens, and 300 output tokens including the decoder start token. The latter
explicitly overrides the 150-token default in the released generation config.
Overlong inputs raise an error instead of being truncated. Beam hypotheses use
length penalty 1.0 and the reference's default heuristic stopping; raw generation
is verified against Transformers. Representative ``log_likelihoods`` are sums of
token log probabilities, distinct from length-normalized raw beam scores and
shared augmentation ``ranking_scores``.

The shared ten-augmentation, twenty-beam, ten-distinct-product pipeline and
stereo-free full-InChIKey matching apply unchanged. Invalid candidate structures
are discarded, including a chython 1.108 normalization error for invalid valence;
unexpected errors and failures to identify the expected product still propagate.
Single-reactant steps retain the documented 1.0 convention. Route aggregation and
optional search-score combination are unchanged. Tokenizer version, blank-reagent
format, generation settings, backend and asset hashes enter the cache identity.
Chemistry lives in ``synplan.chem``; each injected model supplies its inference
identity, so the assessor does not import the ML package.

.. code-block:: bash

   python -m scripts.assess_routes --model-type reactiont5 --backend onnx \
     --model synplan_data/forward/reactiont5v2-uspto-mit/v1 \
     --input results/routes.json.gz --output results/reactiont5_forward.json.gz \
     --cache results/reactiont5_cache.json
   python -m scripts.score_routes --input results/reactiont5_forward.json.gz \
     --output results/reactiont5_scored.json.gz --html results/reactiont5.html

ReactionT5 selection is currently available through this post-processing script
and the Python adapter. Planning YAML continues to use Chemformer. Comparison
reports should distinguish this shared augmented policy from historical raw
Chemformer beam-10 results, and report loading, cold inference, and warm-cache
time separately. All scores describe model product recovery, not experimental
accuracy or probabilities of synthetic success.
