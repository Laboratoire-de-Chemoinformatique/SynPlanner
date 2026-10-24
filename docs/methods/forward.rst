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
