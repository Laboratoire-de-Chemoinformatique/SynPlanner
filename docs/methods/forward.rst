.. _forward_prediction:

Forward reaction assessment
===========================

Forward prediction estimates candidate products from reactant SMILES.

Models and shared prediction policy
-----------------------------------

``OnnxChemformer`` lives in ``synplan.ml.inference.chemformer``. Its directory
contains ``encoder.onnx``, ``decoder.onnx``, ``vocabulary.json`` and ``model.json``.
The compact SynFwd-1 bundle is at ``synplan_data/forward/synfwd-1/v1/``.
The model directory is ignored by Git. A single graph is insufficient.
The adapter uses existing base dependencies and imports neither PyTorch nor
executable code from the model directory. Sessions load once. CPU defaults to four threads. ``device="cuda"`` requires an
independently installed ONNX Runtime CUDA provider and its CUDA/cuDNN libraries;
provider initialization failures raise instead of silently falling back.

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

   chemformer = OnnxChemformer("synplan_data/forward/synfwd-1/v1", threads=4)
   predictions = chemformer.predict(["CC(=O)Cl.N"], num_results=10,
                                    beam_width=20, batch_size=16)

``beam_width`` controls decoding per augmentation, independently of the final
result count. The internal ``_predict_batch`` decoders remain available for raw
parity tests; they receive already prepared spellings and retain raw beams.
