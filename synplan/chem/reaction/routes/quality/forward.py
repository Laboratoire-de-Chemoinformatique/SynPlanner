"""Stereo-free product recovery with batched inference and a reusable JSON cache."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from chython import inchi_key
from tqdm.auto import tqdm

from synplan.ml.inference.forward import PIPELINE_VERSION, canonical_reactants
from synplan.utils.json import read_json, write_json_atomic

# Bump whenever preprocessing, beam search, or matching semantics change.
ASSESSMENT_VERSION = 2
# Scoring changes do not invalidate cached predictions.
SCORING_VERSION = 2
BEAM_WIDTH = 20
NUM_RESULTS = 10


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(model):
    backend = model.backend
    return {
        "model": type(model).__name__,
        "backend": backend,
        "files": {
            name: _sha256(model.directory / name) for name in model.runtime_files
        },
        "beam_width": BEAM_WIDTH,
        "num_results": NUM_RESULTS,
        "augmentation_size": model.augmentation_size,
        "seed": model.seed,
        "decoder_batch_size": getattr(model, "decoder_batch_size", None),
        "max_length": model.max_length,
        "max_generated_length": getattr(
            model, "max_generated_length", model.max_length
        ),
        "assessment_version": ASSESSMENT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "identity": "full-standard-inchikey-ignore-stereo",
        "ranking": "clean-best-rank-then-reciprocal-support-stable-first",
        "runtime": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("synplanner", "numpy", "rdkit", "chython-synplan")
            },
            backend: __import__(
                "onnxruntime" if backend == "onnx" else "torch"
            ).__version__,
        },
        "providers": model.providers,
    }


class ForwardReactionAssessor:
    """Annotate copies of routes with reciprocal Top-K product-recovery ranks.

    ``cache_path`` names a retained JSON file, shared sequentially across runs.
    Expected products are deliberately excluded from prediction cache keys.
    One-reactant steps receive an assumed score of 1.0 without inference.
    ``last_stats`` describes the most recent successful assessment.
    """

    def __init__(
        self, model, cache_path="forward_cache.json", *, batch_size=16, top_k=10
    ):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        if not 1 <= top_k <= 10:
            raise ValueError("Top-K must be between 1 and 10")
        self.top_k = top_k
        self.model = model
        self.batch_size = batch_size
        self.cache_path = Path(cache_path)
        self.identity = _model_identity(model)
        self.fingerprint = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()
        self.cache = (
            read_json(self.cache_path)
            if self.cache_path.exists()
            else {"version": 1, "models": {}}
        )
        if self.cache["version"] != 1:
            raise ValueError("Unsupported forward cache version")
        self.entries = self.cache["models"].setdefault(
            self.fingerprint, {"identity": self.identity, "inputs": {}}
        )["inputs"]
        self.last_stats = {}

    def assess(self, routes, *, silent=True):
        """Return annotated copies in order, clearing stale route aggregates."""
        routes = list(routes)
        reactants, products = [], []
        with tqdm(
            total=len(routes), desc="Collecting reactions", unit="route", disable=silent
        ) as bar:
            for route in routes:
                for step in route.steps:
                    reactants.append(step.reaction.reactants)
                    products.append(step.product)
                bar.update()
        annotations = iter(self.assess_reactions(reactants, products, silent=silent))
        assessed = [
            replace(
                route,
                steps=tuple(
                    replace(step, forward_prediction=next(annotations))
                    for step in route.steps
                ),
                assessment=None,
            )
            for route in routes
        ]
        self.last_stats["routes"] = len(routes)
        if not silent:
            stats = self.last_stats
            tqdm.write(
                f"{len(routes)} routes; {stats['reaction_occurrences']} reactions; "
                f"{stats['unique_inputs']} unique inputs; "
                f"{stats['cache_hits']} cache hits, {stats['cache_misses']} misses; "
                f"{stats['nonzero_scored']} positive, {stats['unmatched']} zero; "
                f"{stats['elapsed_seconds']:.2f}s"
            )
        return assessed

    def assess_reactions(self, reactants, products, *, silent=True):
        """Assess aligned reactant collections and designated products.

        Expected products are matched separately from cached predictions. Top-K
        is an acceptance cutoff, not part of the prediction cache identity.
        """
        started = perf_counter()
        inputs, expected, unique = [], [], {}
        single_reactant_steps = 0
        for molecules, product in zip(reactants, products, strict=True):
            # The one-reactant scoring convention is documented in methods/forward.rst.
            if len(molecules) == 1:
                value = None
                single_reactant_steps += 1
            else:
                value = canonical_reactants(".".join(format(m, "") for m in molecules))
                unique.setdefault(value, None)
            key = inchi_key(product, ignore_stereo=True)
            if not key:
                raise ValueError("Could not identify the reaction's product")
            inputs.append(value)
            expected.append(key)
        inference_seconds = 0.0
        missing = [value for value in unique if value not in self.entries]
        with tqdm(
            total=len(missing),
            desc="Forward predictions",
            unit="reaction",
            disable=silent,
        ) as bar:
            for offset in range(0, len(missing), self.batch_size):
                batch = missing[offset : offset + self.batch_size]
                inference_started = perf_counter()
                rows = self.model.predict(
                    batch,
                    beam_width=BEAM_WIDTH,
                    batch_size=self.batch_size,
                    num_results=NUM_RESULTS,
                )
                inference_seconds += perf_counter() - inference_started
                completed = {}
                for reactants, row in zip(batch, rows, strict=True):
                    completed[reactants] = {
                        key: row[key]
                        for key in (
                            "predictions",
                            "canonical_products",
                            "product_inchikeys",
                            "log_likelihoods",
                            "ranking_scores",
                            "augmentation_count",
                        )
                    }
                self.entries.update(completed)
                write_json_atomic(self.cache_path, self.cache)
                bar.update(len(batch))

        annotations, matched = [], 0
        for value, product in zip(inputs, expected, strict=True):
            if value is None:
                rank, score, alert = None, 1.0, False
            else:
                keys = self.entries[value]["product_inchikeys"][: self.top_k]
                rank = next(
                    (i for i, key in enumerate(keys, 1) if key == product), None
                )
                score, alert = (1.0 / rank if rank else 0.0), rank is None
                matched += rank is not None
            annotations.append(
                {
                    "product_inchikey": product,
                    "rank": rank,
                    "score": score,
                    "alert": alert,
                }
            )
        occurrences = len(inputs)
        model_occurrences = occurrences - single_reactant_steps
        self.last_stats = {
            "scoring_version": SCORING_VERSION,
            "inference_seconds": inference_seconds,
            "reaction_occurrences": occurrences,
            "single_reactant_steps": single_reactant_steps,
            "model_assessed_occurrences": model_occurrences,
            "unique_inputs": len(unique),
            "cache_hits": len(unique) - len(missing),
            "cache_misses": len(missing),
            "model_predictions": len(missing),
            "predictions_saved": model_occurrences - len(missing),
            "matched": matched,
            "unmatched": model_occurrences - matched,
            "nonzero_scored": matched + single_reactant_steps,
            "elapsed_seconds": perf_counter() - started,
        }
        self.last_inputs = tuple(unique)
        return annotations
