"""Opt-in cached/full-prefix parity using local bundles, without downloading.

Set SYNPLAN_FORWARD_MODEL_DIR to the original bundle and
SYNPLAN_FORWARD_CACHED_MODEL_DIR to the cached bundle. All inference is CPU.
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from synplan.ml.inference import ForwardPredictor

INPUTS = ["CC(=O)Cl.N", "CCO.CC(=O)O", "C[C@H](N)C(=O)O.CC(=O)Cl"]


@pytest.fixture(scope="module")
def models():
    paths = [
        os.environ.get(name)
        for name in ("SYNPLAN_FORWARD_MODEL_DIR", "SYNPLAN_FORWARD_CACHED_MODEL_DIR")
    ]
    if not all(paths):
        pytest.skip("Set both original and cached Chemformer bundle directories")
    # Deliberately split beam siblings across decoder chunks.
    return tuple(ForwardPredictor(path, decoder_batch_size=7) for path in paths)


def compare(left, right):
    assert [r["reactants"] for r in left] == [r["reactants"] for r in right]
    assert [r["predictions"] for r in left] == [r["predictions"] for r in right]
    for a, b in zip(left, right, strict=True):
        np.testing.assert_allclose(
            a["log_likelihoods"], b["log_likelihoods"], rtol=1e-4, atol=2e-4
        )


@pytest.mark.parametrize("width", [1, 10, 20])
def test_raw_beams_and_partial_batch(models, width):
    original, cached = models
    expected = original._predict_batch(INPUTS, width)
    actual = cached._predict_batch(INPUTS[:2], width) + cached._predict_batch(
        INPUTS[2:], width
    )
    compare(actual, expected)


def test_augmented_predictions_and_batch_order(models):
    original, cached = models
    expected = original.predict(INPUTS, batch_size=2)
    actual = cached.predict(INPUTS, batch_size=2)
    compare(actual, expected)
    compare(actual, cached.predict(INPUTS, batch_size=1))
    for a, b in zip(actual, expected, strict=True):
        assert a["product_inchikeys"] == b["product_inchikeys"]
        assert a["ranking_scores"] == b["ranking_scores"]


def test_cached_logits_parent_reordering_padding_and_position_limit(models):
    original, cached = models
    ids, mask = cached.tokenizer.encode(INPUTS[:2])
    memory = cached.encoder.run(None, {"input_ids": ids, "padding_mask": mask})[0]
    rng = np.random.default_rng(73)
    prefix = np.full((2, 1), cached.tokenizer.special["start"], dtype=np.int64)
    past, parents = None, None
    for position in range(cached.max_length - 1):
        actual, past = cached._decode_cached(prefix, memory, mask, past, parents)
        if position in (0, 1, 7, 31, 126, cached.max_length - 2):
            expected = original._decode_full_prefix(prefix, memory, mask)
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
        parents = (
            np.array([1, 0, 1], dtype=np.int64)
            if position == 0
            else np.array([2, 2, 0], dtype=np.int64)
        )
        memory, mask = memory[parents], mask[parents]
        prefix = np.concatenate(
            (
                prefix[parents],
                rng.integers(6, len(cached.tokenizer.tokens), size=(3, 1)),
            ),
            axis=1,
        )


def test_cached_bundle_manifest_and_separate_identity(models):
    original, cached = models
    assert not original.kv_cache and cached.kv_cache
    assert original.prediction_identity() != cached.prediction_identity()
    assert (
        hashlib.sha256((original.directory / "encoder.onnx").read_bytes()).digest()
        == hashlib.sha256((cached.directory / "encoder.onnx").read_bytes()).digest()
    )
    entries = json.loads((cached.directory / "manifest.json").read_text())["files"]
    assert set(entries) == {
        str(p.relative_to(cached.directory))
        for p in cached.directory.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    for name, entry in entries.items():
        content = (cached.directory / name).read_bytes()
        assert len(content) == entry["bytes"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]


def test_cached_bundle_needs_no_training_packages(models, tmp_path):
    cached = models[1]
    for name in cached.runtime_files:
        (tmp_path / name).symlink_to((cached.directory / name).resolve())
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'pytorch_lightning', 'aizynthmodels', 'chemformer_workflows', 'retrochimera'}:
            raise ImportError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from synplan.ml.inference import ForwardPredictor
model = ForwardPredictor(sys.argv[1])
assert model.kv_cache
assert model.predict(['CC(=O)Cl.N'])[0]['product_inchikeys']
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
