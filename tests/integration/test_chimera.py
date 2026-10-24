"""Explicit local assets only; no downloads and no upstream runtime dependency."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from synplan.ml.inference.chimera import ForwardChimera


@pytest.fixture(scope="module")
def directory():
    value = os.environ.get("SYNPLAN_CHIMERA_MODEL_DIR")
    if not value:
        pytest.skip("Set SYNPLAN_CHIMERA_MODEL_DIR to a local exported bundle")
    return Path(value).resolve()


@pytest.fixture(scope="module")
def model(directory):
    return ForwardChimera(directory)


def test_bundle_manifest(directory):
    files = json.loads((directory / "manifest.json").read_text())["files"]
    assert set(files) == {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    for name, entry in files.items():
        path = directory / name
        with path.open("rb") as stream:
            checksum = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
            digest = checksum.hexdigest()
        assert entry == {"sha256": digest, "bytes": path.stat().st_size}


@pytest.mark.parametrize("backend", ["onnx", "ckpt"])
def test_inference_without_training_dependencies(directory, backend):
    if backend == "ckpt":
        pytest.importorskip("torch")
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {'retrochimera', 'pytorch_lightning', 'lightning', 'syntheseus'}
        if sys.argv[2] == 'onnx': blocked.add('torch')
        if fullname.split('.')[0] in blocked: raise ImportError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from synplan.ml.inference.chimera import ForwardChimera
result = ForwardChimera(sys.argv[1], backend=sys.argv[2]).predict(['CC(=O)Cl.N'])[0]
assert 1 <= len(result['predictions']) <= 10
assert len(set(result['product_inchikeys'])) == len(result['predictions'])
"""
    subprocess.run(
        [sys.executable, "-c", script, str(directory), backend],
        check=True,
        capture_output=True,
        text=True,
    )


def test_batching_and_order(model):
    inputs = ["CC(=O)Cl.N", "CCO.CC(=O)O", "CC(=O)Cl.CN"]
    expected = model.predict(inputs, batch_size=1)
    actual = model.predict(inputs, batch_size=2)
    reversed_rows = model.predict(list(reversed(inputs)), batch_size=2)
    assert (
        [r["predictions"] for r in expected]
        == [r["predictions"] for r in actual]
        == [r["predictions"] for r in reversed(reversed_rows)]
    )
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_allclose(
            left["log_likelihoods"], right["log_likelihoods"], atol=2e-4, rtol=1e-4
        )


def test_checkpoint_graphs_and_reordered_kv(model, directory):
    pytest.importorskip("torch")
    native = ForwardChimera(directory, backend="ckpt")
    ids, mask = model.tokenizer.encode(["CC(=O)Cl.N", "CCO.CC(=O)O"])
    memory, reference = model._encode(ids, mask), native._encode(ids, mask)
    np.testing.assert_allclose(
        memory.numpy(), reference.cpu().numpy(), atol=3e-5, rtol=3e-5
    )
    ids = np.zeros((2, 1), dtype=np.int64)
    previous_a = previous_b = parents = None
    for position in range(4):
        actual, previous_a = model._decode(
            ids, mask, position, memory, previous_a, parents
        )
        expected, previous_b = native._decode(
            ids, mask, position, reference, previous_b, parents
        )
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=3e-5)
        parents = (
            np.array([1, 0, 0]) if position == 0 else np.arange(len(ids))[::-1].copy()
        )
        ids = actual.argmax(-1)[parents, None].astype(np.int64)
        mask = mask[parents]
    for width in (1, 10, 20):
        values = ["CC(=O)Cl.N", "CCO.CC(=O)O"]
        actual, expected = (
            model._predict_batch(values, width),
            native._predict_batch(values, width),
        )
        assert [r["predictions"] for r in actual] == [
            r["predictions"] for r in expected
        ]
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                left["log_likelihoods"], right["log_likelihoods"], atol=2e-4, rtol=1e-4
            )
