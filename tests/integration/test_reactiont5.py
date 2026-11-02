"""Opt-in real bundle tests: SYNPLAN_REACTIONT5_MODEL_DIR=/path/to/v1 pytest ..."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from synplan.chem.utils import canonical_reactants
from synplan.ml.inference.forward import augment
from synplan.ml.inference.reactiont5 import INPUT_TEMPLATE, ReactionT5


@pytest.fixture(scope="module")
def directory():
    value = os.environ.get("SYNPLAN_REACTIONT5_MODEL_DIR")
    if not value:
        pytest.skip("Supply SYNPLAN_REACTIONT5_MODEL_DIR for real model tests")
    return Path(value)


@pytest.fixture(scope="module")
def model(directory):
    return ReactionT5(directory)


def test_tokenizer_and_limits(directory, model):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        directory / "source", local_files_only=True
    )
    values = ["C[C@H](O)F.[13CH4]", "CCO.CC(=O)O"]
    ids, mask = model.tokenizer.encode(values)
    expected = tokenizer(
        [INPUT_TEMPLATE.format(s) for s in values], padding=True, return_tensors="np"
    )
    np.testing.assert_array_equal(ids, expected["input_ids"])
    np.testing.assert_array_equal(mask, expected["attention_mask"])
    with pytest.raises(ValueError, match="No truncation"):
        model.tokenizer.encode(["C" * 2000])


def test_batch_order_and_augmentation(model):
    values = ["CC(=O)Cl.Nc1ccccc1", "CCO.CC(=O)O", "CC(=O)Cl.Nc1ccccc1"]
    # Raw batches exercise the neural cache; shared augmentation has separate unit coverage.
    together = model._predict_batch(values, 3)
    apart = [model._predict_batch([value], 3)[0] for value in values]
    for a, b in zip(together, apart, strict=True):
        assert a["predictions"] == b["predictions"]
        np.testing.assert_allclose(
            a["log_likelihoods"], b["log_likelihoods"], rtol=1e-4, atol=5e-4
        )
    mapped = canonical_reactants("[CH3:1][CH2:2]O.CC(=O)O")
    assert augment(mapped) == augment(canonical_reactants(values[1]))


@pytest.mark.parametrize("backend", ["onnx", "ckpt"])
def test_backend_dependency_boundary(directory, backend):
    if backend == "ckpt":
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {'retrochimera', 'pytorch_lightning', 'lightning', 'syntheseus'}
        if sys.argv[2] == 'onnx': blocked.update({'torch', 'transformers'})
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Forbidden backend import: ' + fullname)
sys.meta_path.insert(0, Block())
from synplan.ml.inference.reactiont5 import ReactionT5
m = ReactionT5(sys.argv[1], backend=sys.argv[2], augmentation_size=1)
r = m.predict(['CCO.CC(=O)O'], beam_width=1)
assert r[0]['predictions']
"""
    subprocess.run(
        [sys.executable, "-c", script, str(directory), backend], check=True, timeout=180
    )


def test_checkpoint_logits_beams_and_cached_reordering(directory, model):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    checkpoint = ReactionT5(directory, backend="ckpt")
    inputs = ["CC(=O)Cl.Nc1ccccc1", "CCO.CC(=O)O"]
    ids, mask = model.tokenizer.encode(inputs)
    memory, reference = model._encode(ids, mask), checkpoint._encode(ids, mask)
    np.testing.assert_allclose(memory.numpy(), reference.numpy(), atol=2e-5, rtol=1e-4)
    tokens = np.zeros((2, 1), dtype=np.int64)
    past = reference_past = parents = None
    for position in range(4):
        actual, past = model._decode(tokens, mask, memory, past, parents)
        expected, reference_past = checkpoint._decode(
            tokens, mask, reference, reference_past, parents
        )
        np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=1e-4)
        parents = (
            np.array([1, 0, 1])
            if position == 0
            else np.arange(len(tokens))[::-1].copy()
        )
        tokens = actual.argmax(-1)[parents, None].astype(np.int64)
        mask = mask[parents]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        directory / "source", local_files_only=True
    )
    encoded = tokenizer(
        [INPUT_TEMPLATE.format(value) for value in inputs],
        padding=True,
        return_tensors="pt",
    )
    for width in (1, 10, 20):
        actual, expected = (
            model._predict_batch(inputs, width),
            checkpoint._predict_batch(inputs, width),
        )
        with torch.inference_mode():
            generated = checkpoint.network.generate(
                **encoded,
                num_beams=width,
                num_return_sequences=width,
                max_length=300,
                min_length=1,
                length_penalty=1.0,
                early_stopping=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        assert [checkpoint.tokenizer.decode(row) for row in generated.sequences] == [
            value for row in actual for value in row["predictions"]
        ]
        for a, b in zip(actual, expected, strict=True):
            assert a["predictions"] == b["predictions"]
            np.testing.assert_allclose(
                a["log_likelihoods"], b["log_likelihoods"], atol=5e-4, rtol=1e-4
            )
        if width > 1:
            np.testing.assert_allclose(
                [score for row in actual for score in row["sequence_scores"]],
                generated.sequences_scores.numpy(),
                atol=3e-5,
                rtol=1e-4,
            )


def test_manifest(directory):
    import hashlib
    import json

    files = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))[
        "files"
    ]
    assert set(files) == {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    for name, entry in files.items():
        path = directory / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        assert path.stat().st_size == entry["bytes"]
        assert digest.hexdigest() == entry["sha256"]


def test_augmented_batching_and_partial_final_batch(model):
    inputs = ["CC(=O)Cl.Nc1ccccc1", "CCO.CC(=O)O", "CC(=O)Cl.Nc1ccccc1"]
    separate = model.predict(inputs, batch_size=1)
    together = model.predict(inputs, batch_size=16)
    assert [row["reactants"] for row in together] == [
        row["reactants"] for row in separate
    ]
    for left, right in zip(separate, together, strict=True):
        assert left["predictions"] == right["predictions"]
        assert left["product_inchikeys"] == right["product_inchikeys"]
        assert left["ranking_scores"] == right["ranking_scores"]
        np.testing.assert_allclose(
            left["log_likelihoods"], right["log_likelihoods"], rtol=1e-4, atol=5e-4
        )
