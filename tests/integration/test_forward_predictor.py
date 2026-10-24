"""Opt-in real-model checks.

Set SYNPLAN_FORWARD_MODEL_DIR to a local SynFwd-1 bundle. For parity checks,
also set SYNPLAN_FORWARD_REFERENCE_DIR to the original bundle with predict.py.
No model is downloaded by these tests.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from synplan.ml.inference.chemformer import OnnxChemformer, canonical_reactants

INPUTS = ["CC(=O)Cl.N", "CCO.CC(=O)O", "CC(=O)Cl.CN"]
RUNTIME_FILES = ("encoder.onnx", "decoder.onnx", "vocabulary.json", "model.json")


@pytest.fixture(scope="module")
def model_dir():
    path = os.environ.get("SYNPLAN_FORWARD_MODEL_DIR")
    if not path:
        pytest.skip("Set SYNPLAN_FORWARD_MODEL_DIR to run real-model checks")
    return Path(path).resolve()


@pytest.fixture(scope="module")
def model(model_dir):
    return OnnxChemformer(model_dir, decoder_batch_size=7)


@pytest.fixture(scope="module")
def reference():
    path = os.environ.get("SYNPLAN_FORWARD_REFERENCE_DIR")
    if not path:
        pytest.skip("Set SYNPLAN_FORWARD_REFERENCE_DIR to compare the original runtime")
    spec = importlib.util.spec_from_file_location(
        "synfwd_reference", Path(path) / "predict.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OnnxChemformer(path)


@pytest.mark.parametrize("width", [1, 10])
def test_original_wrapper_parity_and_partial_batch(model, reference, width):
    expected = reference.predict(INPUTS, beam_width=width)
    inputs = [canonical_reactants(value) for value in INPUTS]
    actual = model._predict_batch(inputs[:2], width) + model._predict_batch(
        inputs[2:], width
    )
    assert [row["reactants"] for row in actual] == [
        row["reactants"] for row in expected
    ]
    assert [row["predictions"] for row in actual] == [
        row["predictions"] for row in expected
    ]
    np.testing.assert_allclose(
        [row["log_likelihoods"] for row in actual],
        [row["log_likelihoods"] for row in expected],
        rtol=1e-4,
        atol=1e-4,
    )


def test_single_and_batched_predictions_agree(model):
    expected = model.predict(INPUTS, beam_width=10, batch_size=1)
    actual = model.predict(INPUTS, beam_width=10, batch_size=2)
    assert [row["predictions"] for row in actual] == [
        row["predictions"] for row in expected
    ]
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            left["log_likelihoods"], right["log_likelihoods"], rtol=1e-4, atol=1e-4
        )


def test_mapped_input_matches_unmapped(model):
    actual = model.predict(["[NH3:5].[CH3:1][C:2](=[O:3])[Cl:4]", INPUTS[0]])
    assert actual[0] == actual[1]


def test_four_file_bundle_runs_without_torch_or_remote_code(model_dir, tmp_path):
    for name in RUNTIME_FILES:
        (tmp_path / name).symlink_to(model_dir / name)
    script = """
import importlib.abc
import sys

class BlockTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'pytorch_lightning', 'aizynthmodels', 'chemformer_workflows'
        }:
            raise ImportError('Unexpected dependency: ' + fullname)

sys.meta_path.insert(0, BlockTrainingDependencies())
from synplan.ml.inference.chemformer import OnnxChemformer
rows = OnnxChemformer(sys.argv[1]).predict(['CC(=O)Cl.N'])
assert 1 <= len(rows[0]['predictions']) <= 10
assert rows[0]['product_inchikeys']
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_compact_bundle_manifest(model_dir):
    expected = {*RUNTIME_FILES, "README.md", "LICENSE", "NOTICE.md", "manifest.json"}
    assert {p.name for p in model_dir.iterdir()} == expected
    manifest = json.loads((model_dir / "manifest.json").read_text())["files"]
    assert set(manifest) == expected - {"manifest.json"}
    for name, entry in manifest.items():
        path = model_dir / name
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
