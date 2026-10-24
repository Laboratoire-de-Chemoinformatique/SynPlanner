"""Forward inference contracts without model weights or Torch."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from synplan.ml.inference.chemformer import (
    OnnxChemformer,
    _Tokenizer,
    canonical_reactants,
)


@pytest.fixture
def vocabulary(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(
        json.dumps(
            {
                "vocabulary": ["<pad>", "^", "&", "?", "C", "N", "O"],
                "properties": {
                    "special_tokens": {
                        "pad": "<pad>",
                        "start": "^",
                        "end": "&",
                        "unknown": "?",
                    },
                    "regex": "[A-Z]",
                },
            }
        )
    )
    return path


@pytest.fixture
def model(vocabulary, monkeypatch):
    """A two-candidate decoder with known sequence scores."""
    (vocabulary.parent / "model.json").write_text(
        json.dumps({"architecture": {"max_seq_len": 6}})
    )

    class Session:
        def __init__(self, path, **kwargs):
            self.encoder = Path(path).stem == "encoder"
            self.calls = []
            self.finish = True
            self.empty_candidates = False

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, outputs, inputs):
            self.calls.append(inputs)
            if self.encoder:
                return [inputs["input_ids"][..., None].astype(np.float32)]
            prefix = inputs["decoder_input_ids"]
            values = np.full((len(prefix), 7), -100.0, dtype=np.float32)
            if prefix.shape[1] > 1 and self.finish:
                values[:, 2] = -0.3  # end
            elif self.empty_candidates:
                values[:, 2] = -0.1  # end
                values[:, 0] = -0.2  # pad
            else:
                values[:, 4] = -0.1  # C
                values[:, 5] = -0.2  # N
            return [values]

    monkeypatch.setattr(ort, "InferenceSession", Session)
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    return OnnxChemformer(vocabulary.parent, decoder_batch_size=3)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("[NH3:5].[CH3:1][C:2](=[O:3])[Cl:4]", "CC(=O)Cl.N"),
        ("N.CC(=O)Cl", "CC(=O)Cl.N"),
        ("[13CH3:1][OH:2].[13CH3:3][OH:4]", "[13CH3]O.[13CH3]O"),
        ("[CH2:1]=[CH:2][OH:3]", "C=CO"),  # keep the enol
        ("N[C@@H:2](C)C(=O)O", "C[C@H](N)C(=O)O"),
    ],
)
def test_preprocessing_preserves_chemistry_without_mapping(value, expected):
    assert canonical_reactants(value) == expected


@pytest.mark.parametrize("value", ["", "CCO>>CC=O"])
def test_preprocessing_requires_reactants_only(value):
    with pytest.raises(ValueError, match="reactants-only"):
        canonical_reactants(value)


def test_tokenizer_padding_and_length_limit(vocabulary):
    tokenizer = _Tokenizer(vocabulary, max_length=5)
    ids, mask = tokenizer.encode(["C", "CON"])
    np.testing.assert_array_equal(ids, [[1, 4, 2, 0, 0], [1, 4, 6, 5, 2]])
    np.testing.assert_array_equal(
        mask, [[False, False, False, True, True], [False] * 5]
    )
    with pytest.raises(ValueError, match="maximum is 5"):
        tokenizer.encode(["CCCC"])
    with pytest.raises(ValueError, match="tokenized completely"):
        tokenizer.encode(["C!"])


def test_batching_and_decoder_chunking_preserve_order_and_scores(model):
    encoder, decoder = model.encoder, model.decoder
    result = model.predict(["C", "N", "O"], beam_width=2, batch_size=2)
    assert [row["reactants"] for row in result] == ["C", "N", "O"]
    assert [row["predictions"] for row in result] == [["C", "N"]] * 3
    np.testing.assert_allclose(
        [row["log_likelihoods"] for row in result],
        [[-0.4, -0.5]] * 3,
        atol=1e-6,
    )
    assert [len(call["input_ids"]) for call in encoder.calls] == [2, 1]
    assert [len(call["decoder_input_ids"]) for call in decoder.calls] == [2, 3, 1, 1, 2]
    assert max(call["decoder_input_ids"].shape[1] for call in decoder.calls) == 2
    assert model.encoder is encoder and model.decoder is decoder


def test_finished_beams_and_duplicate_invalid_outputs_are_retained(model):
    model.decoder.empty_candidates = True
    result = model._predict_batch(["C"], beam_width=2)[0]
    assert result["predictions"] == ["", ""]
    # The first beam ended immediately: later decoder calls must not penalize it.
    np.testing.assert_allclose(result["log_likelihoods"], [-0.1, -0.5], atol=1e-6)


def test_maximum_generation_length_forces_end_and_scores_it(model):
    model.decoder.finish = False
    result = model._predict_batch(["C"], beam_width=1)[0]
    assert result["predictions"] == ["CCCC"]
    np.testing.assert_allclose(result["log_likelihoods"], [-100.4], atol=1e-5)
    assert [call["decoder_input_ids"].shape[1] for call in model.decoder.calls] == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_empty_batch_does_not_run_sessions(model):
    assert model.predict([], beam_width=2) == []
    assert model.encoder.calls == model.decoder.calls == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"beam_width": 0}, "Beam width"),
        ({"beam_width": 2, "batch_size": 0}, "Batch size"),
    ],
)
def test_invalid_generation_settings(model, options, message):
    with pytest.raises(ValueError, match=message):
        model.predict(["C"], **options)


def test_cuda_unavailable_is_not_silently_replaced_by_cpu(model):
    with pytest.raises(RuntimeError, match="provider is unavailable"):
        OnnxChemformer(model.directory, device="cuda")


def test_cuda_initialization_fallback_is_rejected(model, monkeypatch):
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    with pytest.raises(RuntimeError, match="provider failed to initialize"):
        OnnxChemformer(model.directory, device="cuda")
