"""Small exported graphs exercise cached decoding in CI without model assets."""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from synplan.ml.inference._chemformer_torch import (  # noqa: E402
    CachedDecoderGraph,
    ChemformerDecoder,
    InitialDecoderGraph,
)
from synplan.ml.inference.chemformer import OnnxChemformer  # noqa: E402


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    torch.manual_seed(73)
    config = {
        "d_model": 16,
        "num_heads": 2,
        "d_feedforward": 32,
        "num_layers": 2,
        "vocabulary_size": 7,
        "max_seq_len": 8,
        "pad_token_idx": 0,
        "activation": "gelu",
    }
    model = ChemformerDecoder(config).eval()
    with torch.no_grad():
        model.pos_emb.copy_(torch.randn_like(model.pos_emb))
    directory = tmp_path_factory.mktemp("chemformer")
    new = directory

    class Encoder(torch.nn.Module):
        def forward(self, input_ids, padding_mask):
            return input_ids[..., None].float().expand(-1, -1, 16) * (
                ~padding_mask[..., None]
            )

    class FullDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, decoder_input_ids, memory, memory_padding_mask):
            value = self.model.emb(decoder_input_ids) * math.sqrt(model.width)
            value = value + self.model.pos_emb[: decoder_input_ids.shape[1]][None]
            length = decoder_input_ids.shape[1]
            causal = torch.triu(
                torch.ones(length, length, dtype=torch.bool), diagonal=1
            )
            value = self.model.decoder(
                value,
                memory,
                tgt_mask=causal,
                memory_key_padding_mask=memory_padding_mask,
            )
            return torch.log_softmax(self.model.token_fc(value[:, -1]), dim=-1)

    def export(module, args, path, inputs, outputs, axes):
        torch.onnx.export(
            module.eval(),
            args,
            str(path),
            input_names=inputs,
            output_names=outputs,
            dynamic_axes=axes,
            opset_version=17,
            dynamo=False,
        )

    ids, mask = (
        torch.ones((2, 3), dtype=torch.long),
        torch.zeros((2, 3), dtype=torch.bool),
    )
    memory = Encoder()(ids, mask)
    start = torch.ones((2, 1), dtype=torch.long)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    export(
        Encoder(),
        (ids, mask),
        new / "encoder.onnx",
        ["input_ids", "padding_mask"],
        ["memory"],
        {
            n: {0: "batch", 1: "source_length"}
            for n in ("input_ids", "padding_mask", "memory")
        },
    )
    names = [
        f"{i}_{k}"
        for i in range(2)
        for k in ("key", "value", "cross_key", "cross_value")
    ]
    present, past = ["present_" + n for n in names], ["past_" + n for n in names]
    axes = {
        "input_ids": {0: "batch"},
        "memory": {0: "batch", 1: "source_length"},
        "padding_mask": {0: "batch", 1: "source_length"},
        "log_probs": {0: "batch"},
    }
    axes.update(
        {
            n: {
                0: "batch",
                2: "source_length" if "cross" in n else "present_length",
            }
            for n in present
        }
    )
    export(
        InitialDecoderGraph(model),
        (start, memory, mask),
        new / "decoder.onnx",
        ["input_ids", "memory", "padding_mask"],
        ["log_probs", *present],
        axes,
    )
    with torch.inference_mode():
        state = model.decode(start, mask, 0, memory=memory)[1:]
    del axes["memory"]
    axes["beam_indices"] = {0: "batch"}
    axes.update(
        {
            n: {
                0: "past_batch",
                2: "source_length" if "cross" in n else "past_length",
            }
            for n in past
        }
    )
    export(
        CachedDecoderGraph(model),
        (start, mask, torch.tensor(1), torch.tensor([1, 0]), *state),
        new / "decoder_with_past.onnx",
        ["input_ids", "padding_mask", "position", "beam_indices", *past],
        ["log_probs", *present],
        axes,
    )
    (new / "model.json").write_text(
        json.dumps({"architecture": config, "kv_cache": True})
    )
    (new / "vocabulary.json").write_text(
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
    yield new, FullDecoder().eval()
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_exported_cached_sampler_matches_full_prefix(bundles, chunk, monkeypatch):
    path, reference = bundles
    old = OnnxChemformer(path, threads=1, decoder_batch_size=chunk)
    new = OnnxChemformer(path, threads=1, decoder_batch_size=chunk)
    old.kv_cache = False

    def full_prefix(prefix, memory, mask):
        with torch.inference_mode():
            return reference(
                torch.from_numpy(prefix),
                torch.from_numpy(memory),
                torch.from_numpy(mask),
            ).numpy()

    monkeypatch.setattr(old, "_decode", full_prefix)
    for width in (1, 5):
        expected = old._predict_batch(["CO", "N", "CCC"], width)
        actual = new._predict_batch(["CO", "N", "CCC"], width)
        assert [r["predictions"] for r in actual] == [
            r["predictions"] for r in expected
        ]
        np.testing.assert_allclose(
            [r["log_likelihoods"] for r in actual],
            [r["log_likelihoods"] for r in expected],
            atol=1e-5,
            rtol=1e-5,
        )
