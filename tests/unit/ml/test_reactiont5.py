"""T5 generation semantics without downloading a checkpoint."""

from types import SimpleNamespace

import numpy as np
import pytest

from synplan.ml.inference.beam_search import reactiont5 as _beam_search
from synplan.ml.inference.tokenizer import INPUT_TEMPLATE


def test_blank_reagents_template():
    assert (
        INPUT_TEMPLATE.format("C[C@H](O)F.[13CH4]")
        == "REACTANT:C[C@H](O)F.[13CH4]REAGENT: "
    )


def test_eos_termination_and_partial_batch():
    calls = []

    def decode(ids, mask, past, parents):
        calls.append((ids.copy(), parents))
        logits = np.full((len(ids), 4), -20.0, dtype=np.float32)
        logits[:, 1] = 0  # EOS is always best.
        return logits, SimpleNamespace()

    rows = _beam_search(decode, np.ones((3, 2), dtype=np.int64), 1, 8)
    assert len(calls) == 1
    assert len(rows) == 3
    assert all(row[0][2].tolist() == [0, 1] for row in rows)


def test_max_length_and_parent_reordering():
    calls = []

    def decode(ids, mask, past, parents):
        calls.append(parents)
        logits = np.full((len(ids), 5), -100.0, dtype=np.float32)
        logits[:, 2] = 0
        logits[:, 3] = -1
        return logits, SimpleNamespace()

    rows = _beam_search(decode, np.ones((2, 2), dtype=np.int64), 2, 5)
    assert len(calls) == 4
    np.testing.assert_array_equal(calls[1], [0, 0, 1, 1])
    assert all(len(row) == 2 and len(row[0][2]) == 5 for row in rows)
    for row in rows:
        assert row[0][0] == pytest.approx(row[0][1] / 4)
        assert row[0][0] >= row[1][0]


@pytest.mark.parametrize("beam", [1, 3])
def test_tiny_t5_matches_transformers_generate(beam):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from synplan.ml.inference._reactiont5_torch import DecoderGraph, EncoderGraph

    torch.manual_seed(13)
    torch.set_num_threads(1)
    config = transformers.T5Config(
        vocab_size=13,
        d_model=16,
        d_kv=8,
        d_ff=32,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        dropout_rate=0.0,
        feed_forward_proj="gated-gelu",
        tie_word_embeddings=False,
        decoder_start_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = transformers.T5ForConditionalGeneration(config).eval()
    encoder, decoder, cached = (
        EncoderGraph(model),
        DecoderGraph(model),
        DecoderGraph(model, cached=True),
    )
    ids = torch.tensor([[4, 5, 1], [6, 1, 0]])
    mask = (ids != 0).long()
    with torch.inference_mode():
        memory = encoder(ids, mask)

        def decode(tokens, attention, past, parents):
            with torch.inference_mode():
                tokens, attention = torch.tensor(tokens), torch.tensor(attention)
                out = (
                    decoder(tokens, attention, memory)
                    if past is None
                    else cached(tokens, attention, torch.tensor(parents), *past)
                )
                return out[0].numpy(), out[1:]

        rows = _beam_search(decode, mask.numpy(), beam, 9)
        expected = model.generate(
            input_ids=ids,
            attention_mask=mask,
            num_beams=beam,
            num_return_sequences=beam,
            max_length=9,
            return_dict_in_generate=True,
            output_scores=True,
        )
    actual = [row for group in rows for row in group]
    for row, sequence in zip(actual, expected.sequences, strict=True):
        tokens = sequence.tolist()
        if 1 in tokens:
            tokens = tokens[: tokens.index(1) + 1]
        assert row[2].tolist() == tokens
    if beam > 1:
        np.testing.assert_allclose(
            [row[0] for row in actual], expected.sequences_scores.numpy(), atol=2e-6
        )


@pytest.mark.parametrize("width", [0, 268])
def test_invalid_beam_width(width):
    from synplan.ml.inference import ForwardPredictor

    model = ForwardPredictor.__new__(ForwardPredictor)
    model.model_type = "reactiont5"
    model.metadata = {"architecture": {"vocab_size": 268}}
    with pytest.raises(ValueError, match="Beam width"):
        model.predict(["CC.O"], beam_width=width)
