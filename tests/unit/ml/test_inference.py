"""Unified reader selection, lazy dependencies, and tokenizer batch contracts."""

import json
import subprocess
import sys

import numpy as np
import pytest

from synplan.ml.inference import ForwardPredictor, Tokenizer


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"model_type": "unknown"}, "Choose chemformer"),
        ({"model_type": "chemformer", "backend": "ckpt"}, "ONNX backend"),
        ({"device": "tpu"}, "Choose cpu/cuda"),
        ({"threads": 0}, "positive"),
        ({"decoder_batch_size": 0}, "positive"),
        ({"augmentation_size": 11}, "1-10"),
    ],
)
def test_selection_errors_before_opening_assets(tmp_path, options, message):
    with pytest.raises(ValueError, match=message):
        ForwardPredictor(tmp_path, **options)


def test_public_import_needs_no_optional_runtimes():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'tokenizers', 'onnxruntime',
            'retrochimera', 'lightning', 'pytorch_lightning',
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from synplan.ml.inference import ForwardPredictor, Tokenizer
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("model_type", ["chemformer", "chimera"])
def test_regex_tokenizer_batch_decode_and_unknown_tokens(tmp_path, model_type):
    tokens = ["<BOS>", "<EOS>", "<PAD>", "<UNK>", "C", "N"]
    if model_type == "chemformer":
        (tmp_path / "vocabulary.json").write_text(
            json.dumps(
                {
                    "vocabulary": tokens,
                    "properties": {
                        "regex": "[A-Z]",
                        "special_tokens": {
                            "start": "<BOS>",
                            "end": "<EOS>",
                            "pad": "<PAD>",
                            "unknown": "<UNK>",
                        },
                    },
                }
            )
        )
    else:
        (tmp_path / "source").mkdir()
        (tmp_path / "source/vocab.txt").write_text("\n".join(tokens))
    tokenizer = Tokenizer(tmp_path, model_type, 6)
    ids, mask = tokenizer.encode(["CN", "O"])
    assert ids.dtype == np.int64 and mask.dtype == np.bool_
    expected = (
        [[0, 4, 5, 1], [0, 3, 1, 2]] if model_type == "chemformer" else [[4, 5], [3, 2]]
    )
    np.testing.assert_array_equal(ids, expected)
    assert tokenizer.decode([[0, 4, 1, 5], [0, 3, 1]]) == ["C", "<UNK>"]
    assert tokenizer.decode([]) == []


def test_t5_padding_mask_is_based_on_length(tmp_path):
    native = pytest.importorskip("tokenizers")
    (tmp_path / "source").mkdir()
    tokenizer = native.Tokenizer(
        native.models.WordLevel(
            {
                "<pad>": 0,
                "</s>": 1,
                "<unk>": 2,
                "REACTANT": 3,
                ":": 4,
                "CCO": 5,
                "REAGENT": 6,
                "C": 7,
                ".": 8,
                "N": 9,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = native.pre_tokenizers.Whitespace()
    tokenizer.post_processor = native.processors.TemplateProcessing(
        single="$A </s>",
        special_tokens=[("</s>", 1)],
    )
    tokenizer.save(str(tmp_path / "source/tokenizer.json"))
    unified = Tokenizer(tmp_path, "reactiont5", 30)
    values = ["CCO", "C.N"]
    ids, mask = unified.encode(values)
    expected = tokenizer.encode_batch([f"REACTANT:{s}REAGENT: " for s in values])
    assert mask.dtype == np.bool_
    for row, padding, original in zip(ids, mask, expected, strict=True):
        assert row[~padding].tolist() == original.ids
        assert np.all(row[padding] == 0)
    expected_decode = [
        tokenizer.decode(row.ids).replace(" ", "").rstrip(".") for row in expected
    ]
    assert unified.decode([row.ids for row in expected]) == expected_decode
