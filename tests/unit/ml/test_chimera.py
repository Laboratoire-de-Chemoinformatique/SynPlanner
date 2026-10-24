"""Token boundaries and finished-hypothesis beam bookkeeping without weights."""

import numpy as np
import pytest

from synplan.ml.inference.chimera import ForwardChimera, _Tokenizer


def tokenizer(tmp_path, max_length=12):
    path = tmp_path / "vocab.txt"
    path.write_text("<BOS>\n<EOS>\n<PAD>\n<UNK>\nC\nN\nO\n")
    return _Tokenizer(path, max_length)


def test_tokenizer_and_length(tmp_path):
    model = tokenizer(tmp_path, max_length=2)
    ids, mask = model.encode(["CN", "O"])
    np.testing.assert_array_equal(ids, [[4, 5], [6, 2]])
    np.testing.assert_array_equal(mask, [[False, False], [False, True]])
    assert model.decode([4, 1, 5]) == "C"
    assert model.decode([0, 4, 1, 5]) == "C"
    assert model.decode([4, 0, 1]) == "C<BOS>"
    with pytest.raises(ValueError, match="maximum is 2"):
        model.encode(["CCC"])
    with pytest.raises(ValueError, match="tokenized completely"):
        model.encode(["C!"])


def test_finished_hypotheses_survive_pruning_and_kv_parents(tmp_path):
    model = object.__new__(ForwardChimera)
    model.tokenizer = tokenizer(tmp_path)
    model.max_generated_length = 10
    model._encode = lambda ids, mask: None
    parents_seen = []

    def decode(ids, mask, position, memory=None, past=None, parents=None):
        parents_seen.append(None if parents is None else parents.tolist())
        result = np.full((len(ids), 7), -100.0, dtype=np.float32)
        if position == 0:
            result[:, 4], result[:, 5] = 0.0, -0.01
        elif position == 1:
            result[ids[:, 0] == 4, 1] = -0.1
            result[ids[:, 0] == 5, 6] = 0.0
        elif position == 2:
            result[:, 4], result[:, 5] = 0.0, -0.01
        else:
            result[:, 1] = -0.5
        return result, [position]

    model._decode = decode
    row = model._predict_batch(["C"], 2)[0]
    assert row["predictions"] == ["C", "NOC"]
    np.testing.assert_allclose(row["log_likelihoods"], [-0.1, -0.51], atol=1e-6)
    assert parents_seen == [None, [0, 0], [1, 0], [0, 0]]
    # Token limit returns active hypotheses without padding the candidate list.
    model.max_generated_length = 1
    assert model._predict_batch(["C"], 2)[0]["predictions"] == ["C", "N"]
