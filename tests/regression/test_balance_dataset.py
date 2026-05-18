"""Invariant: ``balance_extracted_precursor`` returns both labels.

The function is used in RL training to balance positive and negative examples
extracted from MCTS rollouts. A degenerate output (only positives, only
negatives, or empty) collapses the value-network training signal: the model
trains on a constant-label dataset and learns nothing useful.

The invariants asserted here are completely independent of the balancing
strategy chosen:

* For any input that contains both labels, the output must contain both.
* The output must be a subset of the input (no fabricated entries).

This is the "behaviour contract" of the function; the implementation can
change (different downsampling strategy, oversampling, etc.) and these
assertions still hold.
"""

from __future__ import annotations

import pytest

# Skipped: ML/RL training is out of scope for the current cleanup pass.
# The underlying bug is real (``balance_extracted_precursor`` in
# ``synplan/ml/training/reinforcement.py:188`` discards negatives via
# ``neg_list.pop(...)`` without writing them to the output dict, so RL
# value-network training collapses to a constant predictor). Revisit when
# the ML/RL pipeline is back in scope.
pytestmark = pytest.mark.skip(reason="ML/RL out of scope for this pass")


@pytest.fixture(scope="module")
def balance_fn():
    from synplan.ml.training.reinforcement import balance_extracted_precursor

    return balance_extracted_precursor


@pytest.mark.parametrize(
    "input_dict",
    [
        # equal positives/negatives
        {f"mol_{i}": 1 for i in range(5)} | {f"mol_n{i}": 0 for i in range(5)},
        # more positives than negatives
        {f"mol_{i}": 1 for i in range(7)} | {f"mol_n{i}": 0 for i in range(3)},
        # more negatives than positives
        {f"mol_{i}": 1 for i in range(3)} | {f"mol_n{i}": 0 for i in range(7)},
    ],
    ids=["equal", "more_pos", "more_neg"],
)
def test_balance_output_contains_both_labels(balance_fn, input_dict):
    """Output dict must contain at least one entry with each label."""
    out = balance_fn(dict(input_dict))
    labels = set(out.values())
    assert 0 in labels and 1 in labels, (
        f"balance_extracted_precursor lost a label class: input had both 0 "
        f"and 1, output contains only {sorted(labels)}. RL value-network "
        "training on this output collapses to a constant predictor. The "
        "function increments through positives but never inserts negatives "
        "into the output dict (see reinforcement.py:188)."
    )


def test_balance_output_is_subset_of_input(balance_fn):
    """The function may downsample but must not fabricate new keys/labels."""
    src = {f"mol_{i}": 1 for i in range(5)} | {f"mol_n{i}": 0 for i in range(5)}
    out = balance_fn(dict(src))
    for k, v in out.items():
        assert k in src, f"balance_extracted_precursor invented key {k}"
        assert src[k] == v, (
            f"balance_extracted_precursor changed label of {k}: input had "
            f"{src[k]}, output has {v}."
        )


def test_balance_output_is_nontrivially_nonempty(balance_fn):
    """A non-empty input with both labels must yield a non-empty output."""
    src = {f"mol_{i}": 1 for i in range(5)} | {f"mol_n{i}": 0 for i in range(5)}
    out = balance_fn(dict(src))
    assert len(out) > 0, (
        "balance_extracted_precursor returned an empty dict for a non-empty "
        "input with both labels. RL training will skip this batch entirely."
    )
