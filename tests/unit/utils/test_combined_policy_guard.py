"""The combined policy adds two heads' logits, so their rule sets must match."""

import pytest

from synplan.utils import loading


class _FakePolicy:
    def __init__(self, n_rules: int) -> None:
        self.n_rules = n_rules


@pytest.fixture
def mismatched_heads(monkeypatch):
    """synplanner-gps pairs a 24094-rule filtering head with an 11235-rule ranking head."""
    monkeypatch.setattr(
        loading,
        "build_policy_from_config",
        lambda cfg: _FakePolicy(24094 if cfg.policy_type == "filtering" else 11235),
    )


def test_mismatched_rule_sets_fail_at_load(mismatched_heads):
    with pytest.raises(ValueError, match="same rule set"):
        loading.load_combined_policy_function(
            filtering_weights_path="f.ckpt", ranking_weights_path="r.ckpt"
        )


def test_matched_rule_sets_build(monkeypatch):
    monkeypatch.setattr(
        loading, "build_policy_from_config", lambda cfg: _FakePolicy(11235)
    )
    policy = loading.load_combined_policy_function(
        filtering_weights_path="f.ckpt", ranking_weights_path="r.ckpt"
    )
    assert policy.n_rules == 11235
