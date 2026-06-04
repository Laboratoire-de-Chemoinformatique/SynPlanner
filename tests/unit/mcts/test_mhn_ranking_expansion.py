"""Expansion-wrapper contracts for MHN ranking policies."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import synplan.mcts.expansion as expansion


class _Rule:
    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


class _MHNPolicy(torch.nn.Module):
    architecture = "mhn_ranking"
    policy_type = "ranking"
    n_rules = 2
    mhn_template_fp_size = 4
    mhn_template_fp_min_radius = 1
    mhn_template_fp_max_radius = 2
    mhn_template_fp_active_bits = 2

    def encode_templates(self, features):
        return features + 1


class _LinearPolicy(torch.nn.Module):
    architecture = "linear"
    policy_type = "filtering"
    n_rules = 2


def _wrapper(monkeypatch, policy):
    monkeypatch.setattr(
        expansion,
        "load_policy_net",
        lambda *_args, **_kwargs: policy,
    )
    return expansion.PolicyNetworkFunction(
        SimpleNamespace(weights_path="policy.ckpt", priority_rules_fraction=0.5)
    )


def test_mhn_preparation_caches_encoded_templates(monkeypatch):
    calls = []

    def fake_features(rule_smarts, **_kwargs):
        calls.append(tuple(rule_smarts))
        return torch.zeros((len(rule_smarts), 4))

    monkeypatch.setattr(expansion, "template_features_from_smarts", fake_features)
    wrapper = _wrapper(monkeypatch, _MHNPolicy())
    rules = [_Rule("A"), _Rule("B")]

    wrapper._prepare_template_associations(rules)
    first = wrapper._template_associations
    wrapper._prepare_template_associations(rules)

    assert calls == [("A", "B")]
    assert wrapper._template_associations is first
    assert wrapper.n_rules == 2


def test_mhn_association_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(
        expansion,
        "template_features_from_smarts",
        lambda rule_smarts, **_kwargs: torch.zeros((len(rule_smarts), 4)),
    )
    wrapper = _wrapper(monkeypatch, _MHNPolicy())

    for index in range(expansion._MAX_TEMPLATE_ASSOCIATION_CACHE_SIZE + 2):
        wrapper._prepare_template_associations([_Rule(f"A{index}")])

    assert len(wrapper._template_association_cache) == (
        expansion._MAX_TEMPLATE_ASSOCIATION_CACHE_SIZE
    )


def test_linear_template_preparation_is_noop(monkeypatch):
    wrapper = _wrapper(monkeypatch, _LinearPolicy())

    wrapper._prepare_template_associations([_Rule("A"), _Rule("B")])

    assert wrapper._template_associations is None


def test_light_prediction_uses_integer_count_without_preparing_templates(monkeypatch):
    wrapper = _wrapper(monkeypatch, _MHNPolicy())
    observed = []
    wrapper._predict_rules_common = lambda _precursor, n_rules: observed.append(n_rules)

    assert list(wrapper.predict_reaction_rules_light(SimpleNamespace(), 2)) == []
    assert wrapper._template_associations is None
    assert observed == [2]


def test_combined_prediction_prepares_mhn_templates():
    prepared = []
    ranking = SimpleNamespace(
        _prepare_template_associations=lambda rules: prepared.append(rules)
    )
    combined = expansion.CombinedPolicyNetworkFunction.__new__(
        expansion.CombinedPolicyNetworkFunction
    )
    combined.ranking_net = ranking
    combined._predict_rules_common = lambda _precursor, _n_rules: None
    rules = [_Rule("A"), _Rule("B")]

    assert list(combined.predict_reaction_rules(SimpleNamespace(), rules)) == []
    assert prepared == [rules]


def test_combined_light_prediction_uses_integer_count_without_preparing_templates():
    prepared = []
    observed = []
    ranking = SimpleNamespace(
        _prepare_template_associations=lambda rules: prepared.append(rules)
    )
    combined = expansion.CombinedPolicyNetworkFunction.__new__(
        expansion.CombinedPolicyNetworkFunction
    )
    combined.ranking_net = ranking
    combined._predict_rules_common = lambda _precursor, n_rules: observed.append(
        n_rules
    )

    assert list(combined.predict_reaction_rules_light(SimpleNamespace(), 2)) == []
    assert prepared == []
    assert observed == [2]
