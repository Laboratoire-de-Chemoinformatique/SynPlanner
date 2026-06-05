"""Expansion-wrapper contracts for MHN ranking policies."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import synplan.mcts.expansion as expansion
from synplan.chem.reaction_rules.graphs import RULE_GRAPH_SCHEMA_VERSION
from synplan.chem.reaction_rules.rule_fingerprints import (
    RULE_FINGERPRINT_SCHEMA_VERSION,
)


class _Rule:
    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


class _MHNPolicy(torch.nn.Module):
    architecture = "mhn_ranking"
    policy_type = "ranking"
    n_rules = 2
    mhn_rule_encoder_type = "fingerprint"
    mhn_rule_fp_size = 4
    mhn_rule_fp_min_radius = 1
    mhn_rule_fp_max_radius = 2
    mhn_rule_fp_active_bits = 2
    mhn_rule_fp_type = "query_cgr"
    mhn_rule_fp_schema_version = RULE_FINGERPRINT_SCHEMA_VERSION
    mhn_rule_embedder_type = "gps"
    mhn_rule_graph_batch_size = 2
    mhn_rule_graph_schema_version = RULE_GRAPH_SCHEMA_VERSION

    def encode_rules(self, rule_representations):
        return rule_representations + 1


class _MHNGraphPolicy(_MHNPolicy):
    mhn_rule_encoder_type = "query_cgr_graph"

    def encode_rules(self, rule_representations):
        return torch.ones((len(rule_representations), 4))


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
    return expansion.policy_network_function_from_config(
        SimpleNamespace(weights_path="policy.ckpt", priority_rules_fraction=0.5)
    )


def test_mhn_preparation_caches_encoded_rules(monkeypatch):
    calls = []

    def fake_rule_fingerprints(rule_smarts, fingerprint_config):
        calls.append((tuple(rule_smarts), fingerprint_config))
        return torch.zeros((len(rule_smarts), 4))

    monkeypatch.setattr(
        expansion, "rule_fingerprints_from_smarts", fake_rule_fingerprints
    )
    wrapper = _wrapper(monkeypatch, _MHNPolicy())
    rules = [_Rule("A"), _Rule("B")]

    wrapper._prepare_rule_associations(rules)
    first = wrapper._rule_associations
    wrapper._prepare_rule_associations(rules)

    assert len(calls) == 1
    assert calls[0][0] == ("A", "B")
    fingerprint_config = calls[0][1]
    assert fingerprint_config.fp_size == 4
    assert fingerprint_config.min_radius == 1
    assert fingerprint_config.max_radius == 2
    assert fingerprint_config.active_bits == 2
    assert fingerprint_config.fp_type == "query_cgr"
    assert fingerprint_config.schema_version == RULE_FINGERPRINT_SCHEMA_VERSION
    assert wrapper._rule_associations is first
    assert wrapper.n_rules == 2


def test_mhn_preparation_uses_query_cgr_rule_graphs(monkeypatch):
    calls = []

    def fake_rule_graphs(rule_smarts, *, schema_version):
        calls.append((tuple(rule_smarts), schema_version))
        return [SimpleNamespace(rule=text) for text in rule_smarts]

    monkeypatch.setattr(expansion, "query_cgr_graphs_from_smarts", fake_rule_graphs)
    wrapper = _wrapper(monkeypatch, _MHNGraphPolicy())
    rules = [_Rule("A"), _Rule("B")]

    wrapper._prepare_rule_associations(rules)
    first = wrapper._rule_associations
    wrapper._prepare_rule_associations(rules)

    assert calls == [(("A", "B"), RULE_GRAPH_SCHEMA_VERSION)]
    assert tuple(first.shape) == (2, 4)
    assert wrapper._rule_associations is first


def test_mhn_association_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(
        expansion,
        "rule_fingerprints_from_smarts",
        lambda rule_smarts, _config: torch.zeros((len(rule_smarts), 4)),
    )
    wrapper = _wrapper(monkeypatch, _MHNPolicy())

    for index in range(expansion._MAX_RULE_ASSOCIATION_CACHE_SIZE + 2):
        wrapper._prepare_rule_associations([_Rule(f"A{index}")])

    assert len(wrapper._rule_association_cache) == (
        expansion._MAX_RULE_ASSOCIATION_CACHE_SIZE
    )


def test_linear_wrapper_has_no_mhn_rule_preparation_state(monkeypatch):
    wrapper = _wrapper(monkeypatch, _LinearPolicy())

    assert not isinstance(wrapper, expansion.MHNPolicyNetworkFunction)
    assert not hasattr(wrapper, "_prepare_rule_associations")
    assert not hasattr(wrapper, "_rule_associations")


def test_direct_policy_wrapper_rejects_mhn_checkpoint():
    with pytest.raises(ValueError, match="policy_network_function_from_config"):
        expansion.PolicyNetworkFunction(
            SimpleNamespace(weights_path="policy.ckpt"), policy_net=_MHNPolicy()
        )


def test_mhn_light_prediction_requires_prepared_rule_associations(monkeypatch):
    wrapper = _wrapper(monkeypatch, _MHNPolicy())
    wrapper._get_graph = lambda _precursor: SimpleNamespace()

    with pytest.raises(ValueError, match="prepared by predict_reaction_rules"):
        list(wrapper.predict_reaction_rules_light(SimpleNamespace(), 2))


def test_light_prediction_uses_integer_count_without_preparing_rules(monkeypatch):
    wrapper = _wrapper(monkeypatch, _MHNPolicy())
    observed = []
    wrapper._predict_rules_common = lambda _precursor, n_rules: observed.append(n_rules)

    assert list(wrapper.predict_reaction_rules_light(SimpleNamespace(), 2)) == []
    assert wrapper._rule_associations is None
    assert observed == [2]


def test_combined_prediction_prepares_mhn_rules():
    prepared = []
    ranking = expansion.MHNPolicyNetworkFunction.__new__(
        expansion.MHNPolicyNetworkFunction
    )
    ranking._prepare_rule_associations = lambda rules: prepared.append(rules)
    combined = expansion.CombinedPolicyNetworkFunction.__new__(
        expansion.CombinedPolicyNetworkFunction
    )
    combined.ranking_net = ranking
    combined._predict_rules_common = lambda _precursor, _n_rules: None
    rules = [_Rule("A"), _Rule("B")]

    assert list(combined.predict_reaction_rules(SimpleNamespace(), rules)) == []
    assert prepared == [rules]


def test_combined_light_prediction_uses_integer_count_without_preparing_rules():
    prepared = []
    observed = []
    ranking = expansion.MHNPolicyNetworkFunction.__new__(
        expansion.MHNPolicyNetworkFunction
    )
    ranking._prepare_rule_associations = lambda rules: prepared.append(rules)
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
