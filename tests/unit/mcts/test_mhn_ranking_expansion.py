"""Expansion-wrapper contracts for MHN ranking policies."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import synplan.mcts.policy.template_based as template_based
from synplan.chem.reaction.rules.representation import (
    RULE_FINGERPRINT_SCHEMA_VERSION,
    RULE_GRAPH_SCHEMA_VERSION,
    RuleFingerprintConfig,
    RuleRepresentationConfig,
)
from synplan.mcts.policy.composite import CompositePolicy
from synplan.mcts.policy.template_based import (
    _MAX_RULE_ASSOCIATION_CACHE_SIZE,
    LinearPolicy,
    MHNReactPolicy,
)


class _Rule:
    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


class _MHNPolicy(nn.Module):
    architecture = "mhn_ranking"
    policy_type = "ranking"
    n_rules = 2
    rule_representation_config = RuleRepresentationConfig(
        embedding_type="fingerprint",
        fingerprint_config=RuleFingerprintConfig(
            fp_size=4,
            min_radius=1,
            max_radius=2,
            active_bits=2,
            fp_type="query_cgr",
            schema_version=RULE_FINGERPRINT_SCHEMA_VERSION,
        ),
        graph_embedder_type="gps",
        graph_batch_size=2,
        graph_schema_version=RULE_GRAPH_SCHEMA_VERSION,
    )

    def encode_rules(self, rule_representations):
        return rule_representations + 1


class _MHNGraphPolicy(_MHNPolicy):
    rule_representation_config = RuleRepresentationConfig(
        embedding_type="query_cgr_graph",
        fingerprint_config=RuleFingerprintConfig(
            fp_size=4,
            min_radius=1,
            max_radius=2,
            active_bits=2,
            fp_type="query_cgr",
            schema_version=RULE_FINGERPRINT_SCHEMA_VERSION,
        ),
        graph_embedder_type="gps",
        graph_batch_size=2,
        graph_schema_version=RULE_GRAPH_SCHEMA_VERSION,
    )

    def encode_rules(self, rule_representations):
        return torch.ones((len(rule_representations), 4))


class _MHNRDKitPolicy(_MHNPolicy):
    rule_representation_config = RuleRepresentationConfig(
        embedding_type="fingerprint",
        fingerprint_config=RuleFingerprintConfig(
            fp_size=4,
            min_radius=1,
            max_radius=2,
            active_bits=2,
            fp_type="mhnreact_rdkit",
            schema_version=RULE_FINGERPRINT_SCHEMA_VERSION,
        ),
        graph_embedder_type="gps",
        graph_batch_size=2,
        graph_schema_version=RULE_GRAPH_SCHEMA_VERSION,
    )


class _LinearPolicy(nn.Module):
    architecture = "linear"
    policy_type = "filtering"
    n_rules = 2


def _mhn_wrapper(policy):
    return MHNReactPolicy(policy, top_rules=50, priority_rules_fraction=0.5)


def _linear_wrapper(policy):
    return LinearPolicy(policy, top_rules=50, priority_rules_fraction=0.5)


def test_mhn_preparation_caches_encoded_rules(monkeypatch):
    calls = []
    smarts_calls = []

    def fake_rule_smarts(rules):
        smarts = tuple(str(rule) for rule in rules)
        smarts_calls.append(smarts)
        return smarts

    def fake_rule_fingerprints(rule_smarts, fingerprint_config):
        calls.append((tuple(rule_smarts), fingerprint_config))
        return torch.zeros((len(rule_smarts), 4))

    monkeypatch.setattr(template_based, "rule_smarts_from_reactors", fake_rule_smarts)
    monkeypatch.setattr(
        template_based, "rule_fingerprints_from_smarts", fake_rule_fingerprints
    )
    wrapper = _mhn_wrapper(_MHNPolicy())
    rules = [_Rule("A"), _Rule("B")]

    wrapper.prepare_rule_associations(rules)
    first = wrapper._rule_associations
    wrapper.prepare_rule_associations(rules)

    assert smarts_calls == [("A", "B")]
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


def test_mhn_preparation_rebinds_new_rule_sequence(monkeypatch):
    calls = []

    def fake_rule_fingerprints(rule_smarts, _fingerprint_config):
        calls.append(tuple(rule_smarts))
        return torch.full((len(rule_smarts), 4), float(len(calls)))

    monkeypatch.setattr(
        template_based, "rule_fingerprints_from_smarts", fake_rule_fingerprints
    )
    wrapper = _mhn_wrapper(_MHNPolicy())
    first_rules = [_Rule("A"), _Rule("B")]
    second_rules = [_Rule("C"), _Rule("D")]

    wrapper.prepare_rule_associations(first_rules)
    first = wrapper._rule_associations
    wrapper.prepare_rule_associations(second_rules)

    assert calls == [("A", "B"), ("C", "D")]
    assert wrapper._rule_associations is not first
    assert wrapper._bound_reaction_rules is second_rules
    assert wrapper._bound_reaction_rules_len == len(second_rules)


def test_mhn_preparation_passes_mhnreact_rdkit_fingerprint_config(monkeypatch):
    calls = []

    def fake_rule_fingerprints(rule_smarts, fingerprint_config):
        calls.append((tuple(rule_smarts), fingerprint_config))
        return torch.zeros((len(rule_smarts), 4))

    monkeypatch.setattr(
        template_based, "rule_fingerprints_from_smarts", fake_rule_fingerprints
    )
    wrapper = _mhn_wrapper(_MHNRDKitPolicy())

    wrapper.prepare_rule_associations([_Rule("A"), _Rule("B")])

    assert len(calls) == 1
    assert calls[0][0] == ("A", "B")
    assert calls[0][1].fp_type == "mhnreact_rdkit"


def test_mhn_preparation_uses_query_cgr_rule_graphs(monkeypatch):
    calls = []

    def fake_rule_graphs(rule_smarts, *, schema_version):
        calls.append((tuple(rule_smarts), schema_version))
        return [SimpleNamespace(rule=text) for text in rule_smarts]

    monkeypatch.setattr(
        template_based, "query_cgr_graphs_from_smarts", fake_rule_graphs
    )
    wrapper = _mhn_wrapper(_MHNGraphPolicy())
    rules = [_Rule("A"), _Rule("B")]

    wrapper.prepare_rule_associations(rules)
    first = wrapper._rule_associations
    wrapper.prepare_rule_associations(rules)

    assert calls == [(("A", "B"), RULE_GRAPH_SCHEMA_VERSION)]
    assert tuple(first.shape) == (2, 4)
    assert wrapper._rule_associations is first


def test_mhn_association_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(
        template_based,
        "rule_fingerprints_from_smarts",
        lambda rule_smarts, _config: torch.zeros((len(rule_smarts), 4)),
    )
    wrapper = _mhn_wrapper(_MHNPolicy())

    for index in range(_MAX_RULE_ASSOCIATION_CACHE_SIZE + 2):
        wrapper.prepare_rule_associations([_Rule(f"A{index}")])

    assert len(wrapper._rule_association_cache) == _MAX_RULE_ASSOCIATION_CACHE_SIZE


def test_linear_wrapper_has_no_mhn_rule_preparation_state():
    wrapper = _linear_wrapper(_LinearPolicy())

    assert not isinstance(wrapper, MHNReactPolicy)
    assert not hasattr(wrapper, "prepare_rule_associations")
    assert not hasattr(wrapper, "_rule_associations")


def test_mhn_wrapper_rejects_non_mhn_network():
    with pytest.raises(ValueError, match="mhn_ranking"):
        _mhn_wrapper(_LinearPolicy())


def test_mhn_light_prediction_requires_prepared_rule_associations():
    wrapper = _mhn_wrapper(_MHNPolicy())
    wrapper._get_graph = lambda _precursor: SimpleNamespace()

    with pytest.raises(ValueError, match="prepared by predict_reaction_rules"):
        list(wrapper.predict_reaction_rules_light(SimpleNamespace(), 2))


def test_light_prediction_uses_integer_count_without_preparing_rules():
    wrapper = _mhn_wrapper(_MHNPolicy())
    observed = []
    wrapper._predict_rules_common = lambda _precursor, n_rules: observed.append(n_rules)

    assert list(wrapper.predict_reaction_rules_light(SimpleNamespace(), 2)) == []
    assert wrapper._rule_associations is None
    assert observed == [2]


def test_combined_prediction_prepares_mhn_rules():
    prepared = []
    ranking = MHNReactPolicy.__new__(MHNReactPolicy)
    ranking.prepare_rule_associations = lambda rules: prepared.append(rules)
    combined = CompositePolicy.__new__(CompositePolicy)
    combined.ranking_policy = ranking
    combined._predict_rules_common = lambda _precursor, _n_rules: None
    rules = [_Rule("A"), _Rule("B")]

    assert list(combined.predict_reaction_rules(SimpleNamespace(), rules)) == []
    assert prepared == [rules]


def test_combined_light_prediction_uses_integer_count_without_preparing_rules():
    prepared = []
    observed = []
    ranking = MHNReactPolicy.__new__(MHNReactPolicy)
    ranking.prepare_rule_associations = lambda rules: prepared.append(rules)
    combined = CompositePolicy.__new__(CompositePolicy)
    combined.ranking_policy = ranking
    combined._predict_rules_common = lambda _precursor, n_rules: observed.append(
        n_rules
    )

    assert list(combined.predict_reaction_rules_light(SimpleNamespace(), 2)) == []
    assert prepared == []
    assert observed == [2]
