"""Tests for the MHN-style dynamic ranking policy."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from chython import smarts
from torch_geometric.data import Batch, Data

from synplan.chem.utils import reaction_query_to_reaction
from synplan.ml.networks.mhn_ranking import MHNRankingPolicyNetwork
from synplan.ml.networks.policy import PolicyNetwork
from synplan.ml.rule_fingerprints import (
    _MAX_RULE_FINGERPRINT_CACHE_SIZE,
    RuleFingerprintConfig,
    _cache_set,
    _side_fingerprint,
    reaction_rules_path_from_policy_data,
    rule_fingerprint_digest,
    rule_fingerprints_from_smarts,
)
from synplan.utils.config import PolicyNetworkConfig
from synplan.utils.loading import _policy_network_class_from_checkpoint

RULE_A = "[c:1]-[N:2]>>[c:1]-[N+:2](-[O-:3])=[O:4]"
RULE_B = "[C:1]-[O:2]>>[C:1].[O:2]"
RULE_D2 = "[C;D2:1]-[O:2]>>[C:1].[O:2]"
RULE_D3 = "[C;D3:1]-[O:2]>>[C:1].[O:2]"
RULE_H1 = "[O;h1:1]>>[O:1]"
RULE_H0 = "[O;h0:1]>>[O:1]"
RULE_R5 = "[C;r5:1]-[O:2]>>[C:1].[O:2]"
RULE_R6 = "[C;r6:1]-[O:2]>>[C:1].[O:2]"


def _fp_config(
    *,
    fp_size: int = 16,
    fp_type: str = "query_cgr",
    schema_version: str = "1",
) -> RuleFingerprintConfig:
    return RuleFingerprintConfig(
        fp_size=fp_size, fp_type=fp_type, schema_version=schema_version
    )


def _graph_batch() -> Batch:
    graph = Data(
        x=torch.tensor(
            [
                [6, 2, 14, 4, 2, 0, 0, 1, 1, 0, 0],
                [8, 2, 16, 2, 2, 0, 0, 1, 1, 0, 0],
            ],
            dtype=torch.uint8,
        ),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float,
        ),
        y_rules=torch.tensor([0], dtype=torch.long),
    )
    return Batch.from_data_list([graph])


def _network(rule_fingerprints: torch.Tensor) -> MHNRankingPolicyNetwork:
    return MHNRankingPolicyNetwork(
        n_rules=rule_fingerprints.shape[0],
        vector_dim=8,
        batch_size=1,
        dropout=0.0,
        num_conv_layers=1,
        learning_rate=0.001,
        rule_fingerprints=rule_fingerprints,
        mhn_association_dim=4,
        mhn_rule_fp_size=rule_fingerprints.shape[1],
    )


def test_rule_fingerprints_are_deterministic_and_permutation_equivariant():
    fingerprints_1 = rule_fingerprints_from_smarts((RULE_A, RULE_B), _fp_config())
    fingerprints_2 = rule_fingerprints_from_smarts((RULE_A, RULE_B), _fp_config())
    reversed_fingerprints = rule_fingerprints_from_smarts(
        (RULE_B, RULE_A), _fp_config()
    )

    assert torch.equal(fingerprints_1, fingerprints_2)
    assert tuple(fingerprints_1.shape) == (2, 16)
    assert torch.equal(reversed_fingerprints, fingerprints_1.flip(0))


def test_rule_fingerprint_error_identifies_rule():
    with pytest.raises(ValueError, match=r"index 0"):
        rule_fingerprints_from_smarts(("invalid",), _fp_config())


def test_legacy_rule_fingerprints_drop_query_labels():
    legacy_rule_fingerprints = rule_fingerprints_from_smarts(
        (RULE_D2, RULE_D3), _fp_config(fp_size=2048, fp_type="legacy")
    )

    assert torch.equal(legacy_rule_fingerprints[0], legacy_rule_fingerprints[1])


@pytest.mark.parametrize(
    ("left_rule", "right_rule"),
    [
        (RULE_D2, RULE_D3),
        (RULE_H1, RULE_H0),
        (RULE_R5, RULE_R6),
    ],
)
def test_query_cgr_rule_fingerprints_keep_query_labels(left_rule, right_rule):
    query_cgr_rule_fingerprints = rule_fingerprints_from_smarts(
        (left_rule, right_rule), _fp_config(fp_size=2048, fp_type="query_cgr")
    )

    assert not torch.equal(
        query_cgr_rule_fingerprints[0], query_cgr_rule_fingerprints[1]
    )


def test_rule_fingerprint_digest_includes_fingerprint_config():
    legacy_digest = rule_fingerprint_digest((RULE_A,), _fp_config(fp_type="legacy"))
    query_cgr_digest = rule_fingerprint_digest(
        (RULE_A,), _fp_config(fp_type="query_cgr")
    )
    schema_digest = rule_fingerprint_digest(
        (RULE_A,), _fp_config(fp_type="query_cgr", schema_version="2")
    )

    assert legacy_digest != query_cgr_digest
    assert query_cgr_digest != schema_digest


def test_rule_fingerprint_cache_set_is_bounded_lru():
    cache = OrderedDict()

    for index in range(_MAX_RULE_FINGERPRINT_CACHE_SIZE + 2):
        _cache_set(cache, str(index), torch.tensor([float(index)]))

    assert len(cache) == _MAX_RULE_FINGERPRINT_CACHE_SIZE
    assert list(cache) == [
        str(index) for index in range(2, _MAX_RULE_FINGERPRINT_CACHE_SIZE + 2)
    ]

    _cache_set(cache, "2", torch.tensor([2.0]))
    _cache_set(cache, "new", torch.tensor([99.0]))

    assert "3" not in cache
    assert list(cache)[-1] == "new"


def test_reaction_rules_path_is_inferred_from_extracted_policy_mapping(tmp_path):
    rules_path = tmp_path / "reaction_rules.tsv"
    rules_path.write_text("rule_smarts\tpopularity\treaction_indices\n")
    policy_data_path = tmp_path / "reaction_rules_policy_data.tsv"
    policy_data_path.write_text("product_smiles\trule_id\n")

    assert reaction_rules_path_from_policy_data(policy_data_path) == rules_path


def test_reaction_rules_path_rejects_non_extracted_mapping_name(tmp_path):
    policy_data_path = tmp_path / "policy.tsv"
    policy_data_path.write_text("product_smiles\trule_id\n")

    with pytest.raises(ValueError, match=r"\*_policy_data.tsv"):
        reaction_rules_path_from_policy_data(policy_data_path)


def test_side_fingerprint_max_pools_fragments():
    reaction = reaction_query_to_reaction(smarts(RULE_B))
    pooled = _side_fingerprint(reaction.products, _fp_config())
    individual = [
        torch.as_tensor(
            molecule.morgan_fingerprint(
                min_radius=1,
                max_radius=4,
                length=16,
                number_active_bits=2,
            ),
            dtype=torch.float,
        )
        for molecule in reaction.products
    ]

    assert torch.equal(pooled, torch.stack(individual).amax(dim=0))


def test_mhn_config_validation():
    with pytest.raises(ValueError, match="requires policy_type='ranking'"):
        PolicyNetworkConfig(architecture="mhn_ranking", policy_type="filtering")
    with pytest.raises(ValueError, match="positive power of two"):
        PolicyNetworkConfig(mhn_rule_fp_size=1000)
    with pytest.raises(ValueError):
        PolicyNetworkConfig(mhn_rule_fp_min_radius=0)
    with pytest.raises(ValueError):
        PolicyNetworkConfig(mhn_rule_fp_type="unknown")
    with pytest.raises(ValueError):
        RuleFingerprintConfig(min_radius=0)

    config = PolicyNetworkConfig(architecture="mhn_ranking")
    assert config.mhn_rule_fp_type == "query_cgr"
    assert config.mhn_rule_fp_schema_version == "1"


def test_mhn_logits_probabilities_and_gradient_flow():
    fingerprints = rule_fingerprints_from_smarts((RULE_A, RULE_B), _fp_config())
    network = _network(fingerprints)
    batch = _graph_batch()

    logits = network.get_logits(batch)
    probs = network(batch)
    assert tuple(logits.shape) == (1, 2)
    assert tuple(probs.shape) == (1, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))

    network._get_loss(batch)["loss"].backward()
    assert network.molecule_encoder[0].weight.grad is not None
    assert network.rule_encoder[0].weight.grad is not None


def test_mhn_accepts_dynamic_rule_count_without_persisting_rules():
    fingerprints = rule_fingerprints_from_smarts((RULE_A, RULE_B), _fp_config())
    network = _network(fingerprints)
    dynamic_rule_fingerprints = rule_fingerprints_from_smarts((RULE_B,), _fp_config())

    logits = network.get_logits(
        _graph_batch(), rule_fingerprints=dynamic_rule_fingerprints
    )

    assert tuple(logits.shape) == (1, 1)
    assert "_training_rule_fingerprints" not in network.state_dict()


def test_mhn_prepares_training_rules_from_policy_mapping(tmp_path):
    rules_path = tmp_path / "reaction_rules.tsv"
    rules_path.write_text(
        f"rule_smarts\tpopularity\treaction_indices\n{RULE_A}\t1\t0\n{RULE_B}\t1\t1\n",
        encoding="utf-8",
    )
    policy_data_path = tmp_path / "reaction_rules_policy_data.tsv"
    policy_data_path.write_text(
        "product_smiles\trule_id\nCC\t0\n",
        encoding="utf-8",
    )

    network = MHNRankingPolicyNetwork.for_training(
        dataset=SimpleNamespace(
            policy_data_path=str(policy_data_path),
            _data=SimpleNamespace(y_rules=torch.tensor([0])),
        ),
        config=PolicyNetworkConfig(
            architecture="mhn_ranking",
            mhn_association_dim=4,
            mhn_rule_fp_size=16,
            mhn_rule_fp_type="query_cgr",
            mhn_rule_fp_schema_version="1",
        ),
        n_rules=1,
        vector_dim=8,
        batch_size=1,
        dropout=0.0,
        num_conv_layers=1,
        learning_rate=0.001,
        policy_type="ranking",
    )

    assert network.n_rules == 2
    assert tuple(network._training_rule_fingerprints.shape) == (2, 16)
    assert network.hparams["n_rules"] == 2
    assert network.hparams["mhn_rule_fingerprint_digest"] == (
        network.mhn_rule_fingerprint_digest
    )
    assert network.mhn_rule_fingerprint_digest is not None
    assert network.hparams["mhn_rule_fp_type"] == "query_cgr"
    assert network.hparams["mhn_rule_fp_schema_version"] == "1"
    assert "policy_data_path" not in network.hparams


@pytest.mark.parametrize(
    ("hyperparameters", "expected_class"),
    [
        ({}, PolicyNetwork),
        ({"architecture": "linear"}, PolicyNetwork),
        ({"architecture": "mhn_ranking"}, MHNRankingPolicyNetwork),
    ],
)
def test_checkpoint_class_dispatch_defaults_to_linear(
    tmp_path, hyperparameters, expected_class
):
    checkpoint = tmp_path / "policy.ckpt"
    torch.save({"hyper_parameters": hyperparameters}, checkpoint)

    assert _policy_network_class_from_checkpoint(checkpoint) is expected_class
