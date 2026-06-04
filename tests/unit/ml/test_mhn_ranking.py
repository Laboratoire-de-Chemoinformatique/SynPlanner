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
from synplan.ml.template_features import (
    _MAX_TEMPLATE_FEATURE_CACHE_SIZE,
    _cache_set,
    _side_fingerprint,
    reaction_rules_path_from_policy_data,
    template_features_from_smarts,
)
from synplan.utils.config import PolicyNetworkConfig
from synplan.utils.loading import _policy_network_class_from_checkpoint

RULE_A = "[c:1]-[N:2]>>[c:1]-[N+:2](-[O-:3])=[O:4]"
RULE_B = "[C:1]-[O:2]>>[C:1].[O:2]"


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


def _network(template_features: torch.Tensor) -> MHNRankingPolicyNetwork:
    return MHNRankingPolicyNetwork(
        n_rules=template_features.shape[0],
        vector_dim=8,
        batch_size=1,
        dropout=0.0,
        num_conv_layers=1,
        learning_rate=0.001,
        template_features=template_features,
        mhn_association_dim=4,
        mhn_template_fp_size=template_features.shape[1],
    )


def test_template_features_are_deterministic_and_permutation_equivariant():
    features_1 = template_features_from_smarts((RULE_A, RULE_B), fp_size=16)
    features_2 = template_features_from_smarts((RULE_A, RULE_B), fp_size=16)
    reversed_features = template_features_from_smarts((RULE_B, RULE_A), fp_size=16)

    assert torch.equal(features_1, features_2)
    assert tuple(features_1.shape) == (2, 16)
    assert torch.equal(reversed_features, features_1.flip(0))


def test_template_feature_error_identifies_rule():
    with pytest.raises(ValueError, match=r"index 0"):
        template_features_from_smarts(("invalid",), fp_size=16)


def test_template_feature_cache_set_is_bounded_lru():
    cache = OrderedDict()

    for index in range(_MAX_TEMPLATE_FEATURE_CACHE_SIZE + 2):
        _cache_set(cache, str(index), torch.tensor([float(index)]))

    assert len(cache) == _MAX_TEMPLATE_FEATURE_CACHE_SIZE
    assert list(cache) == [
        str(index) for index in range(2, _MAX_TEMPLATE_FEATURE_CACHE_SIZE + 2)
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
    pooled = _side_fingerprint(
        reaction.products,
        fp_size=16,
        min_radius=1,
        max_radius=4,
        active_bits=2,
    )
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
        PolicyNetworkConfig(mhn_template_fp_size=1000)


def test_mhn_logits_probabilities_and_gradient_flow():
    features = template_features_from_smarts((RULE_A, RULE_B), fp_size=16)
    network = _network(features)
    batch = _graph_batch()

    logits = network.get_logits(batch)
    probs = network(batch)
    assert tuple(logits.shape) == (1, 2)
    assert tuple(probs.shape) == (1, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))

    network._get_loss(batch)["loss"].backward()
    assert network.molecule_encoder[0].weight.grad is not None
    assert network.template_encoder[0].weight.grad is not None


def test_mhn_accepts_dynamic_template_count_without_persisting_templates():
    features = template_features_from_smarts((RULE_A, RULE_B), fp_size=16)
    network = _network(features)
    dynamic_features = template_features_from_smarts((RULE_B,), fp_size=16)

    logits = network.get_logits(_graph_batch(), template_features=dynamic_features)

    assert tuple(logits.shape) == (1, 1)
    assert "_training_template_features" not in network.state_dict()


def test_mhn_prepares_training_templates_from_policy_mapping(tmp_path):
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
            mhn_template_fp_size=16,
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
    assert tuple(network._training_template_features.shape) == (2, 16)
    assert network.hparams["n_rules"] == 2
    assert network.hparams["mhn_template_feature_digest"] == (
        network.mhn_template_feature_digest
    )
    assert network.mhn_template_feature_digest is not None
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
