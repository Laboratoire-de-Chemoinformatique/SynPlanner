"""Checkpoint loading across config-based and pre-redesign flat-hparam shapes."""

import torch

from synplan.ml.networks.checkpoint import (
    LEGACY_FIELD_ALIASES,
    _config_from_flat_hparams,
    load_network_from_checkpoint,
)
from synplan.ml.networks.policy.linear import RankingPolicyNetwork
from synplan.utils.config import (
    LinearPolicyNetworkConfig,
    MHNRankingPolicyNetworkConfig,
)


def _ranking_net(n_rules: int) -> RankingPolicyNetwork:
    config = LinearPolicyNetworkConfig(
        policy_type="ranking", vector_dim=32, num_conv_layers=2, batch_size=8
    )
    return RankingPolicyNetwork(config=config, n_rules=n_rules)


def _save(tmp_path, hyper_parameters, state_dict):
    path = tmp_path / "policy.ckpt"
    torch.save({"hyper_parameters": hyper_parameters, "state_dict": state_dict}, path)
    return str(path)


def test_loads_pre_redesign_flat_policy_checkpoint(tmp_path):
    """Old flat-hparam policy checkpoints adapt into the config-based network."""
    net = _ranking_net(7)
    flat_hparams = {
        "vector_dim": 32,
        "num_conv_layers": 2,
        "dropout": 0.4,
        "learning_rate": 0.0005,
        "batch_size": 1000,
        "policy_type": "ranking",
        "n_rules": 7,
    }
    path = _save(tmp_path, flat_hparams, net.state_dict())

    loaded = load_network_from_checkpoint(
        RankingPolicyNetwork, path, batch_size=1, dropout=0
    )

    assert isinstance(loaded, RankingPolicyNetwork)
    assert loaded.n_rules == 7
    assert not loaded.training
    for original, restored in zip(
        net.state_dict().values(), loaded.state_dict().values()
    ):
        assert torch.equal(original, restored)


def test_loads_new_config_checkpoint(tmp_path):
    """The new ``{"config": {...}, "n_rules": N}`` shape still loads."""
    config = LinearPolicyNetworkConfig(
        policy_type="ranking", vector_dim=32, num_conv_layers=2, batch_size=8
    )
    net = RankingPolicyNetwork(config=config, n_rules=5)
    path = _save(
        tmp_path, {"config": config.model_dump(), "n_rules": 5}, net.state_dict()
    )

    loaded = load_network_from_checkpoint(RankingPolicyNetwork, path, batch_size=1)

    assert isinstance(loaded, RankingPolicyNetwork)
    assert loaded.n_rules == 5


def test_flat_hparam_adapter_migrates_renamed_fields():
    """Renamed config fields are migrated when adapting flat hparams."""
    assert LEGACY_FIELD_ALIASES["rule_encoder_type"] == "rule_embedding_type"

    config = _config_from_flat_hparams(
        MHNRankingPolicyNetworkConfig,
        {
            "rule_encoder_type": "fingerprint",
            "vector_dim": 64,
            "n_rules": 3,
            "obsolete_key": 1,
        },
        {},
    )

    assert config.rule_embedding_type == "fingerprint"
    assert config.vector_dim == 64


def test_flat_hparam_adapter_applies_overrides():
    """Scalar overrides win over the saved flat hparams."""
    config = _config_from_flat_hparams(
        LinearPolicyNetworkConfig,
        {"vector_dim": 32, "batch_size": 1000, "policy_type": "ranking"},
        {"batch_size": 1},
    )
    assert config.batch_size == 1
    assert config.vector_dim == 32
