"""Lightning-free loader rebuilding pure networks from checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import torch
from torch.nn import Module

from synplan.ml.networks.policy.linear import (
    FilteringPolicyNetwork,
    RankingPolicyNetwork,
)
from synplan.ml.networks.policy.mhnreact import MHNReact
from synplan.ml.networks.value import ValueNetwork

T = TypeVar("T", bound=Module)

# hparams keys that are not constructor arguments (legacy flat-hparam shape).
DISCRIMINATOR_KEYS = ("architecture", "policy_type")


def load_network_from_checkpoint(
    model_class: type[T],
    path: str | Path,
    map_location: str | torch.device = "cpu",
    **overrides,
) -> T:
    """Rebuild a pure network from a Lightning-format checkpoint.

    Supports both the new config-based hparam shape
    ``{"config": {...}, "n_rules": N}`` and the old flat shape (ValueNetwork).
    Reads ``hyper_parameters`` to construct ``model_class`` and loads its
    ``state_dict``.

    :param model_class: The pure ``nn.Module`` subclass to instantiate.
    :param path: Path to the checkpoint file.
    :param map_location: Device for ``torch.load``.
    :param overrides: For flat-hparam networks: constructor overrides applied on
        top of the saved hparams. For config-based networks: scalar fields present
        in the config dict are updated before construction (e.g. ``batch_size=1``).
    :return: The constructed model with weights loaded, in eval mode.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    hparams = dict(checkpoint.get("hyper_parameters", {}))

    if "config" in hparams:
        # New-style: reconstruct config, apply overrides into config dict.
        config_cls = getattr(model_class, "CONFIG_CLASS", None)
        if config_cls is None:
            raise ValueError(
                f"{model_class.__name__} has 'config' in hparams but no CONFIG_CLASS"
            )
        config_dict = dict(hparams["config"])
        for key, val in overrides.items():
            if key in config_dict:
                config_dict[key] = val
        config = config_cls.model_validate(config_dict)
        n_rules = hparams.get("n_rules", 0)
        model = model_class(config=config, n_rules=n_rules)
        digest = hparams.get("rule_representation_digest")
        if digest is not None and hasattr(model, "rule_representation_digest"):
            model.rule_representation_digest = digest
    else:
        # Legacy flat-hparam shape (ValueNetwork and pre-refactor checkpoints).
        for key in DISCRIMINATOR_KEYS:
            hparams.pop(key, None)
        hparams.update(overrides)
        model = model_class(**hparams)

    model.load_state_dict(_strip_wrapper_prefix(checkpoint["state_dict"], model))
    return model.eval()


def policy_network_class_from_checkpoint(path: str | Path) -> type[Module]:
    """Return the pure policy class a checkpoint declares.

    Dispatches on the ``architecture`` / ``policy_type`` hparams; handles both
    the new config-based shape and the old flat shape.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})

    if "config" in hparams:
        architecture = hparams["config"].get("architecture", "linear")
        policy_type = hparams["config"].get("policy_type", "ranking")
    else:
        architecture = hparams.get("architecture", "linear")
        policy_type = hparams.get("policy_type", "ranking")

    if architecture == "mhn_ranking":
        return MHNReact
    if policy_type == "filtering":
        return FilteringPolicyNetwork
    return RankingPolicyNetwork


def load_policy_network_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    **overrides,
) -> Module:
    """Load the pure policy network a checkpoint declares (value-free path)."""
    model_class = policy_network_class_from_checkpoint(path)
    return load_network_from_checkpoint(
        model_class, path, map_location=map_location, **overrides
    )


def load_value_network_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    **overrides,
) -> ValueNetwork:
    """Load a pure :class:`ValueNetwork` from a checkpoint."""
    return load_network_from_checkpoint(
        ValueNetwork, path, map_location=map_location, **overrides
    )


def _strip_wrapper_prefix(state_dict: dict, model: Module) -> dict:
    """Strip a single Lightning wrapper prefix (e.g. ``network.``) if present."""
    own_keys = set(model.state_dict().keys())
    if set(state_dict) & own_keys:
        return state_dict
    for key in state_dict:
        head = key.split(".", 1)[0] + "."
        if any(k.startswith(head) and k[len(head) :] in own_keys for k in state_dict):
            return {
                k[len(head) :]: v for k, v in state_dict.items() if k.startswith(head)
            }
    return state_dict
