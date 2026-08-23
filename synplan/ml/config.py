"""Configuration for policy and value networks and their training."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from synplan.chem.reaction.rules.representation.config import (
    RULE_FINGERPRINT_SCHEMA_VERSION,
    RULE_GRAPH_SCHEMA_VERSION,
    RuleFingerprintConfig,
    RuleRepresentationConfig,
)
from synplan.utils.config import BaseConfigModel


class GraphEmbedderConfig(BaseConfigModel):
    """Reusable graph-embedder knobs (``build_graph_embedder`` parameters).

    Used as the MHN ``rule_embedder``; the optional fields fall back to the
    molecule-side values when left unset (``None``).
    """

    embedder_type: Literal["gcn", "gcn_concat", "gps"] = "gps"
    vector_dim: int | None = Field(default=None, gt=0)
    num_conv_layers: int | None = Field(default=None, gt=0)
    heads: int | None = Field(default=None, gt=0)
    attn_type: Literal["performer", "multihead"] | None = None
    dropout: float | None = Field(default=None, ge=0.0, le=1.0)
    attn_dropout: float | None = Field(default=None, ge=0.0, le=1.0)


class PolicyNetworkConfig(BaseConfigModel):
    """Architecture-agnostic base config shared by every policy network.

    Holds the training + inference knobs common to all policies. It is the type
    used at planning time (only the selection knobs matter; the network
    architecture is read from the checkpoint) and the base for the
    architecture-specific training configs :class:`LinearPolicyNetworkConfig` and
    :class:`MHNRankingPolicyNetworkConfig`. ``from_dict``/``from_yaml`` dispatch on
    the ``architecture`` discriminator and return the matching subclass.

    :param vector_dim: Dimension of the input vectors.
    :param batch_size: Number of samples per batch.
    :param dropout: Dropout rate for regularization.
    :param learning_rate: Learning rate for the optimizer.
    :param num_conv_layers: Number of convolutional layers in the network.
    :param num_epoch: Number of training epochs.
    :param policy_type: Mode of operation, either 'filtering' or 'ranking'.
    :param logger: Training logger configuration. ``None`` disables logging.
        A dict with ``"type"`` key (``"csv"``, ``"tensorboard"``, ``"mlflow"``,
        ``"litlogger"``, or ``"wandb"``) and optional logger-specific parameters
        passed to the PyTorch Lightning logger constructor.
    """

    policy_type: Literal["filtering", "ranking"] = "ranking"
    architecture: Literal["linear", "mhn_ranking"] = "linear"
    embedder_type: Literal["gcn", "gcn_concat", "gps"] = "gcn"
    vector_dim: int = Field(default=256, gt=0)
    batch_size: int = Field(default=500, gt=0)
    dropout: float = Field(default=0.4, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.008, gt=0.0)
    num_conv_layers: int = Field(default=5, gt=0)
    num_epoch: int = Field(default=100, gt=0)
    weights_path: str | Path | None = None

    # GPS embedder parameters (only used when embedder_type="gps")
    heads: int = Field(default=4, gt=0)
    attn_type: Literal["performer", "multihead"] = "performer"
    attn_dropout: float = Field(default=0.5, ge=0.0, le=1.0)

    # training logger (None disables logging, or dict with "type" + logger kwargs)
    logger: dict | None = None

    # extra Trainer kwargs (None = use defaults, or dict passed to Lightning Trainer)
    trainer: dict | None = None

    # logging gradient norms per module (embedder, y_predictor, etc.)
    log_grad_norm: bool = False

    # for filtering policy
    priority_rules_fraction: float = Field(default=0.5, ge=0.0)
    rule_prob_threshold: float = Field(default=0.0, ge=0.0)
    top_rules: int = Field(default=50, gt=0)

    @model_validator(mode="after")
    def _validate_architecture(self) -> "PolicyNetworkConfig":
        if self.architecture == "mhn_ranking" and self.policy_type != "ranking":
            raise ValueError(
                "architecture='mhn_ranking' requires policy_type='ranking'"
            )
        if self.embedder_type == "gcn_concat" and (
            self.vector_dim % self.num_conv_layers
        ):
            raise ValueError(
                "embedder_type='gcn_concat' requires vector_dim to be divisible "
                "by num_conv_layers"
            )
        return self

    @field_validator("logger")
    @classmethod
    def _validate_logger(cls, v: dict | None) -> dict | None:
        if v is not None:
            if "type" not in v:
                raise ValueError("logger dict must contain a 'type' key.")
            valid_types = ("csv", "tensorboard", "mlflow", "wandb", "litlogger")
            if v["type"].lower() not in valid_types:
                raise ValueError(
                    f"logger type must be one of {valid_types}, got '{v['type']}'"
                )
            v["type"] = v["type"].lower()
        return v

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        """Build a policy config, dispatching on ``architecture``.

        Called on the base class it returns the concrete subclass selected by the
        ``architecture`` key (absent → ``linear``); called on a subclass it parses
        as that subclass directly.
        """
        data = dict(config_dict or {})
        if cls is PolicyNetworkConfig:
            cls = _POLICY_CONFIG_BY_ARCHITECTURE.get(
                data.get("architecture", "linear"), cls
            )
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, file_path: str):
        """Load a policy config from YAML, dispatching on ``architecture``."""
        with open(file_path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})


class LinearPolicyNetworkConfig(PolicyNetworkConfig):
    """Fixed-class (linear) policy network training config.

    The standard graph-embedder policy with a fixed reaction-rule output head
    (filtering or ranking). Pins ``architecture``; an explicit peer of
    :class:`MHNRankingPolicyNetworkConfig`.
    """

    architecture: Literal["linear"] = "linear"


class MHNRankingPolicyNetworkConfig(PolicyNetworkConfig):
    """Modern-Hopfield ranking policy network training config.

    Extends the shared policy config with the MHN association knobs and the rule
    representation. The molecule/product embedding uses the inherited base fields;
    the rule embedding is configured by ``rule_embedder`` (a reusable
    :class:`GraphEmbedderConfig` whose unset fields fall back to the molecule
    side). These fields live ONLY here, never on the base or linear config.
    """

    architecture: Literal["mhn_ranking"] = "mhn_ranking"

    # MHN association mechanism
    association_dim: int = Field(default=512, gt=0)
    beta: float = Field(default=0.05, gt=0.0)
    normalize_associations: bool = True

    # rule representation
    rule_embedding_type: Literal["fingerprint", "query_cgr_graph"] = "fingerprint"
    rule_fp_size: int = Field(default=2048, gt=0)
    rule_fp_min_radius: int = Field(default=1, gt=0)
    rule_fp_max_radius: int = Field(default=4, ge=0)
    rule_fp_active_bits: int = Field(default=2, gt=0)
    rule_fp_type: Literal["legacy", "mhnreact_rdkit", "query_cgr"] = "query_cgr"
    rule_fp_schema_version: str = Field(
        default=RULE_FINGERPRINT_SCHEMA_VERSION, min_length=1
    )
    rule_graph_batch_size: int = Field(default=1024, gt=0)
    rule_graph_schema_version: str = Field(
        default=RULE_GRAPH_SCHEMA_VERSION, min_length=1
    )
    rule_embedder: GraphEmbedderConfig = Field(default_factory=GraphEmbedderConfig)

    def rule_representation_config(self) -> RuleRepresentationConfig:
        """Map the flat rule knobs onto the canonical frozen chem config.

        Constructing the frozen config runs its ``__post_init__`` checks
        (power-of-two fingerprint size, radius ordering, the ``query_cgr_graph``
        → ``gps`` rule), so it doubles as the rule-field validator.
        """
        return RuleRepresentationConfig(
            embedding_type=self.rule_embedding_type,
            fingerprint_config=RuleFingerprintConfig(
                fp_size=self.rule_fp_size,
                min_radius=self.rule_fp_min_radius,
                max_radius=self.rule_fp_max_radius,
                active_bits=self.rule_fp_active_bits,
                fp_type=self.rule_fp_type,
                schema_version=self.rule_fp_schema_version,
            ),
            graph_schema_version=self.rule_graph_schema_version,
            graph_embedder_type=self.rule_embedder.embedder_type,
            graph_batch_size=self.rule_graph_batch_size,
        )

    @model_validator(mode="after")
    def _validate_rule_representation(self) -> "MHNRankingPolicyNetworkConfig":
        self.rule_representation_config()
        return self

    def network_kwargs(self) -> dict[str, Any]:
        """Return the MHN association + rule-embedding kwargs for the network.

        The nested ``rule_embedder`` is expanded to the ``rule_*`` embedder kwargs
        the network expects (``None`` keeps the molecule-side fallback).
        """
        emb = self.rule_embedder
        return {
            "association_dim": self.association_dim,
            "beta": self.beta,
            "normalize_associations": self.normalize_associations,
            "rule_embedding_type": self.rule_embedding_type,
            "rule_fp_size": self.rule_fp_size,
            "rule_fp_min_radius": self.rule_fp_min_radius,
            "rule_fp_max_radius": self.rule_fp_max_radius,
            "rule_fp_active_bits": self.rule_fp_active_bits,
            "rule_fp_type": self.rule_fp_type,
            "rule_fp_schema_version": self.rule_fp_schema_version,
            "rule_graph_batch_size": self.rule_graph_batch_size,
            "rule_graph_schema_version": self.rule_graph_schema_version,
            "rule_embedder_type": emb.embedder_type,
            "rule_vector_dim": emb.vector_dim,
            "rule_num_conv_layers": emb.num_conv_layers,
            "rule_heads": emb.heads,
            "rule_attn_type": emb.attn_type,
            "rule_dropout": emb.dropout,
            "rule_attn_dropout": emb.attn_dropout,
        }


_POLICY_CONFIG_BY_ARCHITECTURE: dict[str, type[PolicyNetworkConfig]] = {
    "linear": LinearPolicyNetworkConfig,
    "mhn_ranking": MHNRankingPolicyNetworkConfig,
}


class ValueNetworkConfig(BaseConfigModel):
    """Configuration class for the value network.

    :param vector_dim: Dimension of the input vectors.
    :param batch_size: Number of samples per batch.
    :param dropout: Dropout rate for regularization.
    :param learning_rate: Learning rate for the optimizer.
    :param num_conv_layers: Number of convolutional layers in the network.
    :param num_epoch: Number of training epochs.
    """

    weights_path: str | Path | None = None
    vector_dim: int = Field(default=256, gt=0)
    batch_size: int = Field(default=500, gt=0)
    dropout: float = Field(default=0.4, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.008, gt=0.0)
    num_conv_layers: int = Field(default=5, gt=0)
    num_epoch: int = Field(default=100, gt=0)


class TuningConfig(BaseConfigModel):
    """Configuration class for the network training.

    :param batch_size: The number of targets per batch in the planning simulation step.
    :param num_simulations: The number of planning simulations.
    """

    batch_size: int = Field(default=100, gt=0)
    num_simulations: int = 1
