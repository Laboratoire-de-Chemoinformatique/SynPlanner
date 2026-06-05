"""Configuration adapter for MHN ranking policy networks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

from synplan.chem.reaction_rules.graphs import RULE_GRAPH_SCHEMA_VERSION
from synplan.chem.reaction_rules.rule_fingerprints import (
    RULE_FINGERPRINT_SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from synplan.utils.config import PolicyNetworkConfig


class MHNRankingNetworkConfig(BaseModel):
    """Pydantic view of the PolicyNetworkConfig fields used by MHN ranking."""

    architecture: Literal["mhn_ranking"] = "mhn_ranking"
    mhn_association_dim: int = Field(default=512, gt=0)
    mhn_beta: float = Field(default=0.05, gt=0.0)
    mhn_rule_encoder_type: Literal["fingerprint", "query_cgr_graph"] = "fingerprint"
    mhn_rule_embedder_type: Literal["gcn", "gcn_concat", "gps"] = "gps"
    mhn_rule_graph_batch_size: int = Field(default=1024, gt=0)
    mhn_rule_graph_schema_version: str = Field(
        default=RULE_GRAPH_SCHEMA_VERSION, min_length=1
    )
    mhn_rule_vector_dim: int | None = Field(default=None, gt=0)
    mhn_rule_num_conv_layers: int | None = Field(default=None, gt=0)
    mhn_rule_heads: int | None = Field(default=None, gt=0)
    mhn_rule_attn_type: Literal["performer", "multihead"] | None = None
    mhn_rule_dropout: float | None = Field(default=None, ge=0.0, le=1.0)
    mhn_rule_attn_dropout: float | None = Field(default=None, ge=0.0, le=1.0)
    mhn_rule_fp_size: int = Field(default=2048, gt=0)
    mhn_rule_fp_min_radius: int = Field(default=1, gt=0)
    mhn_rule_fp_max_radius: int = Field(default=4, ge=0)
    mhn_rule_fp_active_bits: int = Field(default=2, gt=0)
    mhn_rule_fp_type: Literal["legacy", "query_cgr"] = "query_cgr"
    mhn_rule_fp_schema_version: str = Field(
        default=RULE_FINGERPRINT_SCHEMA_VERSION, min_length=1
    )
    mhn_normalize_associations: bool = True

    @model_validator(mode="after")
    def _validate_rule_representation(self) -> MHNRankingNetworkConfig:
        if self.mhn_rule_fp_size & (self.mhn_rule_fp_size - 1):
            raise ValueError("mhn_rule_fp_size must be a positive power of two")
        if self.mhn_rule_fp_max_radius < self.mhn_rule_fp_min_radius:
            raise ValueError("mhn_rule_fp_max_radius must be >= mhn_rule_fp_min_radius")
        if (
            self.mhn_rule_encoder_type == "query_cgr_graph"
            and self.mhn_rule_embedder_type != "gps"
        ):
            raise ValueError(
                "mhn_rule_encoder_type='query_cgr_graph' requires "
                "mhn_rule_embedder_type='gps'"
            )
        return self

    @classmethod
    def from_policy_config(
        cls, config: PolicyNetworkConfig
    ) -> MHNRankingNetworkConfig:
        """Extract the MHN network contract from the general policy config."""
        return cls.model_validate(config.model_dump(include=MHN_NETWORK_CONFIG_FIELDS))

    def to_network_kwargs(self) -> dict[str, Any]:
        """Return constructor kwargs for :class:`MHNRankingPolicyNetwork`."""
        return self.model_dump()


MHN_NETWORK_CONFIG_FIELDS = frozenset(MHNRankingNetworkConfig.model_fields)


def mhn_network_kwargs_from_policy(config: PolicyNetworkConfig) -> dict[str, Any]:
    """Return MHN ranking network kwargs from the general policy config."""
    return MHNRankingNetworkConfig.from_policy_config(config).to_network_kwargs()


__all__ = [
    "MHNRankingNetworkConfig",
    "mhn_network_kwargs_from_policy",
]
