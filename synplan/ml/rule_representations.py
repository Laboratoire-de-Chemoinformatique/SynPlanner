"""Rule representation configuration and digests for MHN ranking."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from synplan.ml.rule_fingerprints import (
    RULE_FINGERPRINT_SCHEMA_VERSION,
    RuleFingerprintConfig,
)
from synplan.ml.rule_graphs import RULE_GRAPH_SCHEMA_VERSION

RuleEncoderType = Literal["fingerprint", "query_cgr_graph"]
RuleGraphEmbedderType = Literal["gcn", "gcn_concat", "gps"]

_RULE_ENCODER_TYPES = {"fingerprint", "query_cgr_graph"}
_RULE_GRAPH_EMBEDDER_TYPES = {"gcn", "gcn_concat", "gps"}


@dataclass(frozen=True)
class RuleRepresentationConfig:
    """Configuration that fully identifies an MHN rule representation."""

    encoder_type: RuleEncoderType = "fingerprint"
    fingerprint_config: RuleFingerprintConfig = field(
        default_factory=RuleFingerprintConfig
    )
    graph_schema_version: str = RULE_GRAPH_SCHEMA_VERSION
    graph_embedder_type: RuleGraphEmbedderType = "gps"
    graph_batch_size: int = 1024

    def __post_init__(self) -> None:
        if self.encoder_type not in _RULE_ENCODER_TYPES:
            expected = "', '".join(sorted(_RULE_ENCODER_TYPES))
            raise ValueError(f"mhn_rule_encoder_type must be one of '{expected}'")
        if self.graph_embedder_type not in _RULE_GRAPH_EMBEDDER_TYPES:
            expected = "', '".join(sorted(_RULE_GRAPH_EMBEDDER_TYPES))
            raise ValueError(f"mhn_rule_embedder_type must be one of '{expected}'")
        if self.encoder_type == "query_cgr_graph" and self.graph_embedder_type != "gps":
            raise ValueError(
                "mhn_rule_encoder_type='query_cgr_graph' requires "
                "mhn_rule_embedder_type='gps' because QueryCGR bond dynamics are "
                "encoded as edge attributes"
            )
        if not self.graph_schema_version:
            raise ValueError("mhn_rule_graph_schema_version must be non-empty")
        if self.graph_batch_size <= 0:
            raise ValueError("mhn_rule_graph_batch_size must be > 0")

    def to_digest_payload(self) -> dict[str, int | str | dict[str, int | str]]:
        """Return stable serializable values that affect rule representations."""
        payload: dict[str, int | str | dict[str, int | str]] = {
            "encoder_type": self.encoder_type,
        }
        if self.encoder_type == "fingerprint":
            payload["fingerprint_config"] = self.fingerprint_config.to_digest_payload()
        else:
            payload["graph_schema_version"] = self.graph_schema_version
            payload["graph_embedder_type"] = self.graph_embedder_type
            payload["graph_batch_size"] = self.graph_batch_size
        return payload


def rule_representation_config_from_policy(
    policy_net: object,
) -> RuleRepresentationConfig:
    """Build the rule representation contract stored on an MHN checkpoint."""
    return RuleRepresentationConfig(
        encoder_type=getattr(policy_net, "mhn_rule_encoder_type", "fingerprint"),
        fingerprint_config=RuleFingerprintConfig(
            fp_size=getattr(policy_net, "mhn_rule_fp_size", 2048),
            min_radius=getattr(policy_net, "mhn_rule_fp_min_radius", 1),
            max_radius=getattr(policy_net, "mhn_rule_fp_max_radius", 4),
            active_bits=getattr(policy_net, "mhn_rule_fp_active_bits", 2),
            fp_type=getattr(policy_net, "mhn_rule_fp_type", "query_cgr"),
            schema_version=getattr(
                policy_net,
                "mhn_rule_fp_schema_version",
                RULE_FINGERPRINT_SCHEMA_VERSION,
            ),
        ),
        graph_schema_version=getattr(
            policy_net, "mhn_rule_graph_schema_version", RULE_GRAPH_SCHEMA_VERSION
        ),
        graph_embedder_type=getattr(policy_net, "mhn_rule_embedder_type", "gps"),
        graph_batch_size=getattr(policy_net, "mhn_rule_graph_batch_size", 1024),
    )


def rule_representation_digest(
    rule_smarts: Sequence[str],
    representation_config: RuleRepresentationConfig | None = None,
) -> str:
    """Hash ordered rules and all settings that affect their representation."""
    representation_config = representation_config or RuleRepresentationConfig()
    payload = {
        "rules": tuple(rule_smarts),
        "representation_config": representation_config.to_digest_payload(),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "RuleEncoderType",
    "RuleGraphEmbedderType",
    "RuleRepresentationConfig",
    "rule_representation_config_from_policy",
    "rule_representation_digest",
]
