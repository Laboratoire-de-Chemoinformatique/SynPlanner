"""Torch-free configuration, schema versions and digests that fully identify an MHN rule representation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

RULE_FINGERPRINT_SCHEMA_VERSION = "1"
RULE_GRAPH_SCHEMA_VERSION = "1"

# Query-CGR rule-graph feature layout: single torch-free source of truth for the
# node/edge feature widths the ML tensorizer builds.
RULE_GRAPH_SIDES = ("reactants", "reagents", "products")
RULE_GRAPH_HYBRIDIZATIONS = (1, 2, 3, 4)
RULE_GRAPH_COUNT_LABELS = tuple(range(17))
RULE_GRAPH_RING_SIZE_LABELS = tuple(range(3, 17))
RULE_GRAPH_ORDER_LABELS = (None, 1, 2, 3, 4)
RULE_GRAPH_CHARGE_OFFSET = 8
_COUNT_SET_FEATURE_DIM = 1 + len(RULE_GRAPH_COUNT_LABELS) + 3
_RING_SET_FEATURE_DIM = 1 + len(RULE_GRAPH_RING_SIZE_LABELS) + 3
_BASE_NODE_FEATURE_DIM = (
    7 + (2 * _COUNT_SET_FEATURE_DIM) + (2 * len(RULE_GRAPH_HYBRIDIZATIONS))
)
_SIDE_NODE_FEATURE_DIM = (
    5
    + (3 * _COUNT_SET_FEATURE_DIM)
    + len(RULE_GRAPH_HYBRIDIZATIONS)
    + _RING_SET_FEATURE_DIM
)
RULE_GRAPH_NODE_FEATURE_DIM = _BASE_NODE_FEATURE_DIM + (
    len(RULE_GRAPH_SIDES) * _SIDE_NODE_FEATURE_DIM
)
RULE_GRAPH_EDGE_FEATURE_DIM = 16

RuleFingerprintType = Literal["legacy", "query_cgr"]
RuleEmbeddingType = Literal["fingerprint", "query_cgr_graph"]
RuleGraphEmbedderType = Literal["gcn", "gcn_concat", "gps"]

_RULE_FINGERPRINT_TYPES = {"legacy", "query_cgr"}
_RULE_EMBEDDING_TYPES = {"fingerprint", "query_cgr_graph"}
_RULE_GRAPH_EMBEDDER_TYPES = {"gcn", "gcn_concat", "gps"}


def validate_morgan_settings(
    *,
    fp_size: int,
    min_radius: int,
    max_radius: int,
    active_bits: int,
) -> None:
    """Validate Chython Morgan fingerprint settings."""
    if fp_size <= 0 or fp_size & (fp_size - 1):
        raise ValueError("rule_fp_size must be a positive power of two")
    if min_radius <= 0:
        raise ValueError("rule_fp_min_radius must be > 0")
    if max_radius < min_radius:
        raise ValueError("rule_fp_max_radius must be >= rule_fp_min_radius")
    if active_bits <= 0:
        raise ValueError("rule_fp_active_bits must be > 0")


def validate_rule_fingerprint_type(fp_type: str) -> None:
    """Validate the configured rule fingerprint implementation."""
    if fp_type not in _RULE_FINGERPRINT_TYPES:
        expected = "', '".join(sorted(_RULE_FINGERPRINT_TYPES))
        raise ValueError(f"rule_fp_type must be one of '{expected}'")


@dataclass(frozen=True)
class RuleFingerprintConfig:
    """Configuration that fully identifies an MHN rule fingerprint schema."""

    fp_size: int = 2048
    min_radius: int = 1
    max_radius: int = 4
    active_bits: int = 2
    fp_type: RuleFingerprintType = "query_cgr"
    schema_version: str = RULE_FINGERPRINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_morgan_settings(
            fp_size=self.fp_size,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            active_bits=self.active_bits,
        )
        validate_rule_fingerprint_type(self.fp_type)
        if not self.schema_version:
            raise ValueError("rule_fp_schema_version must be non-empty")

    def to_digest_payload(self) -> dict[str, int | str]:
        """Return stable serializable values that affect rule fingerprints."""
        return {
            "fp_size": self.fp_size,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "active_bits": self.active_bits,
            "fp_type": self.fp_type,
            "schema_version": self.schema_version,
        }


def rule_fingerprint_digest(
    rule_smarts: Sequence[str],
    fingerprint_config: RuleFingerprintConfig | None = None,
) -> str:
    """Hash ordered rules and all settings that affect rule fingerprints."""
    fingerprint_config = fingerprint_config or RuleFingerprintConfig()
    payload = {
        "rules": tuple(rule_smarts),
        "fingerprint_config": fingerprint_config.to_digest_payload(),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RuleRepresentationConfig:
    """Configuration that fully identifies an MHN rule representation."""

    embedding_type: RuleEmbeddingType = "fingerprint"
    fingerprint_config: RuleFingerprintConfig = field(
        default_factory=RuleFingerprintConfig
    )
    graph_schema_version: str = RULE_GRAPH_SCHEMA_VERSION
    graph_embedder_type: RuleGraphEmbedderType = "gps"
    graph_batch_size: int = 1024

    def __post_init__(self) -> None:
        if self.embedding_type not in _RULE_EMBEDDING_TYPES:
            expected = "', '".join(sorted(_RULE_EMBEDDING_TYPES))
            raise ValueError(f"rule_embedding_type must be one of '{expected}'")
        if self.graph_embedder_type not in _RULE_GRAPH_EMBEDDER_TYPES:
            expected = "', '".join(sorted(_RULE_GRAPH_EMBEDDER_TYPES))
            raise ValueError(f"rule_embedder.embedder_type must be one of '{expected}'")
        if (
            self.embedding_type == "query_cgr_graph"
            and self.graph_embedder_type != "gps"
        ):
            raise ValueError(
                "rule_embedding_type='query_cgr_graph' requires "
                "rule_embedder.embedder_type='gps' because QueryCGR bond dynamics are "
                "encoded as edge attributes"
            )
        if not self.graph_schema_version:
            raise ValueError("rule_graph_schema_version must be non-empty")
        if self.graph_batch_size <= 0:
            raise ValueError("rule_graph_batch_size must be > 0")

    def to_digest_payload(self) -> dict[str, int | str | dict[str, int | str]]:
        """Return stable serializable values that affect rule representations."""
        payload: dict[str, int | str | dict[str, int | str]] = {
            "embedding_type": self.embedding_type,
        }
        if self.embedding_type == "fingerprint":
            payload["fingerprint_config"] = self.fingerprint_config.to_digest_payload()
        else:
            payload["graph_schema_version"] = self.graph_schema_version
            payload["graph_embedder_type"] = self.graph_embedder_type
            payload["graph_batch_size"] = self.graph_batch_size
        return payload


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
    "RULE_FINGERPRINT_SCHEMA_VERSION",
    "RULE_GRAPH_CHARGE_OFFSET",
    "RULE_GRAPH_COUNT_LABELS",
    "RULE_GRAPH_EDGE_FEATURE_DIM",
    "RULE_GRAPH_HYBRIDIZATIONS",
    "RULE_GRAPH_NODE_FEATURE_DIM",
    "RULE_GRAPH_ORDER_LABELS",
    "RULE_GRAPH_RING_SIZE_LABELS",
    "RULE_GRAPH_SCHEMA_VERSION",
    "RULE_GRAPH_SIDES",
    "RuleEmbeddingType",
    "RuleFingerprintConfig",
    "RuleFingerprintType",
    "RuleGraphEmbedderType",
    "RuleRepresentationConfig",
    "rule_fingerprint_digest",
    "rule_representation_digest",
    "validate_morgan_settings",
    "validate_rule_fingerprint_type",
]
