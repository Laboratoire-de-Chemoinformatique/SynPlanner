"""Rule encoders projecting reaction-rule representations into association space."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import Dropout, Identity, LayerNorm, Linear, Module, Sequential
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from synplan.chem.reaction.rules.representation import (
    RULE_GRAPH_EDGE_FEATURE_DIM,
    RULE_GRAPH_NODE_FEATURE_DIM,
)
from synplan.ml.networks.embedders import build_graph_embedder


def _normalization(association_dim: int, normalize: bool) -> Module:
    if normalize:
        return LayerNorm(association_dim, elementwise_affine=False)
    return Identity()


class RuleEncoder(Module):
    """Encode reaction rules into the molecule-rule association space."""

    out_dim: int

    @abstractmethod
    def encode(self, rules: Tensor | Sequence[Data]) -> Tensor:
        """Project raw rule representations into the association space.

        :param rules: Fingerprint tensor or a sequence of rule graphs.
        :return: ``(n_rules, association_dim)`` association tensor.
        """


class FingerprintRuleEncoder(RuleEncoder):
    """Linear projection of fixed-size rule fingerprints."""

    def __init__(
        self,
        fp_size: int,
        association_dim: int,
        dropout: float,
        normalize: bool,
    ) -> None:
        super().__init__()
        self.out_dim = association_dim
        self.projection = Sequential(
            Linear(fp_size, association_dim),
            Dropout(dropout),
            _normalization(association_dim, normalize),
        )

    def encode(self, rules: Tensor | Sequence[Data]) -> Tensor:
        if not torch.is_tensor(rules):
            raise ValueError("Fingerprint rule encoding requires a tensor input")
        return self.projection(rules.float())


class QueryCGRRuleEncoder(RuleEncoder):
    """Graph embedder + linear projection of QueryCGR rule graphs."""

    def __init__(
        self,
        rule_vector_dim: int,
        association_dim: int,
        dropout: float,
        normalize: bool,
        graph_batch_size: int,
        graph_embedder_type: str,
        num_conv_layers: int,
        heads: int,
        attn_type: str,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.out_dim = association_dim
        self.graph_batch_size = graph_batch_size
        self.embedder = build_graph_embedder(
            graph_embedder_type,
            rule_vector_dim,
            dropout=dropout,
            num_conv_layers=num_conv_layers,
            heads=heads,
            attn_type=attn_type,
            attn_dropout=attn_dropout,
            node_dim=RULE_GRAPH_NODE_FEATURE_DIM,
            edge_dim=RULE_GRAPH_EDGE_FEATURE_DIM,
        )
        self.projection = Sequential(
            Linear(rule_vector_dim, association_dim),
            Dropout(dropout),
            _normalization(association_dim, normalize),
        )

    def encode(self, rules: Tensor | Sequence[Data]) -> Tensor:
        if torch.is_tensor(rules):
            raise ValueError("QueryCGR graph rule encoding requires graph inputs")
        device = self.projection[0].weight.device
        if not rules:
            return torch.empty((0, self.out_dim), device=device)
        chunks = []
        for start in range(0, len(rules), self.graph_batch_size):
            batch = Batch.from_data_list(
                list(rules[start : start + self.graph_batch_size])
            ).to(device)
            chunks.append(self.projection(self.embedder(batch)))
        return torch.cat(chunks, dim=0)
