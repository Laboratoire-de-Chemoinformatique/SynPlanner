"""Pure linear policy networks: ranking (softmax head) and filtering (sigmoid heads)."""

from typing import Any

import torch
from torch import Tensor
from torch.nn import Dropout, Linear
from torch_geometric.data.batch import Batch

from synplan.ml.networks.base import GraphMCTSNetwork
from synplan.utils.config import LinearPolicyNetworkConfig


class LinearPolicyNetwork(GraphMCTSNetwork):
    """Shared embedder + rule head plumbing for linear policy networks."""

    architecture = "linear"
    policy_type: str
    CONFIG_CLASS = LinearPolicyNetworkConfig

    def __init__(self, config: LinearPolicyNetworkConfig, n_rules: int) -> None:
        """Build the molecule embedder and the rule-scoring head.

        :param config: Policy network architecture and training config.
        :param n_rules: The number of reaction rules (output dimension).
        """
        super().__init__(
            config.vector_dim,
            config.batch_size,
            dropout=config.dropout,
            num_conv_layers=config.num_conv_layers,
            learning_rate=config.learning_rate,
            embedder_type=config.embedder_type,
            heads=config.heads,
            attn_type=config.attn_type,
            attn_dropout=config.attn_dropout,
        )
        self.n_rules = n_rules
        self.head_dropout = Dropout(config.dropout)
        self.y_predictor = Linear(config.vector_dim, n_rules)
        self.hparams: dict[str, Any] = {
            "config": config.model_dump(),
            "n_rules": n_rules,
        }


class RankingPolicyNetwork(LinearPolicyNetwork):
    """Linear ranking policy: embedder + softmax rule head."""

    policy_type = "ranking"

    def forward(self, batch: Batch) -> Tensor:
        """Return softmax rule probabilities for a batch of molecular graphs."""
        x = self.head_dropout(self.embedder(batch))
        return torch.softmax(self.y_predictor(x), dim=-1)


class FilteringPolicyNetwork(LinearPolicyNetwork):
    """Linear filtering policy: embedder + sigmoid rule head + sigmoid priority head."""

    policy_type = "filtering"

    def __init__(self, config: LinearPolicyNetworkConfig, n_rules: int) -> None:
        super().__init__(config, n_rules)
        self.priority_predictor = Linear(config.vector_dim, n_rules)

    def forward(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Return ``(rule_sigmoid, priority_sigmoid)`` for a batch of graphs."""
        x = self.head_dropout(self.embedder(batch))
        y = torch.sigmoid(self.y_predictor(x))
        priority = torch.sigmoid(self.priority_predictor(x))
        return y, priority
