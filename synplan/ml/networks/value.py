"""Pure value network: embedder + value head + forward (no training plumbing)."""

from typing import Any

import torch
from torch.nn import Linear

from synplan.ml.networks.base import GraphMCTSNetwork


class ValueNetwork(GraphMCTSNetwork):
    """Value network predicting precursor synthesisability."""

    def __init__(
        self,
        vector_dim: int,
        batch_size: int,
        dropout: float = 0.4,
        num_conv_layers: int = 5,
        learning_rate: float = 0.001,
        **kwargs: Any,
    ) -> None:
        """Initializes a value network, and creates linear layer for predicting the
        synthesisability of given precursor represented by molecular graph.

        :param vector_dim: The dimensionality of the output linear layer.
        """
        super().__init__(
            vector_dim,
            batch_size,
            dropout=dropout,
            num_conv_layers=num_conv_layers,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.predictor = Linear(vector_dim, 1)
        self.hparams = {
            "vector_dim": vector_dim,
            "batch_size": batch_size,
            "dropout": dropout,
            "num_conv_layers": num_conv_layers,
            "learning_rate": learning_rate,
        }

    def forward(self, batch) -> torch.Tensor:
        """Takes a batch of molecular graphs, applies a graph convolution returns the
        synthesisability (probability given by sigmoid function) of a given precursor
        represented by molecular graph precessed by graph convolution.

        :param batch: The batch of molecular graphs.
        :return: The predicted synthesisability (between 0 and 1).
        """

        x = self.embedder(batch)
        x = torch.sigmoid(self.predictor(x))
        return x
