"""Module containing main class for value network."""

from abc import ABC
from typing import Any, Dict

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.data.batch import Batch
from torchmetrics.functional.classification import (
    binary_f1_score,
    binary_recall,
    binary_specificity,
)

from synplan.ml.networks.modules import MCTSNetwork


class ValueNetwork(MCTSNetwork, LightningModule, ABC):
    """Value network."""

    def __init__(self, vector_dim: int, *args: Any, **kwargs: Any) -> None:
        """Initializes a value network, and creates linear layer for predicting the
        synthesisability of given precursor represented by molecular graph.

        :param vector_dim: The dimensionality of the output linear layer.
        """
        super().__init__(vector_dim, *args, **kwargs)
        self.save_hyperparameters()
        self.predictor = Linear(vector_dim, 1)

    def forward(self, batch) -> torch.Tensor:
        """Takes a batch of molecular graphs, applies a graph convolution returns the
        synthesisability (probability given by sigmoid function) of a given precursor
        represented by molecular graph precessed by graph convolution.

        :param batch: The batch of molecular graphs.
        :return: The predicted synthesisability (between 0 and 1).
        """

        x = self.embedder(batch, self.batch_size)
        x = torch.sigmoid(self.predictor(x))
        return x

    def _get_loss(self, batch: Batch) -> Dict[str, Tensor]:
        """Calculates the loss and various classification metrics for a given batch for
        the precursor synthesysability prediction.

        :param batch: The batch of molecular graphs.
        :return: The dictionary with loss value and balanced accuracy of precursor
            synthesysability prediction.
        """

        true_y = batch.y.float()
        true_y = torch.unsqueeze(true_y, -1)
        x = self.embedder(batch, self.batch_size)
        pred_y = self.predictor(x)
        # calc loss func
        loss = binary_cross_entropy_with_logits(pred_y, true_y)

        true_y = true_y.long()
        ba = (binary_recall(pred_y, true_y) + binary_specificity(pred_y, true_y)) / 2
        f1 = binary_f1_score(pred_y, true_y)
        metrics = {"loss": loss, "balanced_accuracy": ba, "f1_score": f1}
        return metrics
