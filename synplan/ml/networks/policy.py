"""Module containing main class for policy network."""

from abc import ABC
from typing import Dict

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, one_hot
from torch_geometric.data.batch import Batch
from torchmetrics.functional.classification import f1_score, recall, specificity

from synplan.ml.networks.modules import MCTSNetwork


class PolicyNetwork(MCTSNetwork, LightningModule, ABC):
    """Policy network."""

    def __init__(
        self,
        *args,
        n_rules: int,
        vector_dim: int,
        policy_type: str = "ranking",
        **kwargs
    ):
        """Initializes a policy network with the given number of reaction rules (output
        dimension) and vector graph embedding dimension, and creates linear layers for
        predicting the regular and priority reaction rules.

        :param n_rules: The number of reaction rules in the policy network.
        :param vector_dim: The dimensionality of the input vectors.
        """
        super().__init__(vector_dim, *args, **kwargs)
        self.save_hyperparameters()
        self.policy_type = policy_type
        self.n_rules = n_rules
        self.y_predictor = Linear(vector_dim, n_rules)

        if self.policy_type == "filtering":
            self.priority_predictor = Linear(vector_dim, n_rules)

    def forward(self, batch: Batch) -> Tensor:
        """Takes a molecular graph, applies a graph convolution and sigmoid layers to
        predict regular and priority reaction rules.

        :param batch: The input batch of molecular graphs.
        :return: Returns the vector of probabilities (given by sigmoid) of successful
            application of regular and priority reaction rules.
        """
        x = self.embedder(batch, self.batch_size)
        y = self.y_predictor(x)

        if self.policy_type == "ranking":
            y = torch.softmax(y, dim=-1)
            return y

        if self.policy_type == "filtering":
            y = torch.sigmoid(y)
            priority = torch.sigmoid(self.priority_predictor(x))
            return y, priority

    def _get_loss(self, batch: Batch) -> Dict[str, Tensor]:
        """Calculates the loss and various classification metrics for a given batch for
        reaction rules prediction.

        :param batch: The batch of molecular graphs.
        :return: A dictionary with loss value and balanced accuracy of reaction rules
            prediction.
        """
        true_y = batch.y_rules.long()
        x = self.embedder(batch, self.batch_size)
        pred_y = self.y_predictor(x)

        if self.policy_type == "ranking":
            true_one_hot = one_hot(true_y, num_classes=self.n_rules)
            loss = cross_entropy(pred_y, true_one_hot.float())
            ba_y = (
                recall(pred_y, true_y, task="multiclass", num_classes=self.n_rules)
                + specificity(
                    pred_y, true_y, task="multiclass", num_classes=self.n_rules
                )
            ) / 2
            f1_y = f1_score(pred_y, true_y, task="multiclass", num_classes=self.n_rules)

            metrics = {"loss": loss, "balanced_accuracy_y": ba_y, "f1_score_y": f1_y}

        elif self.policy_type == "filtering":
            loss_y = binary_cross_entropy_with_logits(pred_y, true_y.float())

            ba_y = (
                recall(pred_y, true_y, task="multilabel", num_labels=self.n_rules)
                + specificity(
                    pred_y, true_y, task="multilabel", num_labels=self.n_rules
                )
            ) / 2

            f1_y = f1_score(pred_y, true_y, task="multilabel", num_labels=self.n_rules)

            true_priority = batch.y_priority.float()
            pred_priority = self.priority_predictor(x)
            loss_priority = binary_cross_entropy_with_logits(
                pred_priority, true_priority
            )

            loss = loss_y + loss_priority

            true_priority = true_priority.long()
            ba_priority = (
                recall(
                    pred_priority,
                    true_priority,
                    task="multilabel",
                    num_labels=self.n_rules,
                )
                + specificity(
                    pred_priority,
                    true_priority,
                    task="multilabel",
                    num_labels=self.n_rules,
                )
            ) / 2

            f1_priority = f1_score(
                pred_priority, true_priority, task="multilabel", num_labels=self.n_rules
            )

            metrics = {
                "loss": loss,
                "balanced_accuracy_y": ba_y,
                "f1_score_y": f1_y,
                "balanced_accuracy_priority": ba_priority,
                "f1_score_priority": f1_priority,
            }

        return metrics
