"""Lightning trainer subclasses owning per-network loss and metrics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, one_hot
from torch_geometric.data.batch import Batch
from torchmetrics.functional.classification import (
    binary_f1_score,
    binary_recall,
    binary_specificity,
    f1_score,
    multiclass_accuracy,
    recall,
    specificity,
)

from synplan.chem.reaction.rules.representation import (
    load_rule_smarts,
    reaction_rules_path_from_policy_data,
    rule_representation_digest,
)
from synplan.ml.featurization.fingerprints import rule_fingerprints_from_smarts
from synplan.ml.featurization.rules import query_cgr_graphs_from_smarts
from synplan.ml.networks.mhn_ranking import MHNRankingNetwork
from synplan.ml.networks.policy import (
    FilteringPolicyNetwork,
    RankingPolicyNetwork,
)
from synplan.ml.networks.value import ValueNetwork
from synplan.ml.training.lightning import LitNetworkTrainer
from synplan.utils.config import (
    LinearPolicyNetworkConfig,
    MHNRankingPolicyNetworkConfig,
)

if TYPE_CHECKING:
    from synplan.ml.training.preprocessing import RankingPolicyDataset


class LitValue(LitNetworkTrainer):
    """Trains a :class:`ValueNetwork` (BCE-with-logits + balanced accuracy + f1)."""

    def __init__(self, network: ValueNetwork) -> None:
        super().__init__(
            network, learning_rate=network.lr, batch_size=network.batch_size
        )

    def compute_loss(self, batch: Batch) -> dict[str, Tensor]:
        """Loss and classification metrics for synthesisability prediction."""
        true_y = torch.unsqueeze(batch.y.float(), -1)
        x = self.network.embedder(batch)
        pred_y = self.network.predictor(x)
        loss = binary_cross_entropy_with_logits(pred_y, true_y)

        true_y = true_y.long()
        ba = (binary_recall(pred_y, true_y) + binary_specificity(pred_y, true_y)) / 2
        f1 = binary_f1_score(pred_y, true_y)
        return {"loss": loss, "balanced_accuracy": ba, "f1_score": f1}


class LitRankingPolicy(LitNetworkTrainer):
    """Trains a :class:`RankingPolicyNetwork` (softmax cross-entropy + metrics)."""

    def __init__(self, network: RankingPolicyNetwork) -> None:
        super().__init__(
            network, learning_rate=network.lr, batch_size=network.batch_size
        )

    @classmethod
    def from_config(
        cls, config: LinearPolicyNetworkConfig, n_rules: int
    ) -> LitRankingPolicy:
        """Build a ranking policy network from a config and wrap it for training."""
        return cls(RankingPolicyNetwork(config, n_rules))

    def compute_loss(self, batch: Batch) -> dict[str, Tensor]:
        """Cross-entropy loss and ranking metrics for reaction-rule prediction."""
        net = self.network
        n_rules = net.n_rules
        true_y = batch.y_rules.long()
        x = net.head_dropout(net.embedder(batch))
        pred_y = net.y_predictor(x)

        true_one_hot = one_hot(true_y, num_classes=n_rules)
        loss = cross_entropy(pred_y, true_one_hot.float())
        ba_y = (
            recall(pred_y, true_y, task="multiclass", num_classes=n_rules)
            + specificity(pred_y, true_y, task="multiclass", num_classes=n_rules)
        ) / 2
        f1_y = f1_score(pred_y, true_y, task="multiclass", num_classes=n_rules)
        metrics = {"loss": loss, "balanced_accuracy_y": ba_y, "f1_score_y": f1_y}
        for k in (5, 10):
            if n_rules > k:
                metrics[f"top{k}_accuracy_y"] = multiclass_accuracy(
                    pred_y, true_y, num_classes=n_rules, top_k=k
                )
        return metrics


class LitFilteringPolicy(LitNetworkTrainer):
    """Trains a :class:`FilteringPolicyNetwork` (BCE on rule + priority heads)."""

    def __init__(self, network: FilteringPolicyNetwork) -> None:
        super().__init__(
            network, learning_rate=network.lr, batch_size=network.batch_size
        )

    @classmethod
    def from_config(
        cls, config: LinearPolicyNetworkConfig, n_rules: int
    ) -> LitFilteringPolicy:
        """Build a filtering policy network from a config and wrap it for training."""
        return cls(FilteringPolicyNetwork(config, n_rules))

    def compute_loss(self, batch: Batch) -> dict[str, Tensor]:
        """BCE loss + metrics for the rule and priority filtering heads."""
        net = self.network
        n_rules = net.n_rules
        true_y = batch.y_rules.long()
        x = net.head_dropout(net.embedder(batch))
        pred_y = net.y_predictor(x)

        loss_y = binary_cross_entropy_with_logits(pred_y, true_y.float())
        ba_y = (
            recall(pred_y, true_y, task="multilabel", num_labels=n_rules)
            + specificity(pred_y, true_y, task="multilabel", num_labels=n_rules)
        ) / 2
        f1_y = f1_score(pred_y, true_y, task="multilabel", num_labels=n_rules)

        true_priority = batch.y_priority.float()
        pred_priority = net.priority_predictor(x)
        loss_priority = binary_cross_entropy_with_logits(pred_priority, true_priority)
        loss = loss_y + loss_priority

        true_priority = true_priority.long()
        ba_priority = (
            recall(pred_priority, true_priority, task="multilabel", num_labels=n_rules)
            + specificity(
                pred_priority, true_priority, task="multilabel", num_labels=n_rules
            )
        ) / 2
        f1_priority = f1_score(
            pred_priority, true_priority, task="multilabel", num_labels=n_rules
        )
        return {
            "loss": loss,
            "balanced_accuracy_y": ba_y,
            "f1_score_y": f1_y,
            "balanced_accuracy_priority": ba_priority,
            "f1_score_priority": f1_priority,
        }


class LitMHNRanking(LitNetworkTrainer):
    """Trains an :class:`MHNRankingNetwork` (softmax cross-entropy + metrics)."""

    def __init__(self, network: MHNRankingNetwork) -> None:
        super().__init__(
            network, learning_rate=network.lr, batch_size=network.batch_size
        )

    @classmethod
    def from_config(
        cls,
        config: MHNRankingPolicyNetworkConfig,
        dataset: RankingPolicyDataset,
    ) -> LitMHNRanking:
        """Build an MHN network with rules inferred from extracted policy data."""
        return cls(build_mhn_ranking_network(config=config, dataset=dataset))

    def compute_loss(self, batch: Batch) -> dict[str, Tensor]:
        """Cross-entropy ranking loss and the same metrics as the linear policy."""
        net = self.network
        n_rules = net.n_rules
        true_y = batch.y_rules.long().view(-1)
        pred_y = net.get_logits(batch)
        loss = cross_entropy(pred_y, true_y)
        ba_y = (
            recall(pred_y, true_y, task="multiclass", num_classes=n_rules)
            + specificity(pred_y, true_y, task="multiclass", num_classes=n_rules)
        ) / 2
        metrics = {
            "loss": loss,
            "balanced_accuracy_y": ba_y,
            "f1_score_y": f1_score(
                pred_y, true_y, task="multiclass", num_classes=n_rules
            ),
        }
        for k in (5, 10):
            if n_rules > k:
                metrics[f"top{k}_accuracy_y"] = multiclass_accuracy(
                    pred_y, true_y, num_classes=n_rules, top_k=k
                )
        return metrics


def build_mhn_ranking_network(
    config: MHNRankingPolicyNetworkConfig,
    dataset: RankingPolicyDataset,
) -> MHNRankingNetwork:
    """Create an MHN network and attach training rules from policy data.

    File IO + rule featurization live here (training-side), not in the network.
    """
    rule_smarts = load_rule_smarts(
        reaction_rules_path_from_policy_data(dataset.policy_data_path)
    )
    n_rules = len(rule_smarts)
    validate_ranking_labels(dataset._data.y_rules, n_rules)
    network = MHNRankingNetwork(config=config, n_rules=n_rules)
    attach_training_rules(network, rule_smarts)
    return network


def attach_training_rules(network: MHNRankingNetwork, rule_smarts) -> None:
    """Featurize ``rule_smarts`` and attach them to ``network`` for training."""
    representation_config = network.rule_representation_config
    network.n_rules = len(rule_smarts)
    digest = rule_representation_digest(rule_smarts, representation_config)
    network.rule_representation_digest = digest
    network.hparams["n_rules"] = network.n_rules
    network.hparams["rule_representation_digest"] = digest
    if representation_config.encoder_type == "fingerprint":
        network._training_rule_graphs = []
        network.set_training_rule_fingerprints(
            rule_fingerprints_from_smarts(
                rule_smarts, representation_config.fingerprint_config
            )
        )
    else:
        network._training_rule_fingerprints = torch.empty(
            (0, representation_config.fingerprint_config.fp_size), dtype=torch.float
        )
        network.set_training_rule_graphs(
            query_cgr_graphs_from_smarts(
                rule_smarts,
                schema_version=representation_config.graph_schema_version,
            )
        )


def rebind_training_rules(
    network: MHNRankingNetwork, policy_data_path: str | Path
) -> None:
    """Re-attach training rules inferred from a new ranking policy mapping."""
    rule_smarts = load_rule_smarts(
        reaction_rules_path_from_policy_data(policy_data_path)
    )
    attach_training_rules(network, rule_smarts)


def validate_ranking_labels(training_labels: Tensor | None, n_rules: int) -> None:
    if training_labels is None:
        return
    labels = training_labels.view(-1)
    if labels.numel() and (labels.min() < 0 or labels.max() >= n_rules):
        raise ValueError(
            "Ranking policy labels must be within the inferred "
            f"reaction-rule range [0, {n_rules - 1}]"
        )
