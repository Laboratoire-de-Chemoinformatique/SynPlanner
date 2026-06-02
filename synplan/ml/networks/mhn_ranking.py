"""Modern-Hopfield-style ranking policy implemented with SynPlanner components.

The architecture is inspired by MHNreact:
https://github.com/ml-jku/mhn-react
https://doi.org/10.1021/acs.jcim.1c01065
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import torch
from torch import Tensor
from torch.nn import Dropout, Identity, LayerNorm, Linear, Sequential
from torch.nn.functional import cross_entropy
from torch_geometric.data.batch import Batch
from torchmetrics.functional.classification import (
    f1_score,
    multiclass_accuracy,
    recall,
    specificity,
)

from synplan.ml.networks.modules import MCTSNetwork
from synplan.ml.template_features import (
    load_rule_smarts,
    reaction_rules_path_from_policy_data,
    template_features_from_smarts,
)

if TYPE_CHECKING:
    from synplan.ml.training.preprocessing import RankingPolicyDataset
    from synplan.utils.config import PolicyNetworkConfig

_MHN_CONFIG_FIELDS = {
    "architecture",
    "mhn_association_dim",
    "mhn_beta",
    "mhn_template_fp_size",
    "mhn_template_fp_min_radius",
    "mhn_template_fp_max_radius",
    "mhn_template_fp_active_bits",
    "mhn_normalize_associations",
}


class MHNRankingPolicyNetwork(MCTSNetwork):
    """Ranking policy that associates graph embeddings with template embeddings."""

    architecture = "mhn_ranking"
    policy_type = "ranking"

    @classmethod
    def for_training(
        cls,
        *,
        dataset: RankingPolicyDataset,
        config: PolicyNetworkConfig,
        **network_kwargs: Any,
    ) -> Self:
        """Create a network with templates inferred from extracted policy data."""
        return cls(
            **network_kwargs,
            **config.model_dump(include=_MHN_CONFIG_FIELDS),
            policy_data_path=dataset.policy_data_path,
            training_labels=dataset._data.y_rules,
        )

    def __init__(
        self,
        *args,
        n_rules: int,
        vector_dim: int,
        template_features: Tensor | None = None,
        policy_data_path: str | Path | None = None,
        training_labels: Tensor | None = None,
        architecture: str = "mhn_ranking",
        policy_type: str = "ranking",
        mhn_association_dim: int = 512,
        mhn_beta: float = 0.05,
        mhn_template_fp_size: int = 2048,
        mhn_template_fp_min_radius: int = 1,
        mhn_template_fp_max_radius: int = 4,
        mhn_template_fp_active_bits: int = 2,
        mhn_normalize_associations: bool = True,
        **kwargs,
    ):
        if training_labels is not None and policy_data_path is None:
            raise ValueError("training_labels requires policy_data_path")
        if policy_data_path is not None:
            reaction_rules_path = reaction_rules_path_from_policy_data(policy_data_path)
            rule_smarts = load_rule_smarts(reaction_rules_path)
            n_rules = len(rule_smarts)
            if training_labels is not None:
                labels = training_labels.view(-1)
                if labels.numel() and (labels.min() < 0 or labels.max() >= n_rules):
                    raise ValueError(
                        "Ranking policy labels must be within the inferred "
                        f"reaction-rule range [0, {n_rules - 1}]"
                    )
            if template_features is None:
                template_features = template_features_from_smarts(
                    rule_smarts,
                    fp_size=mhn_template_fp_size,
                    min_radius=mhn_template_fp_min_radius,
                    max_radius=mhn_template_fp_max_radius,
                    active_bits=mhn_template_fp_active_bits,
                )

        super().__init__(vector_dim, *args, **kwargs)
        self.save_hyperparameters(
            ignore=["template_features", "policy_data_path", "training_labels"]
        )
        if architecture != "mhn_ranking":
            raise ValueError(
                "MHNRankingPolicyNetwork requires architecture='mhn_ranking'"
            )
        if policy_type != "ranking":
            raise ValueError("MHNRankingPolicyNetwork requires policy_type='ranking'")
        self.policy_type = policy_type
        self.n_rules = n_rules
        self.mhn_beta = mhn_beta
        self.mhn_template_fp_size = mhn_template_fp_size
        self.mhn_template_fp_min_radius = mhn_template_fp_min_radius
        self.mhn_template_fp_max_radius = mhn_template_fp_max_radius
        self.mhn_template_fp_active_bits = mhn_template_fp_active_bits

        def normalization():
            if mhn_normalize_associations:
                return LayerNorm(mhn_association_dim, elementwise_affine=False)
            return Identity()

        dropout = kwargs.get("dropout", 0.4)
        self.molecule_encoder = Sequential(
            Linear(vector_dim, mhn_association_dim),
            Dropout(dropout),
            normalization(),
        )
        self.template_encoder = Sequential(
            Linear(mhn_template_fp_size, mhn_association_dim),
            Dropout(dropout),
            normalization(),
        )
        self.register_buffer(
            "_training_template_features",
            torch.empty((0, mhn_template_fp_size), dtype=torch.float),
            persistent=False,
        )
        if template_features is not None:
            self.set_training_template_features(template_features)

    def set_training_template_features(self, template_features: Tensor) -> None:
        """Attach ordered training templates without storing them in checkpoints."""
        expected_shape = (self.n_rules, self.mhn_template_fp_size)
        if tuple(template_features.shape) != expected_shape:
            raise ValueError(
                f"Expected template features with shape {expected_shape}, "
                f"got {tuple(template_features.shape)}"
            )
        self._training_template_features = template_features.float()

    def encode_templates(self, template_features: Tensor) -> Tensor:
        """Project raw template fingerprints into the association space."""
        return self.template_encoder(template_features.float())

    def encode_molecules(self, batch: Batch) -> Tensor:
        """Project molecular graph embeddings into the association space."""
        return self.molecule_encoder(self.embedder(batch))

    def get_logits(
        self,
        batch: Batch,
        *,
        template_features: Tensor | None = None,
        template_associations: Tensor | None = None,
    ) -> Tensor:
        """Calculate dense molecule-template association logits."""
        if template_associations is None:
            if template_features is None:
                template_features = self._training_template_features
            if template_features.numel() == 0:
                raise ValueError("No template features are attached or bound")
            template_associations = self.encode_templates(template_features)
        molecules = self.encode_molecules(batch)
        return self.mhn_beta * molecules @ template_associations.T

    def forward(
        self,
        batch: Batch,
        *,
        template_features: Tensor | None = None,
        template_associations: Tensor | None = None,
    ) -> Tensor:
        """Return probabilities for all attached or supplied templates."""
        return torch.softmax(
            self.get_logits(
                batch,
                template_features=template_features,
                template_associations=template_associations,
            ),
            dim=-1,
        )

    def _get_loss(self, batch: Batch) -> dict[str, Tensor]:
        """Calculate ranking loss and the same metrics as the linear policy."""
        true_y = batch.y_rules.long().view(-1)
        pred_y = self.get_logits(batch)
        loss = cross_entropy(pred_y, true_y)
        ba_y = (
            recall(pred_y, true_y, task="multiclass", num_classes=self.n_rules)
            + specificity(pred_y, true_y, task="multiclass", num_classes=self.n_rules)
        ) / 2
        metrics = {
            "loss": loss,
            "balanced_accuracy_y": ba_y,
            "f1_score_y": f1_score(
                pred_y, true_y, task="multiclass", num_classes=self.n_rules
            ),
        }
        for k in (5, 10):
            if self.n_rules > k:
                metrics[f"top{k}_accuracy_y"] = multiclass_accuracy(
                    pred_y, true_y, num_classes=self.n_rules, top_k=k
                )
        return metrics
