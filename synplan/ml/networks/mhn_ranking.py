"""Modern-Hopfield-style ranking policy implemented with SynPlanner components.

The architecture is inspired by MHNreact:
https://github.com/ml-jku/mhn-react
https://doi.org/10.1021/acs.jcim.1c01065
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
from synplan.ml.rule_fingerprints import (
    RULE_FINGERPRINT_SCHEMA_VERSION,
    RuleFingerprintConfig,
    load_rule_smarts,
    reaction_rules_path_from_policy_data,
    rule_fingerprint_digest,
    rule_fingerprints_from_smarts,
)

if TYPE_CHECKING:
    from synplan.ml.training.preprocessing import RankingPolicyDataset
    from synplan.utils.config import PolicyNetworkConfig

_MHN_CONFIG_FIELDS = {
    "architecture",
    "mhn_association_dim",
    "mhn_beta",
    "mhn_rule_fp_size",
    "mhn_rule_fp_min_radius",
    "mhn_rule_fp_max_radius",
    "mhn_rule_fp_active_bits",
    "mhn_rule_fp_type",
    "mhn_rule_fp_schema_version",
    "mhn_normalize_associations",
}


class MHNRankingPolicyNetwork(MCTSNetwork):
    """Ranking policy that associates graph embeddings with rule embeddings."""

    architecture = "mhn_ranking"
    policy_type = "ranking"

    @classmethod
    def for_training(
        cls,
        *,
        dataset: RankingPolicyDataset,
        config: PolicyNetworkConfig,
        **network_kwargs: Any,
    ) -> MHNRankingPolicyNetwork:
        """Create a network with rules inferred from extracted policy data."""
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
        rule_fingerprints: Tensor | None = None,
        policy_data_path: str | Path | None = None,
        training_labels: Tensor | None = None,
        architecture: str = "mhn_ranking",
        policy_type: str = "ranking",
        mhn_association_dim: int = 512,
        mhn_beta: float = 0.05,
        mhn_rule_fp_size: int = 2048,
        mhn_rule_fp_min_radius: int = 1,
        mhn_rule_fp_max_radius: int = 4,
        mhn_rule_fp_active_bits: int = 2,
        mhn_rule_fp_type: Literal["legacy", "query_cgr"] = "query_cgr",
        mhn_rule_fp_schema_version: str = RULE_FINGERPRINT_SCHEMA_VERSION,
        mhn_normalize_associations: bool = True,
        mhn_rule_fingerprint_digest: str | None = None,
        **kwargs,
    ):
        if training_labels is not None and policy_data_path is None:
            raise ValueError("training_labels requires policy_data_path")
        rule_fingerprint_config = RuleFingerprintConfig(
            fp_size=mhn_rule_fp_size,
            min_radius=mhn_rule_fp_min_radius,
            max_radius=mhn_rule_fp_max_radius,
            active_bits=mhn_rule_fp_active_bits,
            fp_type=mhn_rule_fp_type,
            schema_version=mhn_rule_fp_schema_version,
        )
        if policy_data_path is not None:
            reaction_rules_path = reaction_rules_path_from_policy_data(policy_data_path)
            rule_smarts = load_rule_smarts(reaction_rules_path)
            n_rules = len(rule_smarts)
            mhn_rule_fingerprint_digest = rule_fingerprint_digest(
                rule_smarts, rule_fingerprint_config
            )
            if training_labels is not None:
                labels = training_labels.view(-1)
                if labels.numel() and (labels.min() < 0 or labels.max() >= n_rules):
                    raise ValueError(
                        "Ranking policy labels must be within the inferred "
                        f"reaction-rule range [0, {n_rules - 1}]"
                    )
            if rule_fingerprints is None:
                rule_fingerprints = rule_fingerprints_from_smarts(
                    rule_smarts, rule_fingerprint_config
                )

        super().__init__(vector_dim, *args, **kwargs)
        self.save_hyperparameters(
            ignore=["rule_fingerprints", "policy_data_path", "training_labels"]
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
        self.mhn_rule_fp_size = rule_fingerprint_config.fp_size
        self.mhn_rule_fp_min_radius = rule_fingerprint_config.min_radius
        self.mhn_rule_fp_max_radius = rule_fingerprint_config.max_radius
        self.mhn_rule_fp_active_bits = rule_fingerprint_config.active_bits
        self.mhn_rule_fp_type = rule_fingerprint_config.fp_type
        self.mhn_rule_fp_schema_version = rule_fingerprint_config.schema_version
        self.mhn_rule_fingerprint_digest = mhn_rule_fingerprint_digest

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
        self.rule_encoder = Sequential(
            Linear(rule_fingerprint_config.fp_size, mhn_association_dim),
            Dropout(dropout),
            normalization(),
        )
        self.register_buffer(
            "_training_rule_fingerprints",
            torch.empty((0, rule_fingerprint_config.fp_size), dtype=torch.float),
            persistent=False,
        )
        if rule_fingerprints is not None:
            self.set_training_rule_fingerprints(rule_fingerprints)

    def set_training_rule_fingerprints(self, rule_fingerprints: Tensor) -> None:
        """Attach ordered training rules without storing them in checkpoints."""
        expected_shape = (self.n_rules, self.mhn_rule_fp_size)
        if tuple(rule_fingerprints.shape) != expected_shape:
            raise ValueError(
                f"Expected rule fingerprints with shape {expected_shape}, "
                f"got {tuple(rule_fingerprints.shape)}"
            )
        self._training_rule_fingerprints = rule_fingerprints.float()

    def encode_rules(self, rule_fingerprints: Tensor) -> Tensor:
        """Project raw rule fingerprints into the association space."""
        return self.rule_encoder(rule_fingerprints.float())

    def encode_molecules(self, batch: Batch) -> Tensor:
        """Project molecular graph embeddings into the association space."""
        return self.molecule_encoder(self.embedder(batch))

    def get_logits(
        self,
        batch: Batch,
        *,
        rule_fingerprints: Tensor | None = None,
        rule_associations: Tensor | None = None,
    ) -> Tensor:
        """Calculate dense molecule-rule association logits."""
        if rule_associations is None:
            if rule_fingerprints is None:
                rule_fingerprints = self._training_rule_fingerprints
            if rule_fingerprints.numel() == 0:
                raise ValueError("No rule fingerprints are attached or bound")
            rule_associations = self.encode_rules(rule_fingerprints)
        molecules = self.encode_molecules(batch)
        return self.mhn_beta * molecules @ rule_associations.T

    def forward(
        self,
        batch: Batch,
        *,
        rule_fingerprints: Tensor | None = None,
        rule_associations: Tensor | None = None,
    ) -> Tensor:
        """Return probabilities for all attached or supplied rules."""
        return torch.softmax(
            self.get_logits(
                batch,
                rule_fingerprints=rule_fingerprints,
                rule_associations=rule_associations,
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
