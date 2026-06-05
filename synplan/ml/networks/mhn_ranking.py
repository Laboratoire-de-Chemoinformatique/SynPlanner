"""Modern-Hopfield-style ranking policy implemented with SynPlanner components.

The architecture is inspired by MHNreact:
https://github.com/ml-jku/mhn-react
https://doi.org/10.1021/acs.jcim.1c01065
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor
from torch.nn import Dropout, Identity, LayerNorm, Linear, Sequential
from torch.nn.functional import cross_entropy
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torchmetrics.functional.classification import (
    f1_score,
    multiclass_accuracy,
    recall,
    specificity,
)

from synplan.chem.reaction_rules.graphs import (
    RULE_GRAPH_EDGE_FEATURE_DIM,
    RULE_GRAPH_NODE_FEATURE_DIM,
    RULE_GRAPH_SCHEMA_VERSION,
    query_cgr_graphs_from_smarts,
)
from synplan.chem.reaction_rules.representations import (
    RuleRepresentationConfig,
    rule_representation_digest,
)
from synplan.chem.reaction_rules.rule_fingerprints import (
    RULE_FINGERPRINT_SCHEMA_VERSION,
    RuleFingerprintConfig,
    load_rule_smarts,
    reaction_rules_path_from_policy_data,
    rule_fingerprints_from_smarts,
)
from synplan.ml.networks.modules import MCTSNetwork, build_graph_embedder

if TYPE_CHECKING:
    from synplan.ml.training.preprocessing import RankingPolicyDataset
    from synplan.utils.config import PolicyNetworkConfig

_MHN_CONFIG_FIELDS = {
    "architecture",
    "mhn_association_dim",
    "mhn_beta",
    "mhn_rule_encoder_type",
    "mhn_rule_embedder_type",
    "mhn_rule_graph_batch_size",
    "mhn_rule_graph_schema_version",
    "mhn_rule_vector_dim",
    "mhn_rule_num_conv_layers",
    "mhn_rule_heads",
    "mhn_rule_attn_type",
    "mhn_rule_dropout",
    "mhn_rule_attn_dropout",
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
        rule_graphs: Sequence[Data] | None = None,
        policy_data_path: str | Path | None = None,
        training_labels: Tensor | None = None,
        architecture: str = "mhn_ranking",
        policy_type: str = "ranking",
        mhn_association_dim: int = 512,
        mhn_beta: float = 0.05,
        mhn_rule_encoder_type: Literal[
            "fingerprint", "query_cgr_graph"
        ] = "fingerprint",
        mhn_rule_embedder_type: Literal["gcn", "gcn_concat", "gps"] = "gps",
        mhn_rule_graph_batch_size: int = 1024,
        mhn_rule_graph_schema_version: str = RULE_GRAPH_SCHEMA_VERSION,
        mhn_rule_vector_dim: int | None = None,
        mhn_rule_num_conv_layers: int | None = None,
        mhn_rule_heads: int | None = None,
        mhn_rule_attn_type: Literal["performer", "multihead"] | None = None,
        mhn_rule_dropout: float | None = None,
        mhn_rule_attn_dropout: float | None = None,
        mhn_rule_fp_size: int = 2048,
        mhn_rule_fp_min_radius: int = 1,
        mhn_rule_fp_max_radius: int = 4,
        mhn_rule_fp_active_bits: int = 2,
        mhn_rule_fp_type: Literal["legacy", "query_cgr"] = "query_cgr",
        mhn_rule_fp_schema_version: str = RULE_FINGERPRINT_SCHEMA_VERSION,
        mhn_normalize_associations: bool = True,
        mhn_rule_representation_digest: str | None = None,
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
        rule_representation_config = RuleRepresentationConfig(
            encoder_type=mhn_rule_encoder_type,
            fingerprint_config=rule_fingerprint_config,
            graph_schema_version=mhn_rule_graph_schema_version,
            graph_embedder_type=mhn_rule_embedder_type,
            graph_batch_size=mhn_rule_graph_batch_size,
        )

        if policy_data_path is not None:
            reaction_rules_path = reaction_rules_path_from_policy_data(policy_data_path)
            rule_smarts = load_rule_smarts(reaction_rules_path)
            n_rules = len(rule_smarts)
            mhn_rule_representation_digest = rule_representation_digest(
                rule_smarts, rule_representation_config
            )
            if training_labels is not None:
                labels = training_labels.view(-1)
                if labels.numel() and (labels.min() < 0 or labels.max() >= n_rules):
                    raise ValueError(
                        "Ranking policy labels must be within the inferred "
                        f"reaction-rule range [0, {n_rules - 1}]"
                    )
            if rule_representation_config.encoder_type == "fingerprint":
                if rule_fingerprints is None:
                    rule_fingerprints = rule_fingerprints_from_smarts(
                        rule_smarts, rule_fingerprint_config
                    )
            elif rule_graphs is None:
                rule_graphs = query_cgr_graphs_from_smarts(
                    rule_smarts,
                    schema_version=rule_representation_config.graph_schema_version,
                )

        super().__init__(vector_dim, *args, **kwargs)
        self.save_hyperparameters(
            ignore=[
                "rule_fingerprints",
                "rule_graphs",
                "policy_data_path",
                "training_labels",
            ]
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
        self.mhn_rule_encoder_type = rule_representation_config.encoder_type
        self.mhn_rule_embedder_type = rule_representation_config.graph_embedder_type
        self.mhn_rule_graph_batch_size = rule_representation_config.graph_batch_size
        self.mhn_rule_graph_schema_version = (
            rule_representation_config.graph_schema_version
        )
        self.mhn_rule_vector_dim = mhn_rule_vector_dim
        self.mhn_rule_num_conv_layers = mhn_rule_num_conv_layers
        self.mhn_rule_heads = mhn_rule_heads
        self.mhn_rule_attn_type = mhn_rule_attn_type
        self.mhn_rule_dropout = mhn_rule_dropout
        self.mhn_rule_attn_dropout = mhn_rule_attn_dropout
        self.mhn_rule_fp_size = rule_fingerprint_config.fp_size
        self.mhn_rule_fp_min_radius = rule_fingerprint_config.min_radius
        self.mhn_rule_fp_max_radius = rule_fingerprint_config.max_radius
        self.mhn_rule_fp_active_bits = rule_fingerprint_config.active_bits
        self.mhn_rule_fp_type = rule_fingerprint_config.fp_type
        self.mhn_rule_fp_schema_version = rule_fingerprint_config.schema_version
        self.mhn_rule_representation_digest = mhn_rule_representation_digest

        def normalization():
            if mhn_normalize_associations:
                return LayerNorm(mhn_association_dim, elementwise_affine=False)
            return Identity()

        dropout = kwargs.get("dropout", 0.4)
        rule_vector_dim = (
            vector_dim if mhn_rule_vector_dim is None else mhn_rule_vector_dim
        )
        rule_num_conv_layers = (
            kwargs.get("num_conv_layers", 5)
            if mhn_rule_num_conv_layers is None
            else mhn_rule_num_conv_layers
        )
        rule_heads = (
            kwargs.get("heads", 4) if mhn_rule_heads is None else mhn_rule_heads
        )
        rule_attn_type = (
            kwargs.get("attn_type", "performer")
            if mhn_rule_attn_type is None
            else mhn_rule_attn_type
        )
        rule_dropout = dropout if mhn_rule_dropout is None else mhn_rule_dropout
        rule_attn_dropout = (
            kwargs.get("attn_dropout", 0.5)
            if mhn_rule_attn_dropout is None
            else mhn_rule_attn_dropout
        )
        self.molecule_encoder = Sequential(
            Linear(vector_dim, mhn_association_dim),
            Dropout(dropout),
            normalization(),
        )
        if self.mhn_rule_encoder_type == "fingerprint":
            self.rule_embedder = None
            self.rule_encoder = Sequential(
                Linear(rule_fingerprint_config.fp_size, mhn_association_dim),
                Dropout(rule_dropout),
                normalization(),
            )
        else:
            self.rule_embedder = build_graph_embedder(
                self.mhn_rule_embedder_type,
                rule_vector_dim,
                dropout=rule_dropout,
                num_conv_layers=rule_num_conv_layers,
                heads=rule_heads,
                attn_type=rule_attn_type,
                attn_dropout=rule_attn_dropout,
                node_dim=RULE_GRAPH_NODE_FEATURE_DIM,
                edge_dim=RULE_GRAPH_EDGE_FEATURE_DIM,
            )
            self.rule_encoder = Sequential(
                Linear(rule_vector_dim, mhn_association_dim),
                Dropout(rule_dropout),
                normalization(),
            )
        self.register_buffer(
            "_training_rule_fingerprints",
            torch.empty((0, rule_fingerprint_config.fp_size), dtype=torch.float),
            persistent=False,
        )
        self._training_rule_graphs: list[Data] = []
        if rule_fingerprints is not None:
            self.set_training_rule_fingerprints(rule_fingerprints)
        if rule_graphs is not None:
            self.set_training_rule_graphs(rule_graphs)

    def bind_training_rules_from_policy_data(
        self,
        policy_data_path: str | Path,
        *,
        training_labels: Tensor | None = None,
    ) -> None:
        """Attach rule representations inferred from a ranking policy mapping."""
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

        rule_fingerprint_config = RuleFingerprintConfig(
            fp_size=self.mhn_rule_fp_size,
            min_radius=self.mhn_rule_fp_min_radius,
            max_radius=self.mhn_rule_fp_max_radius,
            active_bits=self.mhn_rule_fp_active_bits,
            fp_type=self.mhn_rule_fp_type,
            schema_version=self.mhn_rule_fp_schema_version,
        )
        rule_representation_config = RuleRepresentationConfig(
            encoder_type=self.mhn_rule_encoder_type,
            fingerprint_config=rule_fingerprint_config,
            graph_schema_version=self.mhn_rule_graph_schema_version,
            graph_embedder_type=self.mhn_rule_embedder_type,
            graph_batch_size=self.mhn_rule_graph_batch_size,
        )
        self.n_rules = n_rules
        self.mhn_rule_representation_digest = rule_representation_digest(
            rule_smarts, rule_representation_config
        )
        self.hparams["n_rules"] = n_rules
        self.hparams["mhn_rule_representation_digest"] = (
            self.mhn_rule_representation_digest
        )
        self._training_rule_fingerprints = torch.empty(
            (0, self.mhn_rule_fp_size), dtype=torch.float, device=self.device
        )
        self._training_rule_graphs = []
        if self.mhn_rule_encoder_type == "fingerprint":
            self.set_training_rule_fingerprints(
                rule_fingerprints_from_smarts(rule_smarts, rule_fingerprint_config)
            )
        else:
            self.set_training_rule_graphs(
                query_cgr_graphs_from_smarts(
                    rule_smarts, schema_version=self.mhn_rule_graph_schema_version
                )
            )

    def set_training_rule_fingerprints(self, rule_fingerprints: Tensor) -> None:
        """Attach ordered training rule fingerprints without storing them."""
        if self.mhn_rule_encoder_type != "fingerprint":
            raise ValueError(
                "Rule fingerprints require mhn_rule_encoder_type='fingerprint'"
            )
        expected_shape = (self.n_rules, self.mhn_rule_fp_size)
        if tuple(rule_fingerprints.shape) != expected_shape:
            raise ValueError(
                f"Expected rule fingerprints with shape {expected_shape}, "
                f"got {tuple(rule_fingerprints.shape)}"
            )
        self._training_rule_fingerprints = rule_fingerprints.float()

    def set_training_rule_graphs(self, rule_graphs: Sequence[Data]) -> None:
        """Attach ordered training rule graphs without storing them in checkpoints."""
        if self.mhn_rule_encoder_type != "query_cgr_graph":
            raise ValueError(
                "Rule graphs require mhn_rule_encoder_type='query_cgr_graph'"
            )
        if len(rule_graphs) != self.n_rules:
            raise ValueError(
                f"Expected {self.n_rules} rule graphs, got {len(rule_graphs)}"
            )
        self._training_rule_graphs = list(rule_graphs)

    def encode_rule_graphs(self, rule_graphs: Sequence[Data]) -> Tensor:
        """Embed QueryCGR rule graphs into the association space."""
        if self.rule_embedder is None:
            raise ValueError(
                "Rule graph encoding requires mhn_rule_encoder_type='query_cgr_graph'"
            )
        if not rule_graphs:
            return torch.empty(
                (0, self.rule_encoder[0].out_features), device=self.device
            )

        encoded_chunks = []
        batch_size = self.mhn_rule_graph_batch_size
        for start in range(0, len(rule_graphs), batch_size):
            rule_batch = Batch.from_data_list(
                list(rule_graphs[start : start + batch_size])
            ).to(self.device)
            encoded_chunks.append(self.rule_encoder(self.rule_embedder(rule_batch)))
        return torch.cat(encoded_chunks, dim=0)

    def encode_rules(self, rule_representations: Tensor | Sequence[Data]) -> Tensor:
        """Project raw rule representations into the association space."""
        if self.mhn_rule_encoder_type == "fingerprint":
            if not torch.is_tensor(rule_representations):
                raise ValueError("Fingerprint rule encoding requires a tensor input")
            return self.rule_encoder(rule_representations.float())
        if torch.is_tensor(rule_representations):
            raise ValueError("QueryCGR graph rule encoding requires graph inputs")
        return self.encode_rule_graphs(rule_representations)

    def encode_molecules(self, batch: Batch) -> Tensor:
        """Project molecular graph embeddings into the association space."""
        return self.molecule_encoder(self.embedder(batch))

    def get_logits(
        self,
        batch: Batch,
        *,
        rule_fingerprints: Tensor | None = None,
        rule_graphs: Sequence[Data] | None = None,
        rule_associations: Tensor | None = None,
    ) -> Tensor:
        """Calculate dense molecule-rule association logits."""
        if rule_associations is None:
            if self.mhn_rule_encoder_type == "fingerprint":
                if rule_fingerprints is None:
                    rule_fingerprints = self._training_rule_fingerprints
                if rule_fingerprints.numel() == 0:
                    raise ValueError("No rule fingerprints are attached or bound")
                rule_associations = self.encode_rules(rule_fingerprints)
            else:
                if rule_graphs is None:
                    rule_graphs = self._training_rule_graphs
                if not rule_graphs:
                    raise ValueError("No rule graphs are attached or bound")
                rule_associations = self.encode_rule_graphs(rule_graphs)
        molecules = self.encode_molecules(batch)
        return self.mhn_beta * molecules @ rule_associations.T

    def forward(
        self,
        batch: Batch,
        *,
        rule_fingerprints: Tensor | None = None,
        rule_graphs: Sequence[Data] | None = None,
        rule_associations: Tensor | None = None,
    ) -> Tensor:
        """Return probabilities for all attached or supplied rules."""
        return torch.softmax(
            self.get_logits(
                batch,
                rule_fingerprints=rule_fingerprints,
                rule_graphs=rule_graphs,
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
