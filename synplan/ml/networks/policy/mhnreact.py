"""Modern-Hopfield-style ranking policy network (pure ``nn.Module``).

The architecture is inspired by MHNreact:
https://github.com/ml-jku/mhn-react
https://doi.org/10.1021/acs.jcim.1c01065
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import Dropout, Identity, LayerNorm, Linear, Sequential
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from synplan.ml.config import MHNRankingPolicyNetworkConfig
from synplan.ml.networks.base import GraphMCTSNetwork
from synplan.ml.networks.embedding.rule import (
    FingerprintRuleEmbedding,
    QueryCGRRuleEmbedding,
)


class MHNReact(GraphMCTSNetwork):
    """Ranking policy associating molecule embeddings with rule embeddings."""

    architecture = "mhn_ranking"
    policy_type = "ranking"
    CONFIG_CLASS = MHNRankingPolicyNetworkConfig

    def __init__(self, config: MHNRankingPolicyNetworkConfig, n_rules: int) -> None:
        """Build embedder, molecule projection and the rule embedding.

        File IO and rule featurization live in the loading factory / trainer; the
        network only holds layers and runs forward.

        :param config: MHN architecture and training configuration.
        :param n_rules: The number of training rules (output dimension).
        """
        emb = config.rule_embedder
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
        self.beta = config.beta
        self.rule_representation_digest: str | None = None

        rule_repr_config = config.rule_representation_config()
        self.rule_representation_config = rule_repr_config

        # Rule-side embedder knobs fall back to the molecule-side values when unset.
        rule_vector_dim = (
            emb.vector_dim if emb.vector_dim is not None else config.vector_dim
        )
        rule_num_conv_layers = (
            emb.num_conv_layers
            if emb.num_conv_layers is not None
            else config.num_conv_layers
        )
        rule_heads = emb.heads if emb.heads is not None else config.heads
        rule_attn_type = (
            emb.attn_type if emb.attn_type is not None else config.attn_type
        )
        rule_dropout = emb.dropout if emb.dropout is not None else config.dropout
        rule_attn_dropout = (
            emb.attn_dropout if emb.attn_dropout is not None else config.attn_dropout
        )

        def _normalization():
            if config.normalize_associations:
                return LayerNorm(config.association_dim, elementwise_affine=False)
            return Identity()

        self.molecule_embedding = Sequential(
            Linear(config.vector_dim, config.association_dim),
            Dropout(config.dropout),
            _normalization(),
        )
        if rule_repr_config.embedding_type == "fingerprint":
            self.rule_embedding = FingerprintRuleEmbedding(
                rule_repr_config.fingerprint_config.fp_size,
                config.association_dim,
                rule_dropout,
                config.normalize_associations,
            )
        else:
            self.rule_embedding = QueryCGRRuleEmbedding(
                rule_vector_dim,
                config.association_dim,
                rule_dropout,
                config.normalize_associations,
                rule_repr_config.graph_batch_size,
                emb.embedder_type,
                rule_num_conv_layers,
                rule_heads,
                rule_attn_type,
                rule_attn_dropout,
            )

        self.register_buffer(
            "_training_rule_fingerprints",
            torch.empty(
                (0, rule_repr_config.fingerprint_config.fp_size), dtype=torch.float
            ),
            persistent=False,
        )
        self._training_rule_graphs: list[Data] = []
        self.hparams: dict[str, Any] = {
            "config": config.model_dump(),
            "n_rules": n_rules,
        }

    @property
    def rule_embedder(self):
        """Expose the rule-side graph embedder (``None`` for fingerprint embeddings)."""
        return getattr(self.rule_embedding, "embedder", None)

    def set_training_rule_fingerprints(self, rule_fingerprints: Tensor) -> None:
        """Attach ordered training rule fingerprints without storing them."""
        if self.rule_representation_config.embedding_type != "fingerprint":
            raise ValueError(
                "Rule fingerprints require rule_embedding_type='fingerprint'"
            )
        expected_shape = (
            self.n_rules,
            self.rule_representation_config.fingerprint_config.fp_size,
        )
        if tuple(rule_fingerprints.shape) != expected_shape:
            raise ValueError(
                f"Expected rule fingerprints with shape {expected_shape}, "
                f"got {tuple(rule_fingerprints.shape)}"
            )
        self._training_rule_fingerprints = rule_fingerprints.float()

    def set_training_rule_graphs(self, rule_graphs: Sequence[Data]) -> None:
        """Attach ordered training rule graphs without storing them in checkpoints."""
        if self.rule_representation_config.embedding_type != "query_cgr_graph":
            raise ValueError(
                "Rule graphs require rule_embedding_type='query_cgr_graph'"
            )
        if len(rule_graphs) != self.n_rules:
            raise ValueError(
                f"Expected {self.n_rules} rule graphs, got {len(rule_graphs)}"
            )
        self._training_rule_graphs = list(rule_graphs)

    def encode_rule_graphs(self, rule_graphs: Sequence[Data]) -> Tensor:
        """Embed QueryCGR rule graphs into the association space."""
        if not isinstance(self.rule_embedding, QueryCGRRuleEmbedding):
            raise ValueError(
                "Rule graph encoding requires rule_embedding_type='query_cgr_graph'"
            )
        return self.rule_embedding.encode(rule_graphs)

    def encode_rules(self, rule_representations: Tensor | Sequence[Data]) -> Tensor:
        """Project raw rule representations into the association space."""
        return self.rule_embedding.encode(rule_representations)

    def encode_molecules(self, batch: Batch) -> Tensor:
        """Project molecular graph embeddings into the association space."""
        return self.molecule_embedding(self.embedder(batch))

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
            if self.rule_representation_config.embedding_type == "fingerprint":
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
        return self.beta * molecules @ rule_associations.T

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
