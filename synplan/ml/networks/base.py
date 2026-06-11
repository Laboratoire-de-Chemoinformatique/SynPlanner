"""Pure ``nn.Module`` base for SynPlanner policy and value networks."""

from abc import abstractmethod

from torch import Tensor
from torch.nn import Module
from torch_geometric.data.batch import Batch

from synplan.ml.networks.embedders import build_graph_embedder


class GraphMCTSNetwork(Module):
    """Pure ``nn.Module`` base for MCTS graph networks (no training plumbing)."""

    def __init__(
        self,
        vector_dim: int,
        batch_size: int,
        dropout: float = 0.4,
        num_conv_layers: int = 5,
        learning_rate: float = 0.001,
        gcn_concat: bool = False,
        embedder_type: str = "gcn",
        heads: int = 4,
        attn_type: str = "performer",
        attn_dropout: float = 0.5,
    ):
        """The basic class for MCTS graph convolutional neural networks (policy and
        value network).

        :param vector_dim: The dimensionality of the hidden layers and output layer of
            graph convolution module.
        :param dropout: Dropout is a regularization technique used in neural networks to
            prevent overfitting.
        :param num_conv_layers: The number of convolutional layers in a graph
            convolutional module.
        :param learning_rate: The learning rate determines how quickly the model learns
            from the training data.
        :param gcn_concat: Legacy flag for concat embedder. Use embedder_type instead.
        :param embedder_type: Embedder architecture: "gcn", "gcn_concat", or "gps".
        :param heads: Number of attention heads (GPS only).
        :param attn_type: Attention type: "performer", "multihead", or None (GPS only).
        :param attn_dropout: Attention dropout probability (GPS only).
        """
        super().__init__()
        if gcn_concat and embedder_type != "gps":
            embedder_type = "gcn_concat"
        self.embedder = build_graph_embedder(
            embedder_type,
            vector_dim,
            dropout=dropout,
            num_conv_layers=num_conv_layers,
            heads=heads,
            attn_type=attn_type,
            attn_dropout=attn_dropout,
        )
        self.batch_size = batch_size
        self.lr = learning_rate

    @abstractmethod
    def forward(self, batch: Batch) -> Tensor:
        """Run forward propagation on a batch of molecular graphs.

        :param batch: The batch of molecular graphs processed together in a single
            forward pass through the neural network.
        """
