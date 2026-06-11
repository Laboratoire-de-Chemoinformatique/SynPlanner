"""Base class shared by SynPlanner policy and value neural networks."""

from abc import ABC, abstractmethod

from adabelief_pytorch import AdaBelief
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data.batch import Batch

from synplan.ml.networks.embedders import build_graph_embedder


class MCTSNetwork(LightningModule, ABC):
    """Basic class for policy and value networks."""

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
        """The forward function takes a batch of input data and performs forward
        propagation through the neural network.

        :param batch: The batch of molecular graphs processed together in a single
            forward pass through the neural network.
        """

    @abstractmethod
    def _get_loss(self, batch: Batch) -> Tensor:
        """Calculate the loss for a given batch of data.

        :param batch: The batch of input data that is used to compute the loss.
        """

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        """Calculates the loss for a given training batch and logs the loss value.

        :param batch: The batch of data that is used for training.
        :param batch_idx: The index of the batch.
        :return: The value of the training loss.
        """
        metrics = self._get_loss(batch)
        for name, value in metrics.items():
            self.log(
                "train_" + name,
                value,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )
        return metrics["loss"]

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Calculates the loss for a given validation batch and logs the loss value.

        :param batch: The batch of data that is used for validation.
        :param batch_idx: The index of the batch.
        """
        metrics = self._get_loss(batch)
        for name, value in metrics.items():
            self.log("val_" + name, value, on_epoch=True, batch_size=self.batch_size)

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        """Calculates the loss for a given test batch and logs the loss value.

        :param batch: The batch of data that is used for testing.
        :param batch_idx: The index of the batch.
        """
        metrics = self._get_loss(batch)
        for name, value in metrics.items():
            self.log("test_" + name, value, on_epoch=True, batch_size=self.batch_size)

    def configure_optimizers(
        self,
    ) -> tuple[list[AdaBelief], list[dict[str, bool | str | ReduceLROnPlateau]]]:
        """Returns an optimizer and a learning rate scheduler for training a model using
        the AdaBelief optimizer and ReduceLROnPlateau scheduler.

        :return: The optimizer and a scheduler.
        """

        optimizer = AdaBelief(
            self.parameters(),
            lr=self.lr,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=True,
            rectify=True,
            weight_decay=0.01,
            print_change_log=False,
        )

        lr_scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.8, min_lr=5e-5)
        scheduler = {
            "scheduler": lr_scheduler,
            "reduce_on_plateau": True,
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler]
