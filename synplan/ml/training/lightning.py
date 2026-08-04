"""Lightning trainer base wrapping a pure network and owning training plumbing."""

from abc import ABC, abstractmethod

from adabelief_pytorch import AdaBelief
from pytorch_lightning import Callback, LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data.batch import Batch


class GradNormLogger(Callback):
    """Logs mean gradient norm per top-level module (e.g. embedder, predictor)."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        norms = {}
        for name, p in pl_module.named_parameters():
            if p.grad is not None:
                tag = name.split(".")[0]
                norms.setdefault(tag, []).append(p.grad.norm().item())
        for tag, vals in norms.items():
            pl_module.log(f"grad_norm/{tag}", sum(vals) / len(vals))


class LitNetworkTrainer(LightningModule, ABC):
    """Lightning wrapper: owns steps/optimizer/metric logging, delegates loss."""

    def __init__(self, network: Module) -> None:
        """Wrap a pure network for training.

        :param network: The pure ``nn.Module`` to train; supplies ``lr`` and
            ``batch_size`` (both set by ``GraphMCTSNetwork.__init__``).
        """
        super().__init__()
        self.network = network
        self.lr = network.lr
        self.batch_size = network.batch_size

    @abstractmethod
    def compute_loss(self, batch: Batch) -> dict[str, Tensor]:
        """Return a metrics dict for the batch, with a ``"loss"`` entry."""

    def forward(self, batch: Batch) -> Tensor:
        return self.network(batch)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        metrics = self.compute_loss(batch)
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
        metrics = self.compute_loss(batch)
        for name, value in metrics.items():
            self.log("val_" + name, value, on_epoch=True, batch_size=self.batch_size)

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        metrics = self.compute_loss(batch)
        for name, value in metrics.items():
            self.log("test_" + name, value, on_epoch=True, batch_size=self.batch_size)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Keep the on-disk shape identical to a bare-network checkpoint.

        Strips the ``network.`` prefix Lightning adds and writes the pure
        network's hparams, so loader and trainer agree on the contract.
        """
        checkpoint["state_dict"] = {
            key[len("network.") :]: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("network.")
        }
        checkpoint["hyper_parameters"] = dict(self.network.hparams)

    def configure_optimizers(
        self,
    ) -> tuple[list[AdaBelief], list[dict[str, bool | str | ReduceLROnPlateau]]]:
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
