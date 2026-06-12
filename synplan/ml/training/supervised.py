"""Module for the preparation and training of a policy network used in the expansion of
nodes in tree search.

This module includes functions for creating training datasets and running the training
process for the policy network.
"""

import random
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import Subset, random_split
from torch_geometric.data.lightning import LightningDataset

from synplan.ml.networks.checkpoint import load_policy_network_from_checkpoint
from synplan.ml.networks.policy.mhnreact import MHNReact
from synplan.ml.training.lightning import GradNormLogger, LitNetworkTrainer
from synplan.ml.training.preprocessing import (
    FilteringPolicyDataset,
    RankingPolicyDataset,
)
from synplan.ml.training.trainers import (
    LitFilteringPolicy,
    LitMHNRanking,
    LitRankingPolicy,
    rebind_training_rules,
    validate_ranking_labels,
)
from synplan.utils.cache import cache_digest
from synplan.utils.config import MHNRankingPolicyNetworkConfig, PolicyNetworkConfig
from synplan.utils.logging import DisableLogger, HiddenPrints

warnings.filterwarnings("ignore")


def stratified_ranking_split(
    dataset: RankingPolicyDataset,
    min_rule_count: int = 20,
    max_val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split a ranking policy dataset into train/val with stratification.

    Rules with fewer than *min_rule_count* unique-product examples contribute
    only to training.  For larger rules, at most *max_val_fraction* of their
    unique-product examples go to validation.  Products that appear in more
    than one example (i.e. identical molecules from different reactions) are
    always placed in the training set to prevent data leakage.

    :param dataset: A :class:`RankingPolicyDataset` with ``_product_keys``.
    :param min_rule_count: Minimum number of unique-product candidates a rule
        must have before any of its examples can go to validation.
    :param max_val_fraction: Maximum fraction of a rule's candidates placed
        into the validation set.
    :param seed: Random seed for reproducibility.
    :return: ``(train_indices, val_indices)``
    """
    y_rules = dataset._data.y_rules
    labels = y_rules.view(-1).tolist() if y_rules.dim() > 1 else y_rules.tolist()
    product_keys = dataset._product_keys

    # Group indices by product key.
    product_groups: dict[str, list[int]] = defaultdict(list)
    for idx, pkey in enumerate(product_keys):
        product_groups[pkey].append(idx)

    # Separate: multi-occurrence products → train; single-occurrence → candidates.
    train_indices: list[int] = []
    # candidate indices grouped by rule_id
    rule_candidates: dict[int, list[int]] = defaultdict(list)
    n_leaked = 0

    for _pkey, indices in product_groups.items():
        if len(indices) > 1:
            train_indices.extend(indices)
            n_leaked += len(indices)
        else:
            idx = indices[0]
            rule_candidates[labels[idx]].append(idx)

    # Per-rule stratified split of candidates.
    val_indices: list[int] = []
    rng = random.Random(seed)

    for _rule_id, candidates in rule_candidates.items():
        if len(candidates) <= min_rule_count:
            train_indices.extend(candidates)
            continue
        rng.shuffle(candidates)
        n_val = max(1, int(len(candidates) * max_val_fraction))
        val_indices.extend(candidates[:n_val])
        train_indices.extend(candidates[n_val:])

    n_total = len(train_indices) + len(val_indices)
    n_rules_in_val = len({labels[i] for i in val_indices})
    print(
        f"Stratified split: {n_total} total, "
        f"train {len(train_indices)}, val {len(val_indices)} "
        f"({len(val_indices) / n_total * 100:.1f}%), "
        f"{n_rules_in_val} rules in val, "
        f"{n_leaked} examples forced to train (duplicate products)"
    )

    return train_indices, val_indices


def create_policy_dataset(
    policy_data_path: str | None = None,
    reaction_rules_path: str | None = None,
    molecules_or_reactions_path: str | None = None,
    results_dir: str = ".",
    output_path: str = "",
    dataset_type: str = "filtering",
    batch_size: int = 100,
    num_cpus: int = 1,
    num_workers: int = 0,
    training_data_ratio: float = 0.8,
    cache: bool = True,
):
    """
    Create a training dataset for a policy network.

    For ranking policy, provide *policy_data_path* (the mapping file produced
    by rule extraction).  For filtering policy, provide *reaction_rules_path*
    and *molecules_or_reactions_path*.

    Preprocessed datasets are cached as safetensors files in
    ``{results_dir}/tmp/`` by default.  The cache filename includes a
    digest of the input data so it auto-invalidates when inputs change.
    Pass *output_path* to override the cache location directly.

    :param policy_data_path: Path to the policy training mapping file
        (ranking policy only).
    :param reaction_rules_path: Path to the reaction rules file
        (filtering policy).
    :param molecules_or_reactions_path: Path to the molecules file
        (filtering policy).
    :param results_dir: Directory for results and dataset cache.
    :param output_path: Explicit cache file path (overrides auto-computed
        path). Kept for backward compatibility.
    :param dataset_type: ``'ranking'`` or ``'filtering'``.
    :param batch_size: Batch size for the data module.
    :param training_data_ratio: Train ratio for filtering policy (random split).
    :param num_cpus: Number of CPUs (filtering policy only).
    :param num_workers: CPU workers for ranking preprocessing (0 = auto).
    :param cache: If True (default), cache the preprocessed dataset to disk.
    :return: A ``LightningDataset`` with train and validation splits.
    """

    # Compute cache path: explicit output_path wins, otherwise auto next
    # to the input data so the cache is shared across experiments.
    if not output_path and cache:
        if dataset_type == "ranking" and policy_data_path:
            cache_dir = Path(policy_data_path).resolve().parent / ".cache"
            digest = cache_digest(policy_data_path, extra="ranking")
            output_path = str(cache_dir / f"ranking_{digest}.safetensors")
        elif (
            dataset_type == "filtering"
            and molecules_or_reactions_path
            and reaction_rules_path
        ):
            cache_dir = Path(molecules_or_reactions_path).resolve().parent / ".cache"
            digest = cache_digest(
                molecules_or_reactions_path, reaction_rules_path, extra="filtering"
            )
            output_path = str(cache_dir / f"filtering_{digest}.safetensors")
    if not cache:
        output_path = ""

    with DisableLogger(), HiddenPrints():
        if dataset_type == "filtering":
            full_dataset = FilteringPolicyDataset(
                reaction_rules_path=reaction_rules_path,
                molecules_path=molecules_or_reactions_path,
                output_path=output_path,
                num_cpus=num_cpus,
            )
        elif dataset_type == "ranking":
            full_dataset = RankingPolicyDataset(
                policy_data_path=policy_data_path,
                output_path=output_path,
                num_workers=num_workers,
            )

    # Stratified split for ranking policy; random split for filtering.
    if (
        dataset_type == "ranking"
        and hasattr(full_dataset, "_product_keys")
        and full_dataset._product_keys is not None
    ):
        train_indices, val_indices = stratified_ranking_split(full_dataset)
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
    else:
        train_size = int(training_data_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], torch.Generator().manual_seed(42)
        )
        print(
            f"Training set size: {len(train_dataset)}, "
            f"validation set size: {len(val_dataset)}"
        )

    datamodule = LightningDataset(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
    )

    return datamodule


def create_training_logger(logger_config: dict | None, results_path: Path):
    """Create a PyTorch Lightning logger from a config dict.

    :param logger_config: Dict with ``"type"`` and optional logger kwargs, or None.
    :param results_path: Default ``save_dir`` for the logger.
    :return: A Lightning logger instance, or ``False`` to disable logging.
    """
    if logger_config is None:
        return False

    kwargs = dict(logger_config)
    logger_type = kwargs.pop("type").lower()

    if logger_type in {"lit", "litlogger"}:
        try:
            from pytorch_lightning.loggers import LitLogger
        except ImportError as e:
            raise ImportError(
                "LitLogger requires the 'litlogger' package. "
                "Install SynPlanner with the 'litlogger' or 'loggers' extra."
            ) from e

        root_dir = kwargs.pop("root_dir", kwargs.pop("save_dir", str(results_path)))
        return LitLogger(root_dir=root_dir, **kwargs)

    kwargs.setdefault("save_dir", str(results_path))

    if logger_type == "csv":
        return CSVLogger(**kwargs)
    elif logger_type == "tensorboard":
        return TensorBoardLogger(**kwargs)
    elif logger_type == "mlflow":
        try:
            from pytorch_lightning.loggers import MLFlowLogger
        except ImportError as e:
            raise ImportError(
                "MLflow logger requires the 'mlflow' package. "
                "Install SynPlanner with the 'mlflow' or 'loggers' extra."
            ) from e
        return MLFlowLogger(**kwargs)
    elif logger_type == "wandb":
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError as e:
            raise ImportError(
                "Wandb logger requires the 'wandb' package. "
                "Install SynPlanner with the 'wandb' or 'loggers' extra."
            ) from e
        return WandbLogger(**kwargs)
    else:
        raise ValueError(f"Unknown logger type: '{logger_type}'")


def fit_policy_network(
    trainer_module: LitNetworkTrainer,
    datamodule: LightningDataset,
    config: PolicyNetworkConfig,
    results_path: str,
    *,
    weights_file_name: str = "policy_network",
    accelerator: str = "gpu",
    devices: list[int] | str | int = "auto",
    silent: bool = False,
) -> None:
    """Fit a prepared policy trainer and save the best checkpoint."""
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(
        dirpath=results_path, filename=weights_file_name, monitor="val_loss", mode="min"
    )

    callbacks = [checkpoint]
    if config.log_grad_norm:
        callbacks.append(GradNormLogger())

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.num_epoch,
        callbacks=callbacks,
        logger=create_training_logger(config.logger, results_path),
        gradient_clip_val=1.0,
        enable_progress_bar=not silent,
    )
    if config.trainer:
        trainer_kwargs.update(config.trainer)

    trainer = Trainer(**trainer_kwargs)

    if silent:
        with DisableLogger(), HiddenPrints():
            trainer.fit(trainer_module, datamodule)
    else:
        trainer.fit(trainer_module, datamodule)

    ba = round(trainer.logged_metrics["train_balanced_accuracy_y_step"].item(), 3)
    print(f"Policy network balanced accuracy: {ba}")


TRAINER_BY_KEY: dict[str, type[LitNetworkTrainer]] = {
    "mhn_ranking": LitMHNRanking,
    "ranking": LitRankingPolicy,
    "filtering": LitFilteringPolicy,
}


def build_policy_trainer(config: PolicyNetworkConfig, dataset) -> LitNetworkTrainer:
    """Build the pure network + matching Lit trainer for the configured policy."""
    key = (
        config.architecture
        if config.architecture == "mhn_ranking"
        else config.policy_type
    )
    trainer_cls = TRAINER_BY_KEY[key]
    if key == "mhn_ranking":
        return trainer_cls.from_config(config, dataset)
    return trainer_cls.from_config(config, n_rules=dataset.num_classes)


def run_policy_training(
    datamodule: LightningDataset,
    config: PolicyNetworkConfig,
    results_path: str,
    weights_file_name: str = "policy_network",
    accelerator: str = "gpu",
    devices: list[int] | str | int = "auto",
    silent: bool = False,
) -> None:
    """
    Trains a policy network using a given datamodule and training configuration.

    :param datamodule: A PyTorch Lightning ``DataModule`` instance.
    :param config: Configuration for the policy training process.
    :param results_path: Path to store the training results and logs.
    :param accelerator: Lightning accelerator type. Default: ``"gpu"``.
    :param devices: Lightning devices specification. Default: ``"auto"``.
    :param silent: Suppress progress bars. Default: ``False``.
    :param weights_file_name: Name of the saved weights file.
        Default: ``"policy_network"``.
    :return: None.
    """
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    if config.architecture == "mhn_ranking" and config.policy_type != "ranking":
        raise ValueError("architecture='mhn_ranking' requires policy_type='ranking'")

    dataset = datamodule.train_dataset.dataset
    trainer_module = build_policy_trainer(config, dataset)

    fit_policy_network(
        trainer_module,
        datamodule,
        config,
        str(results_path),
        weights_file_name=weights_file_name,
        accelerator=accelerator,
        devices=devices,
        silent=silent,
    )


def mhn_tuning_config(
    network: MHNReact,
    config: PolicyNetworkConfig | None,
) -> MHNRankingPolicyNetworkConfig:
    """Resolve the tuning config from a user config or the network hparams."""
    if config is not None:
        values = config.model_dump()
        values["architecture"] = "mhn_ranking"
        values["policy_type"] = "ranking"
        return MHNRankingPolicyNetworkConfig.model_validate(values)
    return MHNRankingPolicyNetworkConfig.model_validate(network.hparams["config"])


def run_mhn_network_tuning(
    *,
    policy_network_path: str,
    new_policy_data_path: str,
    results_path: str,
    config: PolicyNetworkConfig | None = None,
    num_workers: int = 0,
    cache: bool = True,
    weights_file_name: str = "policy_network",
    accelerator: str = "gpu",
    devices: list[int] | str | int = "auto",
    silent: bool = False,
) -> None:
    """Fine-tune an existing MHN ranking checkpoint on new ranking policy data."""
    network = load_policy_network_from_checkpoint(policy_network_path, batch_size=1)
    if getattr(network, "architecture", None) != "mhn_ranking":
        raise ValueError("mhn_network_tuning requires an mhn_ranking checkpoint")

    config = mhn_tuning_config(network, config)
    datamodule = create_policy_dataset(
        policy_data_path=new_policy_data_path,
        results_dir=results_path,
        dataset_type="ranking",
        batch_size=config.batch_size,
        num_workers=num_workers,
        cache=cache,
    )
    dataset = datamodule.train_dataset.dataset
    rebind_training_rules(network, new_policy_data_path)
    validate_ranking_labels(dataset._data.y_rules, network.n_rules)
    network.batch_size = config.batch_size
    network.lr = config.learning_rate
    network.hparams["config"]["batch_size"] = config.batch_size
    network.hparams["config"]["learning_rate"] = config.learning_rate

    fit_policy_network(
        LitMHNRanking(network),
        datamodule,
        config,
        results_path,
        weights_file_name=weights_file_name,
        accelerator=accelerator,
        devices=devices,
        silent=silent,
    )
