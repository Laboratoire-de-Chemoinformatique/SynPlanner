#!/usr/bin/env python3
"""Terminal entrypoint for ranking policy training with stratified 90/5/5 split."""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import torch
from torch.utils.data import Subset
from torch_geometric.data.lightning import LightningDataset

from synplan.ml.training.preprocessing import RankingPolicyDataset
from synplan.ml.training.supervised import run_policy_training
from synplan.utils.config import PolicyNetworkConfig
from synplan.utils.loading import download_all_data
from synplan.utils.logging import DisableLogger, HiddenPrints


def _graph_fingerprint(graph) -> bytes:
    """Hash graph tensors and label for exact duplicate detection."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(int(graph.y_rules.item()).to_bytes(8, byteorder="little", signed=False))

    for attr in ("x", "edge_index", "edge_attr"):
        tensor = getattr(graph, attr, None)
        if tensor is None:
            continue
        tensor = tensor.contiguous()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if tensor.requires_grad:
            tensor = tensor.detach()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())

    return digest.digest()


def create_ranking_policy_dataset_stratified(
    reaction_rules_path: Union[str, Path],
    reactions_path: Union[str, Path],
    output_path: Union[str, Path],
    batch_size: int = 100,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    min_count_for_holdout: int = 20,
    deduplicate_identical: bool = False,
    seed: int = 42,
) -> tuple[LightningDataset, Dict[str, List[int]]]:
    """Create datamodule with per-template stratified train/val/test split."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    with DisableLogger(), HiddenPrints():
        full_dataset = RankingPolicyDataset(
            reaction_rules_path=str(reaction_rules_path),
            reactions_path=str(reactions_path),
            output_path=str(output_path),
        )

    y_rules = full_dataset._data.y_rules
    labels = y_rules.view(-1).tolist() if y_rules.dim() > 1 else y_rules.tolist()

    if deduplicate_identical:
        candidate_indices: Iterable[int] = []
        unique_indices: List[int] = []
        seen = set()
        for idx in range(len(labels)):
            fp = _graph_fingerprint(full_dataset[idx])
            if fp in seen:
                continue
            seen.add(fp)
            unique_indices.append(idx)
        candidate_indices = unique_indices
    else:
        candidate_indices = range(len(labels))

    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx in candidate_indices:
        buckets[int(labels[idx])].append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for idxs in buckets.values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_t = len(idxs)

        if n_t < min_count_for_holdout:
            train_indices.extend(idxs)
            continue

        n_val = max(1, int(round(n_t * val_ratio)))
        n_test = max(1, int(round(n_t * test_ratio)))
        n_train = n_t - n_val - n_test

        if n_train < 1:
            train_indices.extend(idxs)
            continue

        train_indices.extend(idxs[:n_train])
        val_indices.extend(idxs[n_train : n_train + n_val])
        test_indices.extend(idxs[n_train + n_val :])

    datamodule = LightningDataset(
        Subset(full_dataset, train_indices),
        Subset(full_dataset, val_indices),
        Subset(full_dataset, test_indices),
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
    )

    total = len(train_indices) + len(val_indices) + len(test_indices)
    print(f"Total examples used: {total}")
    print(
        f"Split sizes -> train: {len(train_indices)} ({len(train_indices) / total:.3f}), "
        f"val: {len(val_indices)} ({len(val_indices) / total:.3f}), "
        f"test: {len(test_indices)} ({len(test_indices) / total:.3f})"
    )

    return datamodule, {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


def _parse_devices(devices: str) -> Union[str, int, Sequence[int]]:
    """Parse Lightning devices argument from CLI string."""
    raw = devices.strip()
    if raw == "auto":
        return "auto"
    if "," in raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return int(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Train ranking policy network with stratified 90/5/5 split."
    )
    parser.add_argument(
        "--reaction-rules-path",
        type=Path,
        default=repo_root / "mapping/uspto_full_rules_light.tsv",
        help="Path to extracted reaction rules TSV/pickle.",
    )
    parser.add_argument(
        "--filtered-data-path",
        type=Path,
        default=repo_root / "mapping/uspto_filtered_light.smi",
        help="Path to filtered reactions file (.smi/.rdf).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "tutorial_results/ranking_policy_network",
        help="Output directory for dataset cache and trained weights.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-epoch", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.0008)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--num-conv-layers", type=int, default=5)
    parser.add_argument("--vector-dim", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--min-count-for-holdout", type=int, default=20)
    parser.add_argument("--deduplicate-identical", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help='Lightning accelerator: "cpu", "gpu", "auto", ...',
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help='Lightning devices: "auto", "1", "-1", or comma list like "0,1".',
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download SynPlanner data into ./synplan_data before training.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Disable progress bars/logging during trainer.fit.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.download_data:
        data_folder = Path("synplan_data").resolve()
        download_all_data(save_to=data_folder)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.results_dir / "ranking_policy_dataset.pt"

    torch.manual_seed(args.seed)

    datamodule, _split_indices = create_ranking_policy_dataset_stratified(
        reaction_rules_path=args.reaction_rules_path,
        reactions_path=args.filtered_data_path,
        output_path=dataset_path,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_count_for_holdout=args.min_count_for_holdout,
        deduplicate_identical=args.deduplicate_identical,
        seed=args.seed,
    )

    training_config = PolicyNetworkConfig(
        policy_type="ranking",
        num_conv_layers=args.num_conv_layers,
        vector_dim=args.vector_dim,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
    )

    run_policy_training(
        datamodule=datamodule,
        config=training_config,
        results_path=args.results_dir,
        accelerator=args.accelerator,
        devices=_parse_devices(args.devices),
        silent=args.silent,
    )


if __name__ == "__main__":
    main()
