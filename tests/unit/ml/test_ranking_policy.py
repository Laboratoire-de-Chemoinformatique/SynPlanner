"""Tests for ranking policy dataset and stratified splitting."""

from collections import Counter
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

import synplan.ml.training.preprocessing as preprocessing
from synplan.ml.training.preprocessing import (
    FilteringPolicyDataset,
    RankingPolicyDataset,
)
from synplan.ml.training.supervised import _stratified_ranking_split
from synplan.utils.cache import cache_digest

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
POLICY_DATA = str(DATA_DIR / "policy_data_small.tsv")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Build dataset once for the whole module."""
    cache = str(tmp_path_factory.mktemp("cache") / "ds.safetensors")
    return RankingPolicyDataset(policy_data_path=POLICY_DATA, output_path=cache)


# ── Dataset tests ──────────────────────────────────────────────


def test_dataset_loads(dataset):
    assert len(dataset) > 0


def test_dataset_has_product_keys(dataset):
    assert dataset._product_keys is not None
    assert len(dataset._product_keys) == len(dataset)


def test_dataset_graph_attributes(dataset):
    graph = dataset[0]
    assert hasattr(graph, "x")
    assert hasattr(graph, "edge_index")
    assert hasattr(graph, "y_rules")
    assert graph.y_rules.numel() == 1


def test_dataset_num_classes(dataset):
    assert dataset.num_classes > 0


def test_parallel_dataset_build_submits_batches(monkeypatch, tmp_path):
    """Parallel ranking preprocessing should amortize IPC by submitting batches."""
    row_count = 5001
    policy_data = tmp_path / "policy_data.tsv"
    policy_data.write_text(
        "product_smiles\trule_id\n"
        + "".join(f"product-{idx}\t{idx % 7}\n" for idx in range(row_count)),
        encoding="utf-8",
    )

    submitted_batches = []

    def fake_convert_item(item):
        product_smi, rule_id_str = item
        graph = Data(
            x=torch.tensor([[1.0]]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 4), dtype=torch.float),
        )
        graph.y_rules = torch.tensor([int(rule_id_str)], dtype=torch.long)
        return graph, product_smi

    def fake_process_pool_map_stream(items, worker_fn, **_kwargs):
        for batch in items:
            assert isinstance(batch, list)
            assert batch
            submitted_batches.append(batch)
            yield worker_fn(batch)

    monkeypatch.setattr(preprocessing, "_convert_ranking_item", fake_convert_item)
    monkeypatch.setattr(
        preprocessing, "process_pool_map_stream", fake_process_pool_map_stream
    )

    dataset = RankingPolicyDataset(
        policy_data_path=str(policy_data),
        output_path="",
        num_workers=2,
    )

    assert len(dataset) == row_count
    assert sum(len(batch) for batch in submitted_batches) == row_count
    assert any(len(batch) > 1 for batch in submitted_batches)
    assert dataset._product_keys == [f"product-{idx}" for idx in range(row_count)]


def test_ranking_batch_payload_does_not_return_torch_objects(monkeypatch):
    """Worker payloads must avoid torch shared-memory file descriptors."""

    def fake_convert_item(item):
        product_smi, rule_id_str = item
        graph = Data(
            x=torch.tensor([[6, 2, 14, 4, 2, 0, 0, 1, 1, 0, 0]], dtype=torch.uint8),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float,
            ),
        )
        graph.y_rules = torch.tensor([int(rule_id_str)], dtype=torch.long)
        return graph, product_smi

    def contains_torch_object(value):
        if isinstance(value, (torch.Tensor, Data)):
            return True
        if isinstance(value, dict):
            return any(contains_torch_object(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_torch_object(v) for v in value)
        return False

    monkeypatch.setattr(preprocessing, "_convert_ranking_item", fake_convert_item)

    payload = preprocessing._convert_ranking_batch([("CC", "3")])

    assert not contains_torch_object(payload)


def test_filtering_dataset_parallel_crash_refuses_serial_fallback(
    monkeypatch, tmp_path
):
    """Filtering preprocessing should fail visibly if the worker pool crashes."""
    molecules = tmp_path / "molecules.smi"
    molecules.write_text("CCO\n", encoding="utf-8")

    def fake_process_pool_map_stream(*_args, **_kwargs):
        raise BrokenProcessPool("worker crashed")

    monkeypatch.setattr(
        preprocessing, "process_pool_map_stream", fake_process_pool_map_stream
    )

    with pytest.raises(RuntimeError, match="Refusing silent serial fallback"):
        FilteringPolicyDataset(
            molecules_path=str(molecules),
            reaction_rules_path="rules.tsv",
            output_path="",
            num_cpus=2,
        )


# ── Stratified split tests ─────────────────────────────────────


def test_split_covers_all_indices(dataset):
    """Train + val must cover every index exactly once."""
    train, val = _stratified_ranking_split(dataset)
    assert sorted(train + val) == list(range(len(dataset)))


def test_no_data_leakage(dataset):
    """Products appearing multiple times must all be in train."""
    train, val = _stratified_ranking_split(dataset)
    train_set, val_set = set(train), set(val)
    keys = dataset._product_keys

    product_groups = {}
    for idx, key in enumerate(keys):
        product_groups.setdefault(key, []).append(idx)

    for key, indices in product_groups.items():
        if len(indices) > 1:
            assert all(i in train_set for i in indices), (
                f"Duplicate product {key!r} leaked into validation"
            )
            assert all(i not in val_set for i in indices), (
                f"Duplicate product {key!r} found in validation"
            )


def test_val_fraction_per_rule(dataset):
    """No rule contributes more than ~10% of its examples to validation."""
    _train, val = _stratified_ranking_split(dataset)
    set(val)
    y = dataset._data.y_rules.view(-1).tolist()

    rule_total = Counter(y)
    rule_in_val = Counter(y[i] for i in val)

    for rule_id, n_val in rule_in_val.items():
        n_total = rule_total[rule_id]
        # only rules with >20 candidates go to val,
        # and they get at most 10%  (allow small rounding margin)
        assert n_val / n_total <= 0.15, (
            f"Rule {rule_id}: {n_val}/{n_total} "
            f"({n_val / n_total:.0%}) in validation, expected ≤10%"
        )


def test_small_rules_stay_in_train(dataset):
    """Rules with ≤20 single-occurrence examples must have 0 val entries."""
    _train, val = _stratified_ranking_split(dataset)
    set(val)
    y = dataset._data.y_rules.view(-1).tolist()
    keys = dataset._product_keys

    # count single-occurrence candidates per rule (same logic as the function)
    product_groups = {}
    for idx, key in enumerate(keys):
        product_groups.setdefault(key, []).append(idx)

    rule_candidates = Counter()
    for _key, indices in product_groups.items():
        if len(indices) == 1:
            rule_candidates[y[indices[0]]] += 1

    for rule_id, n_cand in rule_candidates.items():
        if n_cand <= 20:
            val_from_rule = [i for i in val if y[i] == rule_id]
            assert len(val_from_rule) == 0, (
                f"Rule {rule_id} has only {n_cand} candidates "
                f"but {len(val_from_rule)} ended up in validation"
            )


def test_split_deterministic(dataset):
    """Same seed → same split."""
    t1, v1 = _stratified_ranking_split(dataset, seed=42)
    t2, v2 = _stratified_ranking_split(dataset, seed=42)
    assert sorted(t1) == sorted(t2)
    assert sorted(v1) == sorted(v2)


def test_split_has_validation(dataset):
    """With rules >20, the validation set must be non-empty."""
    _, val = _stratified_ranking_split(dataset)
    assert len(val) > 0, "Validation set is empty"


# ── Lightweight cache digest test ─────────────────────────────


def test_cache_digest_changes_with_content(tmp_path):
    """Digest must change when file content changes."""
    f = tmp_path / "data.tsv"
    f.write_text("col1\tcol2\nA\t1\nB\t2\n")
    d1 = cache_digest(str(f), extra="test")

    f.write_text("col1\tcol2\nA\t1\nB\t2\nC\t3\n")
    d2 = cache_digest(str(f), extra="test")

    assert d1 != d2, "Digest should change when file content changes"
