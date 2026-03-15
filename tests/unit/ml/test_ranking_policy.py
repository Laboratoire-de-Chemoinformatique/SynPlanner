"""Tests for ranking policy dataset and stratified splitting."""

from collections import Counter
from pathlib import Path

import pytest

from synplan.ml.training.preprocessing import RankingPolicyDataset
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
            assert all(
                i in train_set for i in indices
            ), f"Duplicate product {key!r} leaked into validation"
            assert all(
                i not in val_set for i in indices
            ), f"Duplicate product {key!r} found in validation"


def test_val_fraction_per_rule(dataset):
    """No rule contributes more than ~10% of its examples to validation."""
    train, val = _stratified_ranking_split(dataset)
    val_set = set(val)
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
    train, val = _stratified_ranking_split(dataset)
    val_set = set(val)
    y = dataset._data.y_rules.view(-1).tolist()
    keys = dataset._product_keys

    # count single-occurrence candidates per rule (same logic as the function)
    product_groups = {}
    for idx, key in enumerate(keys):
        product_groups.setdefault(key, []).append(idx)

    rule_candidates = Counter()
    for key, indices in product_groups.items():
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
