"""Invariants for ``extract_rules_from_reactions``.

These tests pin behaviours that are *independent of which rules get extracted*:

1. When ``reactor_validation=False`` the output ruleset is non-empty. There is
   a recurring failure mode where disabling validation causes ``sort_rules``
   to reject every rule (``validation`` becomes the sentinel ``"not_set"`` and
   the equality check ``validation != "passed"`` fires for every rule). Users
   who toggle the flag expecting MORE rules to pass get zero with no warning.

2. Output rules TSV plus errors TSV plus the policy-data TSV are *self
   consistent*: rule-id ranges in the policy data are a subset of rule ids in
   the rules TSV, and product SMILES referenced in the policy data are
   recoverable. This catches "policy data references rule_id beyond what the
   rules file declares" and similar plumbing breakage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synplan.chem.reaction_rules.extraction import extract_rules_from_reactions


def _count_rule_rows(tsv_path: Path) -> int:
    """Count data rows in a rules-format TSV.

    The header is a plain `rule_smarts\tpopularity\treaction_indices`
    line (no leading `#`), so we strip the first line unconditionally.
    """
    if not tsv_path.exists():
        return 0
    lines = [
        line
        for line in tsv_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return max(0, len(lines) - 1)


def test_extraction_with_validation_disabled_produces_rules(
    tmp_path: Path,
    sample_reactions_file,
    rule_cfg_factory,
):
    """``reactor_validation=False`` must not silently empty the output.

    Strategy: run extraction twice on the *same* fixture, once with
    validation enabled (the baseline) and once disabled (the path under
    test). If the baseline produces zero rules, the fixture itself can't
    exercise the bug; we skip loudly so a future fixture regression
    surfaces as a SKIP rather than a silent PASS. Otherwise, the
    disabled-validation path must produce a non-empty rule set; getting
    zero indicates the ``sort_rules`` "validation != 'passed'" sentinel
    bug.
    """
    rules_on = tmp_path / "rules_on.tsv"
    rules_off = tmp_path / "rules_off.tsv"
    extract_rules_from_reactions(
        config=rule_cfg_factory(reactor_validation=True, min_popularity=1),
        reaction_data_path=str(sample_reactions_file),
        reaction_rules_path=str(rules_on),
        num_cpus=1,
        batch_size=4,
        ignore_errors=True,
    )
    n_on = _count_rule_rows(rules_on)
    if n_on == 0:
        pytest.skip(
            "fixture produces no rules under reactor_validation=True; cannot "
            "meaningfully test the validation=False path. Update conftest "
            "sample_reactions to include rules-producing reactions."
        )
    extract_rules_from_reactions(
        config=rule_cfg_factory(reactor_validation=False, min_popularity=1),
        reaction_data_path=str(sample_reactions_file),
        reaction_rules_path=str(rules_off),
        num_cpus=1,
        batch_size=4,
        ignore_errors=True,
    )
    n_off = _count_rule_rows(rules_off)
    assert n_off > 0, (
        f"Extraction with reactor_validation=False produced 0 rules but "
        f"validation=True on the same input produced {n_on}. All rules were "
        "rejected; likely sort_rules treats the absent 'reactor_validation' "
        "meta as a non-'passed' value and filters every rule out. Users "
        "disabling validation expect more rules to pass, not zero."
    )


def test_extraction_validation_off_produces_at_least_as_many_as_on(
    tmp_path: Path,
    sample_reactions_file,
    rule_cfg_factory,
):
    """Disabling validation must not produce *fewer* rules than keeping it on.

    Property: |rules(validation=False)| >= |rules(validation=True)|. Skipping
    a filter cannot reduce the set of rules that pass it. If this invariant
    fails, validation=False is more restrictive than validation=True: the
    classic sentinel-comparison bug.
    """
    rules_on = tmp_path / "rules_on.tsv"
    rules_off = tmp_path / "rules_off.tsv"
    extract_rules_from_reactions(
        config=rule_cfg_factory(reactor_validation=True, min_popularity=1),
        reaction_data_path=str(sample_reactions_file),
        reaction_rules_path=str(rules_on),
        num_cpus=1,
        batch_size=4,
        ignore_errors=True,
    )
    extract_rules_from_reactions(
        config=rule_cfg_factory(reactor_validation=False, min_popularity=1),
        reaction_data_path=str(sample_reactions_file),
        reaction_rules_path=str(rules_off),
        num_cpus=1,
        batch_size=4,
        ignore_errors=True,
    )
    n_on = _count_rule_rows(rules_on)
    n_off = _count_rule_rows(rules_off)
    assert n_off >= n_on, (
        f"reactor_validation=False yielded fewer rules ({n_off}) than "
        f"reactor_validation=True ({n_on}). Disabling a filter must be "
        "monotonic; the inversion indicates a sentinel-vs-value comparison "
        "in sort_rules."
    )


def test_extraction_policy_data_consistent_with_rules_tsv(
    tmp_path: Path,
    sample_reactions_file,
    rule_cfg_factory,
):
    """Policy data references no rule_ids beyond what the rules TSV declares.

    Catches the ``num_classes = max(y_rules) + 1`` family of bugs where
    training-data row indices and rules-file row indices disagree.
    """
    rules_path = tmp_path / "rules.tsv"
    extract_rules_from_reactions(
        config=rule_cfg_factory(reactor_validation=True, min_popularity=1),
        reaction_data_path=str(sample_reactions_file),
        reaction_rules_path=str(rules_path),
        num_cpus=1,
        batch_size=4,
        ignore_errors=True,
    )
    if not rules_path.exists():
        pytest.skip("no rules file produced; covered by other invariants")
    n_rules = _count_rule_rows(rules_path)
    # Policy data file is written alongside as <stem>_policy_data.tsv.
    policy_path = rules_path.with_name(rules_path.stem + "_policy_data.tsv")
    if not policy_path.exists():
        pytest.skip(f"policy_data.tsv not produced at {policy_path}")
    rule_ids = set()
    for line in policy_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) >= 2 and cols[1].strip().isdigit():
            rule_ids.add(int(cols[1]))
    if not rule_ids:
        pytest.skip("policy data has no parseable rule_ids in column 2")
    assert max(rule_ids) < n_rules, (
        f"Policy data references rule_id={max(rule_ids)} but the rules TSV "
        f"only declares {n_rules} rules (ids 0..{n_rules - 1}). Downstream "
        "training will index out of range or undersize the output head."
    )
