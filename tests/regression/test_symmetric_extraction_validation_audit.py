"""Regression coverage for extraction-time symmetry validation and audit attribution."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest
from chython import smiles

from synplan.chem.reaction.rules.config import RuleExtractionConfig
from synplan.chem.reaction.rules.extraction import (
    _make_extracted_rule_record,
    _update_rules_statistics,
    extract_rules,
    extract_rules_from_reactions,
    sort_rules,
)

_FIRST_SYMMETRIC_SUZUKI = (
    "[OH:25][B:24]([OH:26])[c:5]1[cH:6][cH:7][c:2]([F:1])"
    "[c:3]([cH:4]1)[CH3:11].[cH:21]1[cH:22][c:13]([c:14]"
    "([C:15](=[O:16])[O:17][CH3:18])[cH:19][c:20]1[F:23])[Br:27]>>"
    "[c:3]1([cH:4][c:5]([cH:6][cH:7][c:2]1[F:1])-[c:13]2"
    "[cH:22][cH:21][c:20]([cH:19][c:14]2[C:15](=[O:16])[O:17]"
    "[CH3:18])[F:23])[CH3:11]"
)

_SECOND_SYMMETRIC_SUZUKI = (
    "[CH3:23][C:22]([CH3:24])([CH3:25])[O:21][C:20]([NH:19][CH2:18]"
    "[C@H:17]1[CH2:16][CH2:15][C@@H:14]([CH2:28][CH2:27]1)"
    "[C:13]([NH:12][C@@H:7]([CH2:6][c:5]2[cH:4][cH:3][c:2]"
    "([cH:31][cH:30]2)[Br:1])[C:8]([O:9][CH3:10])=[O:11])=[O:29])"
    "=[O:26].[OH:43][B:42]([OH:44])[c:41]1[c:33]([cH:34][c:35]"
    "([C:36]([OH:37])=[O:38])[cH:39][cH:40]1)[Cl:32]>>"
    "[CH3:23][C:22]([CH3:24])([CH3:25])[O:21][C:20]([NH:19][CH2:18]"
    "[C@H:17]1[CH2:16][CH2:15][C@@H:14]([CH2:28][CH2:27]1)"
    "[C:13]([NH:12][C@@H:7]([CH2:6][c:5]2[cH:4][cH:3][c:2]"
    "([cH:31][cH:30]2)-[c:41]3[cH:40][cH:39][c:35]([cH:34]"
    "[c:33]3[Cl:32])[C:36](=[O:38])[OH:37])[C:8]([O:9][CH3:10])"
    "=[O:11])=[O:29])=[O:26]"
)

_SINGLE_CENTER_DEMETHYLATION = (
    "[CH3:1][O:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1>>"
    "[OH:2][c:3]1[cH:4][cH:5][cH:6][cH:7][cH:8]1"
)

_MULTI_CENTER_DEMETHYLATION = (
    "[CH3:1][O:2][c:3]1[cH:4][cH:5][cH:6][c:7]([O:8][CH3:9])"
    "[cH:10]1>>[OH:2][c:3]1[cH:4][cH:5][cH:6][c:7]([OH:8])[cH:10]1"
)


def _extraction_config(**overrides) -> RuleExtractionConfig:
    values = {
        "reactor_validation": True,
        "min_popularity": 1,
        "single_product_only": True,
        "ignore_stereo": True,
        "environment_atom_count": 1,
        "multicenter_rules": False,
        "include_rings": False,
        "include_func_groups": False,
        "keep_leaving_groups": True,
        "keep_incoming_groups": False,
        "keep_reagents": False,
        "atom_info_retention": {
            "reaction_center": {
                "neighbors": False,
                "implicit_hydrogens": False,
                "ring_sizes": False,
            },
            "environment": {
                "neighbors": False,
                "implicit_hydrogens": False,
                "ring_sizes": False,
            },
        },
    }
    values.update(overrides)
    return RuleExtractionConfig(**values)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames:
            reader.fieldnames[0] = reader.fieldnames[0].removeprefix("# ")
        return list(reader)


def _run_extraction(
    tmp_path: Path,
    reactions: list[str],
    *,
    config: RuleExtractionConfig | None = None,
    num_cpus: int = 1,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "reactions.smi"
    rules_path = tmp_path / "rules.tsv"
    audit_path = tmp_path / "rules.audit.tsv"
    policy_path = tmp_path / "rules_policy_data.tsv"
    input_path.write_text("\n".join(reactions) + "\n", encoding="utf-8")

    extract_rules_from_reactions(
        config=config or _extraction_config(),
        reaction_data_path=str(input_path),
        reaction_rules_path=str(rules_path),
        num_cpus=num_cpus,
        batch_size=2,
        ignore_errors=True,
        audit_file_path=audit_path,
    )

    return (
        _read_tsv(rules_path),
        _read_tsv(audit_path),
        _read_tsv(policy_path),
    )


@pytest.fixture(scope="module")
def symmetric_suzuki_records():
    records = []
    for reaction_smiles in (
        _FIRST_SYMMETRIC_SUZUKI,
        _SECOND_SYMMETRIC_SUZUKI,
    ):
        rules, skipped_multi_product = extract_rules(
            _extraction_config(), smiles(reaction_smiles)
        )
        assert not skipped_multi_product
        assert len(rules) == 1
        assert rules[0].meta["reactor_validation"] == "passed"
        records.append(_make_extracted_rule_record(rules[0]))

    assert records[0].cgr_key == records[1].cgr_key
    return tuple(records)


@pytest.mark.parametrize("ingestion_order", [(0, 1), (1, 0)])
def test_real_symmetric_suzuki_support_is_order_independent(
    symmetric_suzuki_records,
    ingestion_order,
):
    all_rules_statistics = defaultdict(list)
    eligible_rules_statistics = defaultdict(list)
    cgr_to_rule = {}

    for reaction_index in ingestion_order:
        _update_rules_statistics(
            all_rules_statistics,
            eligible_rules_statistics,
            cgr_to_rule,
            reaction_index,
            [symmetric_suzuki_records[reaction_index]],
        )

    sorted_rules, _filter_stats = sort_rules(
        all_rules_statistics,
        eligible_rules_statistics,
        cgr_to_rule,
        min_popularity=2,
    )

    assert len(sorted_rules) == 1
    representative, support_indices = sorted_rules[0]
    assert representative.reactor_validation == "passed"
    assert set(support_indices) == {0, 1}


@pytest.mark.parametrize(
    ("validation_results", "passed_index"),
    [
        pytest.param((False, True), 1, id="failed-first"),
        pytest.param((True, False), 0, id="passed-first"),
    ],
)
def test_same_key_failed_occurrence_does_not_count_as_support(
    monkeypatch,
    tmp_path,
    validation_results,
    passed_index,
):
    results = iter(validation_results)
    monkeypatch.setattr(
        "synplan.chem.reaction.rules.extraction.validate_rule",
        lambda _rule, _reaction: next(results),
    )

    rules, audit, policy = _run_extraction(
        tmp_path,
        [_SINGLE_CENTER_DEMETHYLATION, _SINGLE_CENTER_DEMETHYLATION],
    )

    assert len(rules) == 1
    assert rules[0]["popularity"] == "1"
    assert rules[0]["reaction_indices"] == str(passed_index)
    assert len(policy) == 1
    assert [(row["reaction_index"], row["error_type"]) for row in audit] == [
        (str(1 - passed_index), "ReactorValidationFailed")
    ]


@pytest.mark.parametrize(
    ("validation_results", "passed_index"),
    [
        pytest.param((False, True), 1, id="failed-first"),
        pytest.param((True, False), 0, id="passed-first"),
    ],
)
def test_min_popularity_counts_only_eligible_support(
    monkeypatch,
    tmp_path,
    validation_results,
    passed_index,
):
    results = iter(validation_results)
    monkeypatch.setattr(
        "synplan.chem.reaction.rules.extraction.validate_rule",
        lambda _rule, _reaction: next(results),
    )

    rules, audit, policy = _run_extraction(
        tmp_path,
        [_SINGLE_CENTER_DEMETHYLATION, _SINGLE_CENTER_DEMETHYLATION],
        config=_extraction_config(min_popularity=2),
    )

    assert rules == []
    assert policy == []
    error_type_by_index = {row["reaction_index"]: row["error_type"] for row in audit}
    assert error_type_by_index == {
        str(1 - passed_index): "ReactorValidationFailed",
        str(passed_index): "BelowMinPopularity",
    }


@pytest.mark.parametrize(
    ("multicenter_rules", "expected_status", "expected_validation_calls"),
    [
        pytest.param(False, "skipped_multicenter_component", 0, id="split-centers"),
        pytest.param(True, "failed", 1, id="combined-centers"),
    ],
)
def test_multicenter_rule_records_validation_outcome(
    monkeypatch,
    multicenter_rules,
    expected_status,
    expected_validation_calls,
):
    validation_calls = []

    def validation_fails(rule, reaction):
        validation_calls.append((rule, reaction))
        return False

    monkeypatch.setattr(
        "synplan.chem.reaction.rules.extraction.validate_rule",
        validation_fails,
    )

    rules, skipped = extract_rules(
        _extraction_config(multicenter_rules=multicenter_rules),
        smiles(_MULTI_CENTER_DEMETHYLATION),
    )

    assert not skipped
    assert rules
    assert len(validation_calls) == expected_validation_calls
    assert {rule.meta["reactor_validation"] for rule in rules} == {expected_status}


@pytest.mark.parametrize(
    ("multicenter_rules", "expected_multicenter_error"),
    [
        pytest.param(False, "MultiCenter", id="split-centers"),
        pytest.param(True, "ReactorValidationFailed", id="combined-centers"),
    ],
)
@pytest.mark.parametrize(
    "reactions",
    [
        pytest.param(
            [_SINGLE_CENTER_DEMETHYLATION, _MULTI_CENTER_DEMETHYLATION],
            id="single-first",
        ),
        pytest.param(
            [_MULTI_CENTER_DEMETHYLATION, _SINGLE_CENTER_DEMETHYLATION],
            id="multi-first",
        ),
    ],
)
def test_reactor_failure_audit_uses_validation_cause(
    monkeypatch,
    tmp_path,
    reactions,
    multicenter_rules,
    expected_multicenter_error,
):
    monkeypatch.setattr(
        "synplan.chem.reaction.rules.extraction.validate_rule",
        lambda _rule, _reaction: False,
    )

    rules, audit, policy = _run_extraction(
        tmp_path,
        reactions,
        config=_extraction_config(multicenter_rules=multicenter_rules),
    )

    assert rules == []
    assert policy == []
    error_type_by_reaction = {
        row["original_smiles"]: row["error_type"] for row in audit
    }
    assert error_type_by_reaction == {
        _SINGLE_CENTER_DEMETHYLATION: "ReactorValidationFailed",
        _MULTI_CENTER_DEMETHYLATION: expected_multicenter_error,
    }


def test_validation_disabled_occurrence_remains_eligible(tmp_path):
    rules, audit, policy = _run_extraction(
        tmp_path,
        [_SINGLE_CENTER_DEMETHYLATION],
        config=_extraction_config(reactor_validation=False),
    )

    assert len(rules) == 1
    assert rules[0]["popularity"] == "1"
    assert rules[0]["reaction_indices"] == "0"
    assert len(policy) == 1
    assert audit == []


def test_serial_parallel_parity_for_multicenter_audit_and_policy(tmp_path):
    reactions = [
        _SINGLE_CENTER_DEMETHYLATION,
        _MULTI_CENTER_DEMETHYLATION,
    ]
    serial_result = _run_extraction(
        tmp_path / "serial",
        reactions,
        config=_extraction_config(multicenter_rules=False),
        num_cpus=1,
    )
    parallel_result = _run_extraction(
        tmp_path / "parallel",
        reactions,
        config=_extraction_config(multicenter_rules=False),
        num_cpus=2,
    )

    assert parallel_result == serial_result
    rules, audit, policy = serial_result
    assert len(rules) == 1
    assert rules[0]["reaction_indices"] == "0"
    assert len(policy) == 1
    assert [(row["reaction_index"], row["error_type"]) for row in audit] == [
        ("1", "MultiCenter")
    ]
