"""Synthon coverage of a mapped reaction: the label machinery, and the CLI filter.

Every expectation here is chemistry, not a recorded output: each fixture record names the
disconnection the reaction really is, and the negative controls are reactions that share a
substructure or a formed bond with a positive one and are a different reaction.
"""

from pathlib import Path

import pytest
from chython import smarts, smiles

from synplan.synthon import cli
from synplan.synthon.config import SynthonConfig, load_data
from synplan.synthon.coverage import (
    CoverageRule,
    classify_coverage,
    load_coverage_rules,
)
from synplan.synthon.priority import capped_smarts
from synplan.synthon.reactor import (
    RULE_NUCLEOPHILE_CAPS,
    SynthonRuleError,
    query_labels,
)

FIXTURE = Path(__file__).parent.parent / "data" / "synthon" / "reaction_coverage.smi"


def _fixture_records() -> list[tuple[str, str, tuple[str, ...]]]:
    records = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record, name, expected = line.split("\t")
        records.append(
            (record, name, () if expected == "-" else tuple(expected.split(",")))
        )
    return records


RECORDS = _fixture_records()


@pytest.fixture(scope="module")
def rules() -> list[CoverageRule]:
    return load_coverage_rules()


def test_fixture_covers_both_verdicts():
    assert len(RECORDS) >= 20, "fixture shrank"
    assert any(expected for _, _, expected in RECORDS)
    assert any(not expected for _, _, expected in RECORDS)


@pytest.mark.parametrize(
    ("record", "expected"),
    [(record, expected) for record, _, expected in RECORDS],
    ids=[name for _, name, _ in RECORDS],
)
def test_coverage_matches_the_chemistry(record, expected, rules):
    result = classify_coverage(smiles(record), rules)
    assert result.rule_ids == expected, result.evidence
    assert result.covered is bool(expected)


# --- the label machinery ------------------------------------------------------------------


def test_the_matcher_cannot_be_label_aware_so_the_tokens_are_checked_by_hand():
    """chython's ``QueryElement.__eq__`` never consults ``_label``. If that ever changes the
    hand-rolled token check in coverage.py is redundant, not wrong — but revisit it."""
    labelled = smarts("[#6_elec:1]")
    assert getattr(next(iter(labelled.atoms()))[1], "_label", None) == "elec"
    assert labelled.is_substructure(smiles("C")), "chython started consulting _label"


def test_the_raw_rules_carry_the_labels_that_capping_strips(rules):
    """Coverage is label-aware only because it reads the raw ``rules.json`` SMARTS."""
    assert len(rules) == 39
    assert sum(len(rule.labels) for rule in rules) == 78
    data = load_data(SynthonConfig().rules_path)
    records = {r["id"]: r for r in data["disconnections"]}
    capped = capped_smarts(
        records["R1.1"]["smarts"], data["leaving_groups"], rule_id="R1.1"
    )
    assert not query_labels(smarts(capped.split(">>", 1)[1]))


def test_a_rule_that_lost_its_labels_is_refused():
    """A labelless rule matches on substructure alone and over-covers, silently."""
    unlabelled = {
        "id": "Rfake",
        "name": "amide, tokens stripped",
        "smarts": "[C:1](=[O:2])[N:3]>>[C:1](=[O:2]).[N:3]",
    }
    with pytest.raises(SynthonRuleError, match="no synthon label"):
        CoverageRule(unlabelled)


def test_the_named_nucleophiles_come_from_the_capping_table():
    """One table, not two: the rules that must SPELL a reagent are the rules that must SEE it."""
    from synplan.synthon.coverage import _RULE_NUCLEOPHILE_ELEMENTS

    assert set(_RULE_NUCLEOPHILE_ELEMENTS) == set(RULE_NUCLEOPHILE_CAPS)
    assert _RULE_NUCLEOPHILE_ELEMENTS["R12.3a"] == frozenset(("B",))
    assert "Zn" in _RULE_NUCLEOPHILE_ELEMENTS["R10.2"]


def test_a_named_nucleophile_is_a_positive_constraint(rules):
    """`nuc` read as "no halide left this atom" is satisfied by an arene doing nothing, so
    R10.2 used to absorb every Friedel-Crafts acylation. Prove the match still happens and it
    is the label check that refuses it."""
    friedel_crafts = smiles(
        "[CH3:1][C:2](=[O:3])[Cl:4].[cH:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1"
        ">>[CH3:1][C:2](=[O:3])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1"
    )
    blind = classify_coverage(friedel_crafts, rules, check_labels=False)
    assert "R10.2" in blind.rule_ids
    assert not classify_coverage(friedel_crafts, rules).covered


# --- the CLI filter -----------------------------------------------------------------------


@pytest.mark.parametrize("keep", ["uncovered", "covered"])
def test_coverage_file_keeps_the_requested_side_verbatim(keep, tmp_path):
    output = tmp_path / f"{keep}.smi"
    written, read = cli.coverage_file(str(FIXTURE), str(output), keep=keep)
    assert read == len(RECORDS)
    wanted = {
        name for _, name, expected in RECORDS if bool(expected) is (keep == "covered")
    }
    lines = output.read_text(encoding="utf-8").splitlines()
    assert written == len(lines) == len(wanted)
    assert {line.split("\t")[1] for line in lines} == wanted
    assert set(lines) <= set(FIXTURE.read_text(encoding="utf-8").splitlines())


def test_coverage_file_keeps_a_record_it_cannot_parse(tmp_path):
    """Never drop training data on a valence quirk."""
    source = tmp_path / "in.smi"
    source.write_text("not a smiles at all\tjunk\n", encoding="utf-8")
    output = tmp_path / "out.smi"
    assert cli.coverage_file(str(source), str(output), keep="uncovered") == (1, 1)
