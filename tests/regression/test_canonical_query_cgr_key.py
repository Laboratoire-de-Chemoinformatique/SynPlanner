"""Canonical QueryCGR key is invariant under atom renumbering.

Rule extraction buckets rules by ``ExtractedRuleRecord.cgr_key``. If that key
were ``str(query_cgr)`` it would honour atom numbering and chemically
identical rules from different workers would land in distinct buckets,
fragmenting popularity counts and producing duplicate rows in the rules TSV.
The fix is to compute a canonical key that ignores atom numbering. These
tests pin that invariant.
"""

from __future__ import annotations

from chython import smiles as read_smiles

from synplan.chem.reaction_rules.extraction import (
    _make_extracted_rule_record,
    extract_rules,
)
from synplan.chem.utils import canonical_query_cgr_key
from synplan.utils.config import RuleExtractionConfig

# Fully atom-mapped Diels-Alder (ethylene + butadiene → cyclohexene).
# Single-product cycloaddition, so it survives extract_rules' multi-product
# filter; SN2-style replacements get skipped as multi-product reactions.
_MAPPED_RXN = (
    "[CH2:1]=[CH2:2].[CH:3]([CH:4]=[CH2:5])=[CH2:6]"
    ">>[CH:4]1=[CH:3][CH2:6][CH2:5][CH2:2][CH2:1]1"
)

# Same chemistry, mapping numbers shifted by +1000. Extraction emits a
# QueryCGRContainer with different numbering; canonical key must match.
_MAPPED_RXN_SHIFTED = (
    "[CH2:1001]=[CH2:1002].[CH:1003]([CH:1004]=[CH2:1005])=[CH2:1006]"
    ">>[CH:1004]1=[CH:1003][CH2:1006][CH2:1005][CH2:1002][CH2:1001]1"
)


def _first_rule(reaction_smi: str):
    reaction = read_smiles(reaction_smi)
    config = RuleExtractionConfig()
    rules, _skipped = extract_rules(config, reaction)
    assert rules, f"extract_rules returned no rules for {reaction_smi}"
    return rules[0]


def test_canonical_key_invariant_under_remap():
    """Renumbering atoms must not change the canonical key.

    ``QueryCGRContainer.remap`` cannot be called directly: its chython override
    forwards a ``copy=`` kwarg that the base ``Graph.remap`` does not accept
    (chython bug). The test exercises the invariant via two SMILES with
    different mapping offsets and compares the keys after extraction.
    """
    baseline_key = canonical_query_cgr_key(~_first_rule(_MAPPED_RXN))
    shifted_key = canonical_query_cgr_key(~_first_rule(_MAPPED_RXN_SHIFTED))

    assert isinstance(baseline_key, str) and baseline_key
    assert shifted_key == baseline_key


def test_extracted_record_uses_canonical_key():
    rule = _first_rule(_MAPPED_RXN)

    record = _make_extracted_rule_record(rule)

    assert record.cgr_key == canonical_query_cgr_key(~rule)
