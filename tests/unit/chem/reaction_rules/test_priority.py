"""Tests for ``synplan.chem.reaction_rules.priority``."""

import pytest

from synplan.chem.reaction_rules import (
    POLICY_SOURCE_NAME,
    PrioritySmartsError,
    parse_priority_rules,
)

# A real Ugi 4CR retrosynthetic SMARTS (one of the rules used by tutorial 13).
_VALID_UGI_SMARTS = (
    "[C;D3:1](-[N;D2:2]-[C;D3:3])(-[C;D3:4](-[C;D3:5])-[N;D3:7](-[C;D1:8])"
    "-[C;D3:9](=[O;D1:10])-[C;D2:11])=[O;D1:12]"
    ">>[C;D2:4](-[C;D3:5])=[O;D1:6].[N;D1:7]-[C;D1:8]"
    ".[C;D3:9](=[O;D1:10])(-[C;D2:11])-[O;D1:12].[C;D1-:1]#[N;D2+:2]-[C;D3:3]"
)


def test_parse_priority_rules_ok():
    result = parse_priority_rules({"ugi": [_VALID_UGI_SMARTS]})
    assert set(result) == {"ugi"}
    assert len(result["ugi"]) == 1
    # Returned objects are chython Reactors with the standard pattern surface.
    rule = result["ugi"][0]
    assert hasattr(rule, "_patterns")
    assert len(rule._patterns) >= 1


def test_parse_priority_rules_reports_set_and_index():
    """A bad SMARTS surfaces as PrioritySmartsError with the offending location."""
    with pytest.raises(PrioritySmartsError) as exc:
        parse_priority_rules(
            {
                "ugi": [_VALID_UGI_SMARTS],
                "broken": [_VALID_UGI_SMARTS, "this is not a SMARTS at all"],
            }
        )
    msg = str(exc.value)
    assert "priority_rules['broken'][1]" in msg
    assert "this is not a SMARTS at all" in msg
    # Diagnostic line is always present (says either "RDKit not installed",
    # "RDKit also cannot parse", or "RDKit parses ... dialect mismatch").
    assert "RDKit" in msg


def test_parse_priority_rules_multiple_sets():
    result = parse_priority_rules(
        {"a": [_VALID_UGI_SMARTS], "b": [_VALID_UGI_SMARTS, _VALID_UGI_SMARTS]}
    )
    assert list(result) == ["a", "b"]
    assert len(result["a"]) == 1
    assert len(result["b"]) == 2


def test_parse_priority_rules_rejects_reserved_policy_name():
    with pytest.raises(ValueError, match="reserved"):
        parse_priority_rules({POLICY_SOURCE_NAME: [_VALID_UGI_SMARTS]})


def test_parse_priority_rules_rejects_empty_set():
    with pytest.raises(ValueError, match="empty"):
        parse_priority_rules({"ugi": []})


def test_parse_priority_rules_rejects_non_string_key():
    with pytest.raises(ValueError, match="non-empty strings"):
        parse_priority_rules({42: [_VALID_UGI_SMARTS]})  # type: ignore[dict-item]
