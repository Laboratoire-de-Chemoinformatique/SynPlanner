"""Tests for loss-aware Chython-to-RDKit SMARTS conversion."""

from __future__ import annotations

from synplan.chem.reaction.rules.representation import (
    chython_rule_smarts_to_rdkit_smarts,
    rdkit_rule_smarts_to_chython_smarts,
    roundtrip_chython_rdkit_chython,
)


def test_conversion_preserves_atom_maps_and_charge():
    result = chython_rule_smarts_to_rdkit_smarts(
        "[c:1]-[N:2]>>[c:1]-[N+:2](-[O-:3])=[O:4]"
    )

    assert result.ok
    assert result.atom_map_status == "ok"
    assert result.rdkit_smarts == "[c:1]-[N:2]>>[c:1]-[N+:2](-[O-:3])=[O:4]"
    assert result.warnings == ()


def test_conversion_preserves_explicit_hydrogen_and_degree_queries():
    result = chython_rule_smarts_to_rdkit_smarts(
        "[C;D3;H1:1]=[O:2]>>[C;D4;H2:1]-[O;H1:2]"
    )

    assert result.ok
    assert result.warnings == ()
    assert "[C;D3;H1:1]" in result.rdkit_smarts
    assert "[C;D4;H2:1]" in result.rdkit_smarts
    assert "[O;H1:2]" in result.rdkit_smarts


def test_conversion_handles_reagents_and_multiple_fragments():
    result = chython_rule_smarts_to_rdkit_smarts("[C:1].[O:2]>[Na+:3]>[C:1]-[O:2]")

    assert result.ok
    assert result.rdkit_smarts == "[C:1].[O:2]>[Na+:3]>[C:1]-[O:2]"


def test_strict_mode_fails_on_unverified_chython_query_semantics():
    result = chython_rule_smarts_to_rdkit_smarts("[C;R2:1]>>[C:1]")

    assert not result.ok
    assert result.parse_status == "semantic_loss"
    assert any("rings_count" in warning for warning in result.warnings)
    assert any("strict semantic warning" in error for error in result.errors)


def test_non_strict_mode_reports_unverified_semantics_without_failing():
    result = chython_rule_smarts_to_rdkit_smarts("[C;R2:1]>>[C:1]", strict=False)

    assert result.ok
    assert not result.is_lossless
    assert any("rings_count" in warning for warning in result.warnings)
    assert result.errors == ()


def test_invalid_smarts_returns_clear_failed_result():
    result = chython_rule_smarts_to_rdkit_smarts("not a reaction")

    assert not result.ok
    assert result.parse_status == "chython_parse_failed"
    assert result.rdkit_smarts == ""
    assert result.errors


def test_rdkit_to_chython_normalizes_retrochimera_wrappers_and_conjunctions():
    result = rdkit_rule_smarts_to_chython_smarts(
        "([c:1]-[N&H2&+0&D1:2])>>([c:1]-[N&H0&+&D3:2](=O)-[O&-])"
    )

    assert result.ok
    assert result.rdkit_smarts == ("[c:1]-[N;H2;D1:2]>>[c:1]-[N;H0;+;D3:2](=O)-[O;-]")
    assert result.chython_smarts == result.rdkit_smarts
    assert result.atom_map_status == "ok"
    assert any("normalized" in warning for warning in result.warnings)


def test_rdkit_to_chython_normalizes_neutral_charge_without_losing_maps():
    result = rdkit_rule_smarts_to_chython_smarts(
        "([O&H0&D1:1]=[C:2]-[O&H1&+0&D1:3])>>([O&H0&D1:1]=[C:2]-[O&H0&+0&D2:3]-C)"
    )

    assert result.ok
    assert result.rdkit_smarts == (
        "[O;H0;D1:1]=[C:2]-[O;H1;D1:3]>>[O;H0;D1:1]=[C:2]-[O;H0;D2:3]-C"
    )
    assert result.chython_smarts == result.rdkit_smarts
    assert result.atom_map_status == "ok"


def test_rdkit_to_chython_preserves_maps_reagents_and_query_constraints():
    result = rdkit_rule_smarts_to_chython_smarts(
        "[C;D3;H1:1]=[O:2]>[Na+:3]>[C;D4;H2:1]-[O;H1:2]"
    )

    assert result.ok
    assert result.atom_map_status == "ok"
    assert result.chython_smarts == ("[C;D3;H1:1]=[O:2]>[Na+:3]>[C;D4;H2:1]-[O;H1:2]")


def test_rdkit_to_chython_reports_expected_roundtrip_mismatch():
    result = rdkit_rule_smarts_to_chython_smarts(
        "[C:1]>>[C:1]",
        strict=False,
        expected_chython_smarts="[N:1]>>[N:1]",
    )

    assert result.ok
    assert result.roundtrip_equal is False
    assert any("does not match expected" in warning for warning in result.warnings)


def test_rdkit_to_chython_ignores_radical_annotation_roundtrip_loss():
    result = rdkit_rule_smarts_to_chython_smarts(
        "[C:1]>>[C:1]",
        expected_chython_smarts="[C:1]>>[C:1] |^1:1|",
    )

    assert result.ok
    assert result.roundtrip_equal is True
    assert any("dropping radical" in warning for warning in result.warnings)


def test_rdkit_to_chython_fails_for_chython_unsupported_recursive_smarts():
    # OR across primitive types needs an expression tree the atom query cannot hold
    result = rdkit_rule_smarts_to_chython_smarts("[C,X3:1]>>[C:1]")

    assert not result.ok
    assert result.rdkit_parse_status == "ok"
    assert result.chython_parse_status == "chython_parse_failed"
    assert result.errors


def test_chython_rdkit_chython_roundtrip_exact_equality():
    rule = "[C;D3;H1:1]=[O:2]>>[C;D4;H2:1]-[O;H1:2]"

    result = roundtrip_chython_rdkit_chython(rule)

    assert result.roundtrip_equal
    assert result.chython_smarts == rule
    assert result.ok


def test_strict_roundtrip_fails_on_forward_semantic_loss():
    rule = "[C;R2:1]>>[C:1]"

    result = roundtrip_chython_rdkit_chython(rule)

    assert result.forward_parse_status == "semantic_loss"
    assert result.reverse_rdkit_parse_status == "ok"
    assert result.reverse_chython_parse_status == "ok"
    assert result.roundtrip_equal
    assert not result.ok
    assert any("rings_count" in warning for warning in result.warnings)
    assert any("strict semantic warning" in error for error in result.errors)


def test_non_strict_roundtrip_accepts_forward_semantic_warning():
    rule = "[C;R2:1]>>[C:1]"

    result = roundtrip_chython_rdkit_chython(rule, strict=False)

    assert result.forward_parse_status == "ok"
    assert result.roundtrip_equal
    assert result.ok
    assert any("rings_count" in warning for warning in result.warnings)
    assert result.errors == ()


def test_query_bond_roundtrips_but_records_rdkit_numeric_order_limitation():
    rule = "[C:1]#[N:2]>>[C-:1]~[Cu+:3].[N:2]"

    result = roundtrip_chython_rdkit_chython(rule, strict=False)

    assert result.roundtrip_equal
    assert result.chython_smarts == rule
    assert any("bond order mismatch" in error for error in result.errors)
