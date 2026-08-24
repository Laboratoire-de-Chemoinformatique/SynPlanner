"""A ring rule reaches the reactor as its hand-authored reagent form, or not at all.

The failure this guards is silent, not loud: chython's patcher accepts an RHS atom with no map
number and hands back the INTACT target plus a free fragment — a chemically plausible,
purchasable, completely wrong disconnection that raises nothing.
"""

from __future__ import annotations

import json

import pytest
from chython import smiles

from synplan.chem.reaction.reactor import apply_reaction_rule
from synplan.chem.reaction.rules import parse_priority_rules, rule_query_pattern
from synplan.chem.reaction.rules.synthon import (
    SYNTHON_SOURCE_NAME,
    _records,
    _rule_smarts,
    synthon_priority_rules,
)
from synplan.chem.synthon.config import SynthonConfig, load_data
from synplan.chem.synthon.rules.validate_retro import (
    RING_RULES_PATH,
    TAUTOMER_WARNING,
    check_retro_rule,
    check_ring_rules,
)

BENZIMIDAZOLE = (
    "[n;+0:1]1[c:2][n;+0:3][c:4]2[c:5][c:6][c:7][c:8][c:9]12"
    ">>[N:1][c:9]1[c:8][c:7][c:6][c:5][c:4]1[N:3].[C:2]=[O:20]"
)
BENZIMIDAZOLE_TARGET = "c1ccc(-c2nc3ccccc3[nH]2)cc1"


@pytest.fixture(scope="module")
def ring_records() -> list[dict]:
    with open(RING_RULES_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _molecule(smi: str):
    mol = smiles(smi)
    mol.canonicalize()
    return mol


def test_every_authored_retro_form_validates(ring_records: list[dict]) -> None:
    """The authoring gate, run over whatever is in `ring_rules.json` today."""
    failed = [c for c in check_ring_rules(ring_records) if c.authored and not c.ok]
    assert not failed, [(c.rule_id, c.reason) for c in failed]


@pytest.mark.parametrize(
    ("rule_id", "expected"),
    [
        ("R16.1a", ["c1ccccc1C#C", "c1ccccc1CN=[N+]=[N-]"]),
        ("R16.2b", ["[N-]=[N+]=[N-]", "c1cc(ccc1)C#N"]),
        ("R17.70", ["c1cc(ccc1)C(O)=O", "c1ccc(N)c(N)c1"]),
    ],
)
def test_the_seeded_rules_give_the_reagents_not_the_synthons(
    ring_records: list[dict], rule_id: str, expected: list[str]
) -> None:
    """The whole point: an azide and an ALKYNE, not a triazene and a styrene."""
    record = next(r for r in ring_records if r["id"] == rule_id)
    check = check_retro_rule(record)
    assert check.ok, check.reason
    assert sorted(check.all_products) == sorted(expected)


def test_an_unmapped_rhs_atom_is_refused() -> None:
    """`[O]` instead of `[O:20]` returns the intact benzimidazole and a free formaldehyde."""
    record = {
        "id": "fixture",
        "example_target": BENZIMIDAZOLE_TARGET,
        "retro_smarts": BENZIMIDAZOLE.replace("[O:20]", "[O]"),
    }
    check = check_retro_rule(record)
    assert not check.ok
    assert "no map number" in check.reason
    assert check_retro_rule({**record, "retro_smarts": BENZIMIDAZOLE}).ok


def test_the_ring_survival_check_refuses_it_on_its_own() -> None:
    """The behavioural half must reject the same rule, not only the map-number bookkeeping.

    Written against the reactor rather than through `check_retro_rule`, because the textual
    check fires first there — and a rule that re-forms its ring with every atom properly mapped
    would slip past that one.
    """
    broken = BENZIMIDAZOLE.replace("[O:20]", "[O]")
    rule = parse_priority_rules({"r": [broken]}, automorphism_filter=True)["r"][0]
    pattern = rule_query_pattern(rule)
    target = _molecule(BENZIMIDAZOLE_TARGET)
    products = [p for group in apply_reaction_rule(target, rule) for p in group]
    assert any(pattern < product for product in products)


def test_a_tautomer_degenerate_target_is_warned_about_but_still_passes() -> None:
    """The third silent failure: chython picks the tautomer, so the map placement is its call.

    A warning, not a refusal — the products can still be right, they just cannot be trusted
    unread. R16.3a's N-substituted pyrazole has no mobile proton and must stay quiet.
    """
    records = {r["id"]: r for r in json.loads(RING_RULES_PATH.read_text())}
    warned = check_retro_rule(records["R16.4b"])
    assert warned.ok and warned.warnings == (TAUTOMER_WARNING,)
    quiet = check_retro_rule(records["R16.3a"])
    assert quiet.ok and quiet.warnings == ()


def test_a_swapped_product_identity_is_refused() -> None:
    """The fourth silent failure: right reagent classes, wrong partners.

    Ring breaks, maps are unique, both products are stable and stocked — only the identity is
    wrong. `expected_reagents` is the only thing that sees it.
    """
    records = {r["id"]: r for r in json.loads(RING_RULES_PATH.read_text())}
    record = records["R16.9"]
    assert check_retro_rule(record).ok
    swapped = check_retro_rule(
        {**record, "expected_reagents": ["c1ccccc1C(C)=O", "CC(N)=O"]}
    )
    assert any(w.startswith("IDENTITY MISMATCH") for w in swapped.warnings)


def test_a_record_without_expected_reagents_is_warned_not_failed() -> None:
    """The field is being populated rule by rule; an unrecorded rule must still pass."""
    records = {r["id"]: r for r in json.loads(RING_RULES_PATH.read_text())}
    bare = {k: v for k, v in records["R16.9"].items() if k != "expected_reagents"}
    check = check_retro_rule(bare)
    assert check.ok
    assert any("identity unchecked" in w for w in check.warnings)


def test_a_duplicate_rhs_map_number_is_refused() -> None:
    check = check_retro_rule(
        {
            "id": "fixture",
            "example_target": BENZIMIDAZOLE_TARGET,
            "retro_smarts": BENZIMIDAZOLE.replace("[O:20]", "[O:3]"),
        }
    )
    assert not check.ok
    assert "duplicate RHS map numbers" in check.reason


def test_a_ring_rule_without_a_retro_form_never_reaches_the_reactor() -> None:
    """An unauthored ring rule must drop out, not arrive uncapped and propose a styrene."""
    data = load_data(SynthonConfig().rules_path)
    unauthored = {
        r["id"] for r in data["disconnections"] if r["ring"] and not r["retro_smarts"]
    }
    assert unauthored, "no unauthored ring rule left — pick another negative fixture"
    selected = {r["id"] for r in _records(SynthonConfig(), macro=False)}
    assert not selected & unauthored
    assert {
        r["id"] for r in data["disconnections"] if r["ring"]
    } - unauthored <= selected


def test_a_ring_rule_is_loaded_verbatim_and_never_capped() -> None:
    """`cap` is an acyclic-path knob; the ring rules must be byte-identical either way."""
    record = next(r for r in _records(SynthonConfig(), macro=False) if r["ring"])
    leaving_groups = load_data(SynthonConfig().rules_path)["leaving_groups"]
    assert _rule_smarts(record, leaving_groups, cap=True) == record["retro_smarts"]
    assert _rule_smarts(record, leaving_groups, cap=False) == record["retro_smarts"]

    rules = synthon_priority_rules()[SYNTHON_SOURCE_NAME]
    ring = next(rule for rule in rules if rule.rule_id == record["id"])
    assert rule_query_pattern(ring) is not None
