"""Regression tests for the feature-owned deprotection taxonomy."""

import re
from importlib.resources import files

import pytest
from chython import smarts, smiles

from synplan.chem.building_blocks.deprotection import (
    deprotect_molecule,
    remove_protective_groups,
)
from synplan.chem.building_blocks.rules import ProtectiveRule, protective_rules
from synplan.chem.utils import safe_canonicalization


def test_rule_inventory_is_complete() -> None:
    assert len(protective_rules) == 95
    assert (
        sum(rule.policy == "conservative" for rule in protective_rules.values()) == 84
    )
    assert sum(rule.policy == "aggressive" for rule in protective_rules.values()) == 11


def test_rules_use_current_chython_heteroatom_neighbour_primitive() -> None:
    """Archived Chython 1.96 ``xN`` queries migrate exactly to 1.103 ``yN``."""
    text = (
        files("synplan.chem.building_blocks")
        .joinpath("data/protective_rules.tsv")
        .read_text(encoding="utf-8")
    )
    assert re.search(r"(?<![A-Za-z])x[0-9]", text) is None
    assert re.search(r"(?<![A-Za-z])y[0-9]", text) is not None
    for line in text.splitlines()[1:]:
        query = line.split("\t", 2)[1]
        for atom in re.findall(r"\[([^]]+)]", query):
            assert "+" in atom or "-" in atom, atom


def test_rules_preserve_archived_explicit_charge_semantics() -> None:
    unconstrained = [
        (name, atom_index, atom.atomic_symbol)
        for name, rule in protective_rules.items()
        for atom_index, atom in rule.query.atoms()
        if atom._charge is None
    ]
    assert unconstrained == []

    explicit_charges = {
        (atom.atomic_symbol, atom._charge)
        for rule in protective_rules.values()
        for _, atom in rule.query.atoms()
        if atom._charge
    }
    assert {("N", 1), ("O", -1)} <= explicit_charges


@pytest.mark.parametrize(
    ("name", "decoy"),
    [
        ("hydroxyl_tms", "CC(C)[OH+][Si](C)(C)C"),
        ("hydroxyl_tms", "CC(C)[O-][Si](C)(C)C"),
        ("amine_boc", "c1ccccc1[NH2+]C(=O)OC(C)(C)C"),
        ("amine_boc", "c1ccccc1NC(=[O-])OC(C)(C)C"),
        ("amine_boc", "c1ccccc1NC(=[O+])OC(C)(C)C"),
        ("amine_tfa", "CNC(=O)C([F-])(F)F"),
        ("amine_tfa", "CNC(=O)C([F+])(F)F"),
        ("hydroxyl_mom", "CC(C)[OH+]COC"),
    ],
)
def test_charged_decoys_do_not_match_neutral_archived_rules(name, decoy) -> None:
    molecule = safe_canonicalization(smiles(decoy, ignore=True), clean_stereo=False)
    original = str(molecule)
    changed = remove_protective_groups(
        molecule,
        policy="aggressive",
        rules={name: protective_rules[name]},
    )
    assert not changed
    assert str(molecule) == original


def test_conservative_boc_deprotection_is_copy_safe_and_idempotent() -> None:
    molecule = smiles("c1ccccc1NC(=O)OC(C)(C)C", ignore=True)
    original = str(molecule)
    result = deprotect_molecule(molecule)
    assert str(molecule) == original
    assert str(result) == str(smiles("c1ccccc1N", ignore=True))
    assert not remove_protective_groups(result)


def test_aggressive_policy_enables_benzyl_ether_rule() -> None:
    molecule = smiles("CC(C)OCc1ccccc1", ignore=True)
    conservative = deprotect_molecule(molecule, policy="conservative")
    aggressive = deprotect_molecule(molecule, policy="aggressive")
    assert str(conservative) == str(molecule)
    assert str(aggressive) == str(smiles("CC(C)O", ignore=True))


def test_decoy_is_not_deprotected() -> None:
    decoy = smiles("CC(C)OC(C)c1ccccc1", ignore=True)
    assert str(deprotect_molecule(decoy, policy="aggressive")) == str(decoy)


@pytest.mark.parametrize(
    ("name", "rule"), protective_rules.items(), ids=protective_rules
)
def test_every_rule_cleaves_its_canonicalized_reference(name, rule) -> None:
    molecule = safe_canonicalization(
        smiles(rule.protected_smiles, ignore=True), clean_stereo=False
    )
    expected = safe_canonicalization(
        smiles(rule.cleaved_smiles, ignore=True), clean_stereo=False
    )
    remove_protective_groups(molecule, policy="aggressive")
    observed = safe_canonicalization(molecule, clean_stereo=False)
    assert str(observed) == str(expected), name


_CONSERVATIVE_RULES = [
    (name, rule)
    for name, rule in protective_rules.items()
    if rule.policy == "conservative"
]


@pytest.mark.parametrize(
    ("name", "rule"),
    _CONSERVATIVE_RULES,
    ids=[name for name, _ in _CONSERVATIVE_RULES],
)
def test_every_conservative_rule_works_in_conservative_policy(name, rule) -> None:
    molecule = safe_canonicalization(
        smiles(rule.protected_smiles, ignore=True), clean_stereo=False
    )
    expected = safe_canonicalization(
        smiles(rule.cleaved_smiles, ignore=True), clean_stereo=False
    )
    remove_protective_groups(molecule, policy="conservative")
    observed = safe_canonicalization(molecule, clean_stereo=False)
    assert str(observed) == str(expected), name


_AGGRESSIVE_RULES = [
    (name, rule)
    for name, rule in protective_rules.items()
    if rule.policy == "aggressive"
]


@pytest.mark.parametrize(
    ("name", "rule"),
    _AGGRESSIVE_RULES,
    ids=[name for name, _ in _AGGRESSIVE_RULES],
)
def test_aggressive_only_rules_respect_policy_boundary(name, rule) -> None:
    molecule = safe_canonicalization(
        smiles(rule.protected_smiles, ignore=True), clean_stereo=False
    )
    original = str(molecule)
    expected = safe_canonicalization(
        smiles(rule.cleaved_smiles, ignore=True), clean_stereo=False
    )
    assert str(deprotect_molecule(molecule, policy="conservative")) == original, name
    observed = deprotect_molecule(molecule, policy="aggressive")
    assert str(safe_canonicalization(observed, clean_stereo=False)) == str(expected), (
        name
    )


def test_independent_overlapping_protective_groups_are_removed() -> None:
    molecule = safe_canonicalization(
        smiles("CC(C)(C)OC(=O)NCCO[Si](C)(C)C", ignore=True),
        clean_stereo=False,
    )
    expected = safe_canonicalization(smiles("NCCO", ignore=True), clean_stereo=False)
    assert str(deprotect_molecule(molecule)) == str(expected)


def _replacement_rule(query: str, atom_type: str) -> ProtectiveRule:
    return ProtectiveRule(
        query=smarts(query),
        keep_atoms=(1,),
        add_atoms=((1, atom_type, 1),),
        protected_smiles="",
        cleaved_smiles="",
        decoys=(),
        policy="conservative",
        decoy_scope="cycle_test",
    )


def test_deprotection_guards_maximum_passes_and_cycles() -> None:
    rules = {
        "carbon_to_nitrogen": _replacement_rule("[O:1]-[C:2]", "N"),
        "nitrogen_to_carbon": _replacement_rule("[O:1]-[N:2]", "C"),
    }
    with pytest.raises(RuntimeError, match="did not converge within 1 passes"):
        remove_protective_groups(smiles("CO"), rules=rules, max_passes=1)
    with pytest.raises(RuntimeError, match="entered a cycle"):
        remove_protective_groups(smiles("CO"), rules=rules, max_passes=5)


_RULE_DECOYS = [
    (name, rule, decoy)
    for name, rule in protective_rules.items()
    for decoy in rule.decoys
]


@pytest.mark.parametrize(
    ("name", "rule", "decoy"),
    _RULE_DECOYS,
    ids=[f"{name}-{index}" for index, (name, _, __) in enumerate(_RULE_DECOYS)],
)
def test_every_per_rule_decoy_is_preserved(name, rule, decoy) -> None:
    molecule = safe_canonicalization(smiles(decoy, ignore=True), clean_stereo=False)
    original = str(molecule)
    changed = remove_protective_groups(
        molecule, policy="aggressive", rules={name: rule}
    )
    assert not changed, name
    assert str(molecule) == original
