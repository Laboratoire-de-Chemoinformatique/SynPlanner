"""SynthonTransformer: label stamping, hierarchical cuts and the reactant-side raise."""

import pytest
from chython import synthon_smiles

from synplan.chem.synthon.config import SynthonConfig, load_data
from synplan.chem.synthon.transformer import (
    SynthonRuleError,
    SynthonTransformer,
    load_rules,
    query_labels,
)

AMIDE = (
    "[C;$([C]([#7])(=[#8])[#6]):1]-!@[#7;A;+0;$([#7;D2]([#6])[#6]),"
    "$([#7;D3]([#6])([#6])[#6]):2]>>[#6_elec:1].[#7_nuc:2]"
)


def canonical(smi):
    molecule = synthon_smiles(smi)
    molecule.canonicalize()
    return molecule


def test_labels_come_from_the_product_template():
    rule = SynthonTransformer.from_smarts(AMIDE)
    assert rule._synthon_labels == {1: "elec", 2: "nuc"}


def test_a_reactant_side_label_raises():
    # QueryElement.__eq__ never consults the label, so it would be silently inert
    with pytest.raises(SynthonRuleError):
        SynthonTransformer.from_smarts("[C_elec:1]-[Cl:2]>>[#6:1]")


def test_the_cut_stamps_both_ends():
    products = list(
        SynthonTransformer.from_smarts(AMIDE)(canonical("CC(=O)N(C)Cc1ccccc1"))
    )
    assert len(products) == 1
    fragments = products[0].split()
    assert len(fragments) == 2
    assert sorted(t for f in fragments for t in f.synthon_labels.values()) == [
        "elec",
        "nuc",
    ]


def test_a_second_cut_keeps_the_first_label():
    rule = SynthonTransformer.from_smarts("[C;$([C](=[O])[Cl]):1]-[Cl:2]>>[#6_elec:1]")
    out = list(rule(canonical("C(CO[CH3_nuc])C(=O)Cl")))
    assert out and set(out[0].synthon_labels.values()) == {"elec", "nuc"}


def test_a_product_only_atom_gets_its_token():
    rule = SynthonTransformer.from_smarts(
        "[C;$([C](=[O])[Cl]):1]-[Cl:2]>>[#6_elec:1]-[#8_nuc:3]"
    )
    out = list(rule(canonical("CCCC(=O)Cl")))
    assert out and sorted(out[0].synthon_labels.values()) == ["elec", "nuc"]


def test_all_shipped_rules_load_and_keep_their_tokens():
    data = load_data(SynthonConfig().rules_path)
    normal = [r for r in data["disconnections"] if not r["macro"] and not r["ring"]]
    macro = [r for r in data["disconnections"] if r["macro"]]
    ring = [r for r in data["disconnections"] if r["ring"]]
    assert len(normal) == 39  # R12.3 ships as R12.3a (Heck) and R12.3b (Suzuki)
    assert len(macro) == 39
    assert len(ring) == 9  # the R16 heterocyclisations, which have no macrocyclic twin
    assert not any(r["macro"] for r in ring)
    for record, rule in load_rules(normal + macro + ring):
        # the SMARTS writer drops the token, so the source string is the artefact
        assert query_labels(smarts_of(record)) == rule._synthon_labels


def smarts_of(record):
    from chython import smarts

    return smarts(record["smarts"].split(">>", 1)[1])


def test_every_bb_program_step_loads():
    records = load_data(SynthonConfig().marks_path)
    assert len(records) == 147
    steps = sum(len(r["steps"]) for r in records)
    assert steps == 389
    for record in records:
        for step in record["steps"]:
            for variant in step["variants"]:
                SynthonTransformer.from_smarts(variant)
