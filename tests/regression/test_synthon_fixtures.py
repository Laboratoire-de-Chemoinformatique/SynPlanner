"""The paper's catalogue-free published numbers, end to end."""

import re
from pathlib import Path

import pytest
from chython import smiles, synthon_smiles

from synplan.chem.scaffolds import scaffold_smiles
from synplan.chem.synthon.config import SynthonConfig
from synplan.chem.synthon.enumerate import (
    Enumerator,
    join,
    load_pairs,
    open_points,
)
from synplan.chem.synthon.fragment import Fragmenter, fragment_smiles
from synplan.chem.synthon.stock import ro2_pass
from synplan.chem.synthon.synthonise import BBSynthoniser
from synplan.chem.utils import safe_canonicalization

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "synthon"

# the reference spends the atom-map field on the label; the fixtures are in that dialect
_PAPER_CODE = {
    "10": "elec",
    "11": "elec",
    "20": "nuc",
    "21": "elecB",
    "30": "elec2",
    "40": "nuc2",
    "50": "neut2",
    "60": "elec*",
    "70": "nuc*",
}
_MARK = re.compile(r"\[([^]]*?):(\d\d)\]")

CENOBAMATE = "NC(=O)OC(CN1N=CN=N1)C1=CC=CC=C1Cl"
# the seven pathways the reference README documents. Upstream raises TypeError on this molecule
# today and, once patched past that, gives five, because str.replace labelling stopped matching.
README_PATHWAYS = {
    ("R2.2_0",),
    ("R5.1_0",),
    ("R5.2_0",),
    ("R2.2_0", "R5.1_0"),
    ("R2.2_0", "R5.2_0"),
    ("R2.2_0", "R10.1_0"),
    ("R2.2_0", "R10.1_1"),
}
# what the port adds and upstream cannot: once R5.1/R5.2 has cut the N-CH2 bond, R16.2b closes the
# N-H tetrazole that is left. The shipped `[n;D3]` spelling never matched it, because chython's `D`
# counts heavy neighbours only - that is bug 1 of the ring block.
RING_PATHWAYS = {
    ("R5.1_0", "R16.2b_0"),
    ("R5.2_0", "R16.2b_0"),
    ("R2.2_0", "R5.1_0", "R16.2b_0"),
    ("R2.2_0", "R5.2_0", "R16.2b_0"),
}


def as_token(smi: str) -> str:
    return _MARK.sub(lambda m: f"[{m.group(1)}_{_PAPER_CODE[m.group(2)]}]", smi)


def canonical(smi: str) -> str:
    molecule = synthon_smiles(as_token(smi))
    molecule.canonicalize()
    return str(molecule)


def published_synthons():
    for line in (FIXTURES / "outSynth_BBmode.smi").read_text().splitlines():
        fields = line.split("\t")
        yield fields[0], {canonical(s) for s in fields[3].split(".")}


@pytest.fixture(scope="module")
def synthoniser():
    return BBSynthoniser()


@pytest.mark.parametrize("smi,expected", list(published_synthons()))
def test_published_synthons(synthoniser, smi, expected):
    """9 building blocks, 18 synthons, and which token lands on which atom."""
    assert set(synthoniser.synthonise_smiles(smi)) == expected


def test_the_fixture_totals(synthoniser):
    blocks = [
        line.split("\t", 1)[0]
        for line in (FIXTURES / "BBs.cxsmiles").read_text().splitlines()
        if line.strip()
    ]
    produced = set()
    for block in blocks:
        produced.update(synthoniser.synthonise_smiles(block))
    assert len(blocks) == 9
    assert len(produced) == 18


def test_cenobamate_gives_the_readme_pathways():
    dag = fragment_smiles(CENOBAMATE)
    assert {p.rules for p in dag.pathways.values()} == README_PATHWAYS | RING_PATHWAYS
    assert dag.is_acyclic()
    assert len(dag.roots()) == 3


def test_availability_uses_the_stock():
    dag = fragment_smiles(CENOBAMATE)
    root = next(p for p in dag.roots() if p.rules == ("R2.2_0",))
    stocked = {root.key[0]: {"a"}}
    scored = Fragmenter(SynthonConfig(), stocked).fragment(
        safe_canonicalization(smiles(CENOBAMATE))
    )
    hit = next(p for p in scored.pathways.values() if p.rules == ("R2.2_0",))
    assert 0.0 < hit.availability < 1.0
    assert all(p.availability == 0.0 for p in dag.pathways.values())


def test_the_tutorial_round_trip(synthoniser):
    blocks = [
        line.split("\t", 1)[0]
        for line in (FIXTURES / "BBs.cxsmiles").read_text().splitlines()
        if line.strip()
    ]
    synthons = set()
    for block in blocks:
        synthons.update(synthoniser.synthonise_smiles(block))
    config = SynthonConfig(mw_lower=0.0, mw_upper=10_000.0, max_products=100_000)
    produced = {str(m) for m in Enumerator(config).enumerate_library(sorted(synthons))}
    published = {
        str(safe_canonicalization(smiles(line.strip())))
        for line in (FIXTURES / "final_result.smi").read_text().splitlines()
        if line.strip()
    }
    assert len(published) == 47
    # a superset: our join is symmetric and each branch owns its used-reaction set, where
    # upstream shares one mutable set across siblings
    assert published <= produced


@pytest.mark.parametrize(
    "a,b,order",
    [
        ("C[CH_elec]=O", "c1ccccc1[NH2_nuc]", 1),
        ("CCC[CH3_elec2]", "c1ccccc1[NH2_nuc2]", 2),
        ("CCC[CH3_elec2]", "CC[CH3_nuc2]", 2),
        ("CCC[CH3_neut2]", "CC[CH3_neut2]", 2),
    ],
)
def test_join_bond_order(a, b, order):
    left, right = synthon_smiles(a), synthon_smiles(b)
    left.canonicalize()
    right.canonicalize()
    atom_a = open_points(left)[0][0]
    atom_b = open_points(right)[0][0]
    before = sum(int(bond) for *_, bond in left.bonds()) + sum(
        int(bond) for *_, bond in right.bonds()
    )
    merged = join(left, atom_a, right, atom_b)
    assert not merged.synthon_labels
    assert sum(int(bond) for *_, bond in merged.bonds()) - before == order


def test_the_f7_row_exists():
    # c:nuc* is produced by BB synthonisation but has no upstream row, so any stocked
    # aryl-BF3/MIDA/aryl-sulfinate synthon raised KeyError in the enumerator
    pairs = load_pairs()
    assert ("C", True, "nuc*") in pairs
    assert pairs[("C", True, "nuc*")] == {("C", False, "elec*"), ("C", True, "elec*")}


def test_ro2_reproduces_the_readme_example():
    synthon = synthon_smiles("O=C(O)c1c(C2CC2)csc1[NH2_nuc]")
    synthon.canonicalize()
    from rdkit.Chem import AddHs
    from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumHBA

    plain = synthon.unlabelled().to_rdkit(keep_mapping=False)
    assert round(CalcExactMolWt(AddHs(plain)), 6) == 183.035400
    assert CalcNumHBA(AddHs(plain)) == 4
    assert ro2_pass(synthon) is True


@pytest.mark.parametrize(
    "smi,expected",
    [
        ("O=C(O)c1ccc(NC(=O)OCC2c3ccccc3-c3ccccc32)cc1", "c1ccccc1"),
        ("CC1(C)OB(c2ccccc2)OC1(C)C", "c1ccccc1"),
        ("c1ccc(C2CO2)cc1", "c1ccccc1"),
        ("CC(C)(C)OC(=O)N1CCNCC1", "C1CNCCN1"),
    ],
)
def test_published_scaffold_contracts(smi, expected):
    want = smiles(expected)
    want.canonicalize()
    assert scaffold_smiles(smi) == str(want)
