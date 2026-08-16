"""Four defects found in the fresh Synt-On port: one per test, each red before its fix."""

import pytest
from chython import smiles, synthon_smiles

from synplan.chem.utils import safe_canonicalization
from synplan.enumeration.synthon.enumeration import Enumerator, load_pairs
from synplan.enumeration.synthon.fragment import Fragmenter
from synplan.enumeration.synthon.synthonise import BBSynthoniser

# 5-nitrothiophene-2-carbaldehyde, from the real catalogue: Bifunctional_Aldehyde_Nitro is one of
# the five programs whose only `|No|` step is the nitro reduction
NITRO_BB = "[O-][N+](=O)c1sc(C=O)cc1"
# building blocks whose raw transformer output is a non-canonical tautomer
DRIFTING_BBS = ("n1ncc[nH]1", "Ic1[nH]c(C)nc1", "O=Cc1c[nH]nc1")


@pytest.fixture(scope="module")
def synthoniser() -> BBSynthoniser:
    return BBSynthoniser()


def test_suzuki_pairs_join_elec_to_elecb():
    """D1: R12.1/12.2/12.6 emit elec + elecB, so the table has to join them."""
    pairs = load_pairs()
    assert ("C", True, "elecB") in pairs[("C", True, "elec")]  # R12.1, aryl-aryl
    assert ("C", False, "elecB") in pairs[("C", False, "elec")]  # R12.2, sp2-sp2
    assert ("C", False, "elecB") in pairs[("C", True, "elec")]  # R12.6, aryl-sp3
    # no ReconstructionReaction licenses an aliphatic electrophile onto an aryl boronate
    assert ("C", True, "elecB") not in pairs[("C", False, "elec")]


def test_a_biaryl_survives_the_round_trip():
    """D1, end to end: without the row the R12.1 pathway enumerates to nothing."""
    target = safe_canonicalization(smiles("OC(=O)c1ccccc1-c1ccccc1"))
    dag = Fragmenter().fragment(target)
    enumerator = Enumerator()
    rebuilt = {
        str(product)
        for pathway in dag.children[()]
        for product in enumerator.enumerate_analogues(
            pathway, {synthon: [synthon] for synthon in pathway}
        )
    }
    assert dag.target in rebuilt


def test_the_nitro_reduction_is_still_a_section_boundary(synthoniser):
    """D2: `|No|` splits the program whatever the reaction is; the nitro exemption only decides
    whether the protected form is kept."""
    program = synthoniser.programs["Bifunctional_Aldehyde_Nitro"]
    assert sum(step.is_pg_removal for step in program.steps) == 1
    assert [step.keeps_protected for step in program.steps] == [False, True, False]

    synthons = synthoniser.synthonise_smiles(NITRO_BB)
    # the deprotect-and-relabel stage emits the aniline nucleophiles the reference publishes
    assert "c1(sc([CH3_elec])cc1)[NH2_nuc]" in synthons
    assert "[NH2_nuc2]c1ccc([CH3_elec])s1" in synthons
    # and no unlabelled molecule leaks into the stock
    assert all("_" in key for key in synthons)


def test_stock_keys_are_canonical(synthoniser):
    """D3: `Fragmenter` looks synthons up by their canonical SMILES, so a raw key never matches."""
    keys = {key for bb in DRIFTING_BBS for key in synthoniser.synthonise_smiles(bb)}
    assert len(keys) == 9
    for key in keys:
        assert key == str(safe_canonicalization(synthon_smiles(key))), key


def test_an_azole_gives_up_only_one_nh(synthoniser):
    """D4: chython moves the acidic H off the labelled nitrogen, so the azole hook used to label
    the same tautomeric site twice and the pair rebuilt to a quaternary N."""
    for bb in ("c1cc[nH]n1", "c1cnc[nH]1", "c1ccc(-c2cc[nH]n2)cc1"):
        assert len(synthoniser.synthonise_smiles(bb)) == 1, bb
    # the reference's own answer for 3-methylpyrazole, one nucleophile on the N-H
    assert set(synthoniser.synthonise_smiles("Cc1cc[nH]n1")) == {"n1c(cc[nH_nuc]1)C"}
