"""Every deliberate difference from the RDKit reference, pinned so a dependency bump moves it."""

import pytest
from chython import smarts, smiles, synthon_smiles
from chython.periodictable import LABEL_TABLE

from synplan.chem.scaffolds import murcko_atoms
from synplan.chem.synthon.analogues import (
    analogue_key,
    census,
    find_analogues,
    index_for_analogues,
    is_analogue,
)
from synplan.chem.synthon.config import SynthonConfig, load_data
from synplan.chem.synthon.enumerate import Enumerator, load_pairs
from synplan.chem.synthon.fragment import (
    MACROCYCLE_RING,
    Fragmenter,
    fragment_smiles,
)
from synplan.chem.synthon.rules._dialect import to_chython
from synplan.chem.synthon.stock import SynthonStock, ro2_pass
from synplan.chem.synthon.synthonise import BBSynthoniser
from synplan.chem.utils import safe_canonicalization

MACROCYCLE = "O=C1CCCCCCCCCCCCN1"
# seven pathways over four rules, reagent counts 2 and 3 — enough for the sort to be non-vacuous
BRANCHY = "CN(C)C(=O)c1ccc(OCc2ccccc2)cc1"


def canonical(smi):
    molecule = synthon_smiles(smi)
    molecule.canonicalize()
    return molecule


@pytest.fixture(scope="module")
def synthoniser():
    return BBSynthoniser()


# --- the vocabulary ---------------------------------------------------------------------


def test_eight_labels_not_the_published_nine():
    """Code 11 ("electrophilic nitrogen") collapses into `elec`: marksCombinations has no N:10
    key, so on nitrogen "electrophile" already has exactly one meaning."""
    assert set(LABEL_TABLE) == {
        "elec",
        "nuc",
        "elec2",
        "nuc2",
        "neut2",
        "elec*",
        "nuc*",
        "elecB",
    }


def test_no_integer_codes_survive_into_the_shipped_data():
    data = load_data(SynthonConfig().rules_path)
    tokens = {t for pair in data["pairs"] for t in (pair[2], pair[5])}
    assert tokens <= set(LABEL_TABLE)
    assert all("slots" not in record for record in data["disconnections"])


def test_r12_3_ships_as_two_rules():
    """Upstream's `for lablesSet in Labels.split("|")` overwrites its result with no accumulator,
    so only the last alternative survives and the Heck labelling is dead code."""
    data = load_data(SynthonConfig().rules_path)
    ids = {r["id"] for r in data["disconnections"] if not r["macro"]}
    assert {"R12.3a", "R12.3b"} <= ids
    assert "R12.3" not in ids
    heck = next(r for r in data["disconnections"] if r["id"] == "R12.3a")
    suzuki = next(r for r in data["disconnections"] if r["id"] == "R12.3b")
    assert heck["smarts"] != suzuki["smarts"]  # inline tokens make the strings differ


def test_the_dead_forbidden_mark_entries_are_dropped():
    """Four entries are never emitted upstream and one collapses to a single element, which would
    ban every mono-functional umpolung-nitrogen synthon and kill R3.3 hierarchically."""
    entries = load_data(SynthonConfig().rules_path)["forbidden_marks"]
    assert len(entries) == 12
    assert all(len(entry) == 2 for entry in entries)


def test_the_suzuki_pairing_is_systematic_not_incidental():
    """F18: upstream's mark table is a whole-molecule pre-filter and the ReconstructionReaction
    SMIRKS form the bond, so a mono-halide + mono-boronate pair is blocked there and only couples
    when one side happens to carry a second compatible label. Here the table IS the join, so the
    pairing is made systematic. The SMIRKS decline C:10 + c:21, so we decline it too."""
    pairs = load_pairs()
    assert ("C", True, "elecB") in pairs[("C", True, "elec")]  # R12.1 aryl-aryl
    assert ("C", False, "elecB") in pairs[("C", False, "elec")]  # R12.2 sp2-sp2
    assert ("C", False, "elecB") in pairs[("C", True, "elec")]  # R12.6 aryl-sp3
    assert ("C", True, "elecB") not in pairs[("C", False, "elec")]


def test_the_f7_partner_row_is_added():
    pairs = load_data(SynthonConfig().rules_path)["pairs"]
    assert ["C", True, "nuc*", "C", False, "elec*"] in pairs or [
        "C",
        False,
        "elec*",
        "C",
        True,
        "nuc*",
    ] in pairs


# --- the dialect ------------------------------------------------------------------------


def test_bare_atoms_are_bracketed():
    """A bare organic-subset atom is not aliphatic in chython: `C[Cl,Br,I]` matches an ARYL
    iodide, which is 46 hits per 200 real building blocks."""
    aryl_iodide = smiles("c1cc(c(cc1)CC)I")
    aryl_iodide.canonicalize()
    assert smarts("C[Cl,Br,I]").is_substructure(aryl_iodide)
    assert not smarts(to_chython("C[Cl,Br,I]")).is_substructure(aryl_iodide)


def test_juxtaposed_recursive_primitives_are_anded():
    """Daylight juxtaposition is a high-precedence AND; chython ORs same-term constraints."""
    ported = to_chython("[N;!$(NC=O)!$(NS(=O)=O)]")
    assert ported.count(";") > "[N;!$(NC=O)!$(NS(=O)=O)]".count(";")


def test_the_ring_flag_moves_before_the_bond_order():
    assert to_chython("[#6:1]!@-[#7:2]") == "[#6:1]-!@[#7:2]"


# --- fragmentation ----------------------------------------------------------------------


def test_macrocycles_are_an_exclusive_switch():
    dag = fragment_smiles(MACROCYCLE)
    level_one = [p for p in dag.pathways.values() if p.depth == 1]
    assert level_one
    assert all(rule.startswith("MR") for p in level_one for rule in p.rules)
    deeper = [p for p in dag.pathways.values() if p.depth > 1]
    assert all(not rule.startswith("MR") for p in deeper for rule in p.rules[1:])


def test_a_macro_cut_yields_one_fragment_with_two_labels():
    dag = fragment_smiles(MACROCYCLE)
    root = next(p for p in dag.pathways.values() if p.depth == 1)
    assert len(root.key) == 1
    assert len(canonical(root.key[0]).synthon_labels) == 2


def test_a_plain_target_never_sees_the_macro_rules():
    dag = fragment_smiles("CC(=O)NCC")
    assert dag.pathways
    assert all(
        not rule.startswith("MR") for p in dag.pathways.values() for rule in p.rules
    )


def test_the_excluded_ring_sizes_guard_bites():
    """`!r3;...;!r11` is a real field in the fork, not a positive r12..r24 approximation: a
    ring-fusion atom sits in both a small ring and the macrocycle, so r12 accepts what !r6
    rejects."""
    fused = smiles("O=C1CCCCCCCc2ccccc2CN1")
    fused.canonicalize()
    guard = smarts("[c;!r3;!r4;!r5;!r6;!r7;!r8;!r9;!r10;!r11:1]")
    assert (
        len(list(smarts("[c;r12:1]").get_mapping(fused, automorphism_filter=False)))
        == 2
    )
    assert not list(guard.get_mapping(fused, automorphism_filter=False))
    assert MACROCYCLE_RING == 11


def test_the_availability_sort_prefers_fewer_reagents():
    """Upstream sorts (availabilityRate, reagentsNumber) with reverse=True, which returns the
    pathway with the MOST reagents at equal availability.

    The target has to have several pathways of DIFFERENT reagent counts tied on availability, or
    the assertion holds under any ordering.
    """
    stocked = {"C[NH_nuc]C": {"dimethylamine"}}
    dag = Fragmenter(SynthonConfig(), stocked).fragment(
        safe_canonicalization(smiles(BRANCHY))
    )
    ranks = [(-p.availability, len(p.key)) for p in dag.best_available()]
    assert len({availability for availability, _ in ranks}) > 1
    assert len({count for _, count in ranks}) > 1
    tied = {count for availability, count in ranks if availability == ranks[0][0]}
    assert len(tied) > 1  # a tie that spans reagent counts, so the second key decides
    assert ranks == sorted(ranks)


def test_the_pathway_key_is_sorted_at_every_depth():
    """F10: upstream builds the level-1 reagent-set key unsorted and the level->=2 one sorted, so
    the same multiset lands under two dict keys and is fragmented twice."""
    dag = fragment_smiles(BRANCHY)
    assert any(p.depth > 1 for p in dag.pathways.values())
    assert all(list(p.key) == sorted(p.key) for p in dag.pathways.values())


def test_max_pathways_bounds_the_dag_width():
    """Upstream bounds depth and never width, and raises the recursion limit instead."""
    wide = fragment_smiles(BRANCHY)
    narrow = Fragmenter(SynthonConfig(max_pathways=3)).fragment(
        safe_canonicalization(smiles(BRANCHY))
    )
    assert len(narrow.pathways) < len(wide.pathways)


# --- building-block synthonisation ------------------------------------------------------


def test_the_used_step_set_does_not_leak_out_of_its_branch(synthoniser):
    """`usedInds` is a shared mutable list upstream, so a step consumed in one branch is banned
    for every later sibling AND for every later building block.

    Synthonising the same block twice is the cheap witness: with the list shared, the second call
    starts with every index already spent and returns nothing.
    """
    first = synthoniser.synthonise_smiles("NCc1ccccc1C(=O)O")
    assert len(first) == 5
    assert synthoniser.synthonise_smiles("NCc1ccccc1C(=O)O") == first


def test_a_step_must_add_a_label_to_an_already_labelled_input(synthoniser):
    """Without the gate a later step may MOVE a label rather than add one, which invents a synthon
    the building block cannot make: here a second `elec*` on the pyridazine."""
    produced = synthoniser.synthonise_smiles("c1ccn(c1)-c2ccc(Cl)nn2")
    assert len(produced) == 5
    assert "c1ccn(c1)-c2nn[cH_elec*]cc2" not in produced


def test_a_label_survives_on_a_charged_or_isotopic_atom():
    """F3/F4: the mark regex `\\[\\w*:\\w*\\]` misses `+`, so `[N+:20]` is invisible to every
    upstream check, and it mis-slices `[15NH2:20]` into `KeyError: '1:20'`."""
    assert canonical("[15NH2_nuc]C").synthon_labels == {1: "nuc"}
    assert canonical("C[N+](C)(C)CC[O_nuc]").synthon_labels == {7: "nuc"}


def test_one_unparsable_catalogue_row_does_not_take_the_worker_down(synthoniser):
    """Upstream calls `exit()` in four places, which kills a ProcessPoolExecutor worker."""
    assert synthoniser.synthonise_smiles("C1CC") == {}
    assert synthoniser.synthonise_smiles("CCO")


def test_the_unreachable_block_by_block_strategy_is_gone():
    """F13: `"block by block"` reads an attribute it never assigns — an AttributeError, and
    unreachable from the CLI."""
    strategies = {r["strategy"] for r in load_data(SynthonConfig().marks_path)}
    assert strategies == {"normal", "polymer", "protecting_group", "first_as_prep"}


# --- scaffolds --------------------------------------------------------------------------


def test_the_murcko_scaffold_keeps_the_exocyclic_double_bond():
    """The naive "strip degree-1 non-ring atoms to fixpoint" loop drops an amide's `=O`; the
    scaffold keeps every atom multiply bonded to a scaffold atom."""
    molecule = safe_canonicalization(smiles("O=C(Nc1ccccc1)c1ccccc1"))
    core = murcko_atoms(molecule)
    in_a_ring = {n for ring in molecule.sssr for n in ring}
    kept_leaves = {
        n for n in core if n not in in_a_ring and len(molecule._bonds[n]) == 1
    }
    assert [molecule.atom(n).atomic_symbol for n in kept_leaves] == ["O"]
    assert all(int(b) == 2 for n in kept_leaves for b in molecule._bonds[n].values())


# --- the rule of two --------------------------------------------------------------------


def test_the_two_ro2_variants_disagree():
    """OQ2: `paper` reproduces the published Fig. 5 numbers, `corrected` applies the corrections
    the reference's own README documents and its code never calls."""
    synthon = canonical("NCC(O)C[NH2_nuc]")
    assert not ro2_pass(synthon, "paper")  # the labelled NH2 is counted as a donor
    assert ro2_pass(synthon, "corrected")


# --- max_products -----------------------------------------------------------------------


def test_max_products_bounds_every_enumeration_path():
    """Upstream hard-codes 1 000 000 on the fallback path, which is where fedratinib's ~7 M
    products came from."""
    pool = ["C[NH2_nuc]", "CC[NH2_nuc]", "CC[CH_elec]=O", "CCC[CH_elec]=O"]
    pathway = ["C[NH2_nuc]", "CC[CH_elec]=O"]
    slots = SynthonStock({s: {"bb"} for s in pool}).slots(
        pathway, SynthonConfig(find_analogues=True)
    )
    uncapped = Enumerator(SynthonConfig(mw_lower=0.0))
    assert len(list(uncapped.enumerate_library(pool))) == 4
    assert len(list(uncapped.enumerate_analogues(pathway, slots))) == 4
    capped = Enumerator(SynthonConfig(max_products=2, mw_lower=0.0))
    assert len(list(capped.enumerate_library(pool))) == 2
    assert len(list(capped.enumerate_analogues(pathway, slots))) == 2


# --- PAS --------------------------------------------------------------------------------


def test_the_element_census_is_computed_on_the_graph():
    """Upstream scans the SMILES string, so `Cl` contributes C+l and a Cl->F change lands in no
    branch at all."""
    assert census(smiles("CCCl"))["C"] == 2
    assert census(smiles("CCCl"))["Cl"] == 1


def test_the_removal_direction_fires():
    """`elif refList_qList and len(refList_qList) == 0` is unsatisfiable for any list, so the
    entire removal direction never executes upstream and an analogue may only ever GAIN."""
    bigger = canonical("CCC[NH2_nuc]")
    smaller = canonical("CC[NH2_nuc]")
    assert is_analogue(smaller, bigger)  # addition, which upstream does reach
    assert is_analogue(bigger, smaller)  # removal, which it does not
    assert not is_analogue(bigger, smaller, removal_direction=False)


def test_both_analogue_gates_are_exact():
    """The degree signature is stricter than "same types of RCs": [NH2_nuc] and [NH_nuc] are not
    interchangeable."""
    primary = canonical("CCC[NH2_nuc]")
    secondary = canonical("CC[NH_nuc]C")
    assert analogue_key(primary) != analogue_key(secondary)
    index = index_for_analogues(["CC[NH_nuc]C", "CCC[NH2_nuc]", "CCCC[NH2_nuc]"])
    assert find_analogues(primary, index) == ["CCCC[NH2_nuc]"]


def test_the_threshold_is_a_union_with_pas_and_narrows_as_it_rises():
    """The similarity branch is a union with PAS, not a replacement: `-1` disables it and leaves
    the PAS-only floor, and among thresholds >= 0 raising one narrows the set back to it."""
    query = canonical("c1ccccc1[NH2_nuc]")
    # the aminopyridine is a PAS analogue at tanimoto 0.71; the benzylamine is only a similar one
    index = index_for_analogues(
        ["c1ccccc1[NH2_nuc]", "c1ccncc1[NH2_nuc]", "c1ccccc1C[NH2_nuc]"]
    )
    floor = set(find_analogues(query, index, sim_threshold=-1.0))
    everything = set(find_analogues(query, index, sim_threshold=0.0))
    strict = set(find_analogues(query, index, sim_threshold=0.9))
    assert (
        floor < everything
    )  # a threshold of 0 accepts every candidate past both gates
    assert (
        strict == floor
    )  # nothing clears 0.9, so PAS alone answers — it is not skipped


@pytest.mark.parametrize("text", ["[C_Q]", "[C_]", "[C_nuc3]"])
def test_an_unknown_token_is_rejected(text):
    from chython.exceptions import IncorrectSmarts

    with pytest.raises(IncorrectSmarts):
        smarts(text)
