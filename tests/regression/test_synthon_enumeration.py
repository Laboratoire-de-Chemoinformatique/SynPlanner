"""What the enumerator must never do: lose atom state, break a ring, wander, or repeat itself."""

from time import perf_counter

import pytest
from chython import smiles, synthon_smiles
from chython.containers import SynthonContainer
from rdkit import Chem

from synplan.chem.utils import safe_canonicalization
from synplan.enumeration.synthon.config import SynthonConfig, load_data
from synplan.enumeration.synthon.enumeration import (
    Enumerator,
    join,
    load_pairs,
    open_points,
)
from synplan.enumeration.synthon.fragment import Fragmenter
from synplan.enumeration.synthon.reactor import SynthonTransformer

HYDROGEN = 1.008

# nine bifunctional synthons and a window nothing can satisfy: the deadline is the only exit
BIFUNCTIONAL = [
    "[NH2_nuc]CC[CH_elec]=O",
    "[NH2_nuc]CCC[CH_elec]=O",
    "[NH2_nuc]c1ccc([CH_elec]=O)cc1",
    "[OH_nuc]CC[CH_elec]=O",
    "[OH_nuc]CCC[CH_elec]=O",
    "[SH_nuc]CC[CH_elec]=O",
    "[NH2_nuc]CC[CH3_elec]",
    "[OH_nuc]CC[CH3_elec]",
    "[SH_nuc]CCC[CH3_elec]",
]
DIAMINES = [
    "[NH2_nuc]CC[NH2_nuc]",
    "[NH2_nuc]CCC[NH2_nuc]",
    "[NH2_nuc]CCCC[NH2_nuc]",
    "[NH2_nuc]c1ccc([NH2_nuc])cc1",
    "[OH_nuc]CC[OH_nuc]",
]
ALDEHYDES = [
    "C[CH_elec]=O",
    "CC[CH_elec]=O",
    "CCC[CH_elec]=O",
    "CCCC[CH_elec]=O",
    "c1ccccc1[CH_elec]=O",
]


def mk(smi: str) -> SynthonContainer:
    molecule = synthon_smiles(smi)
    molecule.canonicalize()
    return molecule


def point(molecule: SynthonContainer, symbol: str) -> int:
    return next(n for n, key in open_points(molecule) if key[0] == symbol)


@pytest.mark.parametrize(
    "partner,attribute,expected",
    [
        ("C[N+](C)(C)CC[NH2_nuc]", "charge", 1),
        ("[CH2]CC[NH2_nuc]", "is_radical", True),
        ("[13CH3][NH_nuc]C", "isotope", 13),
    ],
)
def test_join_carries_the_partner_atom_state(partner, attribute, expected):
    """A quaternary ammonium that comes back neutral is not a molecule anyone can order."""
    acyl = mk("C[CH_elec]=O")
    other = mk(partner)
    merged = join(acyl, point(acyl, "C"), other, point(other, "N"))
    assert expected in [getattr(atom, attribute) for _, atom in merged.atoms()]


@pytest.mark.parametrize(
    "a,symbol_a,b,symbol_b",
    [
        # the label sits on the ring nitrogen tautomer standardisation stripped of its hydrogen
        ("c1c[nH_nuc]nc1", "N", "[CH3_elec]C", "C"),
        ("c1cnc[nH_nuc]1", "N", "[CH3_elec]C", "C"),
        # the new bond is nowhere near the ring, and the ring still loses its hydrogen
        ("C[CH_elec]=O", "C", "n1n[nH_nuc]cc1[SH_nuc]", "S"),
        ("C[CH_elec]=O", "C", "c1cc[nH]c1CC[NH2_nuc]", "N"),
    ],
)
def test_join_leaves_a_molecule_rdkit_can_read(a, symbol_a, b, symbol_b):
    """The MW window filters on molecular_mass, so a stray hydrogen mis-filters the library."""
    left, right = mk(a), mk(b)
    expected = left.molecular_mass + right.molecular_mass - 2 * HYDROGEN
    merged = join(left, point(left, symbol_a), right, point(right, symbol_b))
    assert Chem.MolFromSmiles(str(merged.unlabelled())) is not None
    assert merged.molecular_mass == pytest.approx(expected, abs=0.01)


class _Reversed(frozenset):
    """A partner set that hands its members out backwards, the way a hash-seeded set may."""

    def __iter__(self):
        return iter(sorted(super().__iter__(), reverse=True))


def test_library_does_not_depend_on_partner_set_order():
    """`load_pairs` returns sets, so unsorted iteration made a capped library run-dependent."""
    stock = [
        "C[CH_elec]=O",
        "CCC[NH2_nuc]",
        "CCC[OH_nuc]",
        "CCC[SH_nuc]",
        "CCC[CH3_nuc]",
        "c1cc[nH_nuc]c1",
    ]
    config = SynthonConfig(
        mw_lower=0.0, mw_upper=10_000.0, max_products=2, max_reacted_synthons=3
    )
    pairs = load_pairs()
    hostile = {key: _Reversed(value) for key, value in pairs.items()}
    assert [str(m) for m in Enumerator(config, pairs).enumerate_library(stock)] == [
        str(m) for m in Enumerator(config, hostile).enumerate_library(stock)
    ]


def test_library_time_budget_fires_without_a_single_product():
    config = SynthonConfig(
        mw_lower=10_000.0,
        mw_upper=20_000.0,
        max_products=100,
        max_reacted_synthons=5,
        time_budget_s=0.2,
    )
    started = perf_counter()
    assert list(Enumerator(config).enumerate_library(BIFUNCTIONAL)) == []
    assert perf_counter() - started < 3.0


def test_analogue_time_budget_fires_without_a_single_product():
    pathway = DIAMINES[:3] + ALDEHYDES[:2]
    slots = {s: (DIAMINES if s in DIAMINES else ALDEHYDES) for s in pathway}
    config = SynthonConfig(
        mw_lower=0.0, mw_upper=10_000.0, max_products=10_000, time_budget_s=0.2
    )
    started = perf_counter()
    assert list(Enumerator(config).enumerate_analogues(pathway, slots)) == []
    assert perf_counter() - started < 3.0


def test_analogues_do_not_repeat_a_product_per_slot_ordering():
    """Duplicates also burn `max_products`, so the cap under-delivers by whatever it repeats."""
    pathway = ["[NH2_nuc]CC[NH2_nuc]", "C[CH_elec]=O", "CC[CH_elec]=O"]
    slots = {s: ["C[CH_elec]=O", "CC[CH_elec]=O"] for s in pathway[1:]}
    config = SynthonConfig(mw_lower=0.0, mw_upper=10_000.0, max_products=10_000)
    products = [str(m) for m in Enumerator(config).enumerate_analogues(pathway, slots)]
    assert len(products) == len(set(products))

    capped = SynthonConfig(mw_lower=0.0, mw_upper=10_000.0, max_products=3)
    under_cap = [str(m) for m in Enumerator(capped).enumerate_analogues(pathway, slots)]
    assert len(set(under_cap)) == 3


# every R16 heterocyclisation, and the target its two-bond cut has to give back
RING_EXAMPLES = {
    "R16.1": "c1ccccc1Cn1cc(-c2ccccc2)nn1",
    "R16.2": "c1ccccc1Cn1nnnc1-c1ccccc1",
    "R16.3": "Cc1cc(-c2ccccc2)n(-c2ccccc2)n1",
    "R16.4": "Cn1cc(-c2ccccc2)nc1-c1ccccc1",
    "R16.5": "Cc1cc(-c2ccccc2)on1",
    "R16.6": "c1ccccc1C(=O)c1cc(-c2ccccc2)cc(-c2ccccc2)n1",
    "R16.7": "c1ccccc1-c1ccc(-c2ccccc2)nn1",
    "R16.8": "Cn1c(-c2ccccc2)c(-c2ccccc2)c2ccccc21",
    "R16.9": "Cc1nc2ccccc2cc1-c1ccccc1",
}


def _cut_ring_example(rule_id, target_smi):
    """(fragmenter, prepared target, its synthons) for one R16 rule and its example target."""
    config = SynthonConfig()
    record = next(
        r
        for r in load_data(config.rules_path)["disconnections"]
        if r["id"] == rule_id and r["ring"]
    )
    fragmenter = Fragmenter(config)
    target = fragmenter._prepare(safe_canonicalization(smiles(target_smi)))
    rule = SynthonTransformer.from_smarts(record["smarts"])
    synthons = [safe_canonicalization(p) for p in next(iter(rule(target))).split()]
    return fragmenter, target, synthons


@pytest.mark.parametrize("rule_id,target_smi", sorted(RING_EXAMPLES.items()))
def test_no_forbidden_mark_row_kills_a_ring_synthon(rule_id, target_smi):
    """A ring synthon carries two mutually reactive labels on purpose — styrene is C:elec + C:nuc.

    `_accept` drops the WHOLE product set when a fragment's label-key set is a `forbidden_marks`
    row, and `ring=True` skips only the density caps, not this gate. So adding `[C:elec, C:nuc]`
    to the table would delete triazole from the rule set with no error and no log line. None of
    the 12 shipped rows hits any of the nine; this fails the moment one does.
    """
    fragmenter, _, synthons = _cut_ring_example(rule_id, target_smi)
    for synthon in synthons:
        keys = frozenset(
            (a.atomic_symbol, a.hybridization == 4, a.label)
            for _, a in synthon.atoms()
            if getattr(a, "_label", None) is not None
        )
        assert keys not in fragmenter.forbidden, (
            f"forbidden_marks row {sorted(keys)} is the whole label set of {synthon}, "
            f"a synthon of {rule_id}; that rule can never fire again"
        )


@pytest.mark.parametrize("rule_id,target_smi", sorted(RING_EXAMPLES.items()))
def test_a_ring_rule_round_trips_its_target(rule_id, target_smi):
    """Cut with the rule, rebuild with the enumerator, and demand the exact target back.

    The second ring bond is intramolecular, which `join` cannot express at all: without
    `close_ring` every one of these rebuilds to nothing.
    """
    config = SynthonConfig()
    fragmenter, target, synthons = _cut_ring_example(rule_id, target_smi)
    # the density caps are tuned for acyclic couplings; hydrazine carries two labels on two atoms
    assert fragmenter._accept(synthons, deep=True, parent_labels=None, ring=True)
    # labels have to survive the cut, the canonicalisation and the re-parse
    assert all(s.synthon_labels for s in synthons)
    keys = [str(s) for s in synthons]
    products = {
        str(m)
        for m in Enumerator(config).enumerate_analogues(keys, {k: [k] for k in keys})
    }
    assert str(target) in products

    off = SynthonConfig(ring_closure_sizes=())
    assert not list(Enumerator(off).enumerate_analogues(keys, {k: [k] for k in keys}))
