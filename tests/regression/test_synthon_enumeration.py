"""What the enumerator must never do: lose atom state, break a ring, wander, or repeat itself."""

from time import perf_counter

import pytest
from chython import synthon_smiles
from chython.containers import SynthonContainer
from rdkit import Chem

from synplan.chem.synthon.cli import enumerate_file
from synplan.chem.synthon.config import SynthonConfig
from synplan.chem.synthon.enumeration import Enumerator, join, load_pairs, open_points
from synplan.chem.synthon.stock import SynthonRecord, write_synthon_stock

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


def _enumerate_file_case(tmp_path, config, synthons, stock_records):
    case = tmp_path / ("audited" if config.write_audit_files else "regular")
    case.mkdir()
    pathways = case / "pathways.tsv"
    pathways.write_text(
        f"target\tR-test\t{'.'.join(synthons)}\t1\t1.0000\n",
        encoding="utf-8",
    )
    stock = case / "stock.smi"
    write_synthon_stock(str(stock), stock_records)
    output = case / "products.smi"

    written = enumerate_file(str(pathways), str(output), str(stock), config)
    products = {line.split("\t", 1)[0] for line in output.read_text().splitlines()}
    return written, products


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


@pytest.mark.parametrize("audited", (False, True))
def test_enumerate_file_uses_configured_analogue_slots(tmp_path, audited):
    synthons = ("C[CH_elec]=O", "CCC[NH2_nuc]")
    records = (
        SynthonRecord("C[CH_elec]=O", ("CC(=O)Cl",), ("AcidHalides_AcylHalides",), 0),
        SynthonRecord("CC[NH2_nuc]", ("CCN",), ("Amines_Amines",), 0),
        SynthonRecord("CCC[NH2_nuc]", ("CCCN",), ("Amines_Amines",), 0),
        SynthonRecord("CCCC[NH2_nuc]", ("CCCCN",), ("Amines_Amines",), 0),
    )
    config = SynthonConfig(
        find_analogues=True,
        similarity_threshold=-1.0,
        pas_removal_direction=True,
        mw_lower=0.0,
        mw_upper=10_000.0,
        num_workers=1,
        write_audit_files=audited,
        audit_overwrite="replace",
    )

    written, products = _enumerate_file_case(tmp_path, config, synthons, records)

    assert written == 3
    assert products == {"CCCNC(=O)C", "CCNC(=O)C", "CCCCNC(=O)C"}


@pytest.mark.parametrize("audited", (False, True))
def test_enumerate_file_ro2_filter_rejects_an_empty_strict_slot(tmp_path, audited):
    synthons = ("C[CH3_elec]", "NCC(O)C[NH2_nuc]")
    records = (
        SynthonRecord("C[CH3_elec]", ("CCl",), ("Test_Electrophile",), 0),
        SynthonRecord(
            "NCC(O)C[NH2_nuc]",
            ("NCC(O)CN",),
            ("Test_Nucleophile",),
            0,
        ),
    )
    config = SynthonConfig(
        ro2_filtration=True,
        ro2_variant="paper",
        strict_availability=True,
        mw_lower=0.0,
        mw_upper=1_000.0,
        num_workers=1,
        write_audit_files=audited,
        audit_overwrite="replace",
    )

    written, products = _enumerate_file_case(tmp_path, config, synthons, records)

    assert written == 0
    assert products == set()


@pytest.mark.parametrize("audited", (False, True))
def test_enumerate_file_non_strict_mode_uses_the_pathway_synthon(tmp_path, audited):
    synthons = ("C[CH3_elec]", "NCC(O)C[NH2_nuc]")
    records = (
        SynthonRecord("C[CH3_elec]", ("CCl",), ("Test_Electrophile",), 0),
        SynthonRecord(
            "NCC(O)C[NH2_nuc]",
            ("NCC(O)CN",),
            ("Test_Nucleophile",),
            0,
        ),
    )
    config = SynthonConfig(
        ro2_filtration=True,
        ro2_variant="paper",
        strict_availability=False,
        mw_lower=0.0,
        mw_upper=1_000.0,
        num_workers=1,
        write_audit_files=audited,
        audit_overwrite="replace",
    )

    written, products = _enumerate_file_case(tmp_path, config, synthons, records)

    assert written == 1
    assert products == {"OC(CNCC)CN"}
