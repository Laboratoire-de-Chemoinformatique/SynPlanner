"""The synthon stock: text round trip, the two indexes, Ro2 filtration and the slot lookup."""

import pytest
from chython import synthon_smiles
from pydantic import ValidationError

from synplan.chem.synthon.config import SynthonConfig
from synplan.chem.synthon.stock import (
    SynthonRecord,
    SynthonStock,
    cap_leaving_group,
    index_by_label,
    label_keys,
    load_synthon_stock,
    read_synthon_records,
    ro2_filter,
    ro2_pass,
    write_synthon_stock,
)

# paper counts the labelled NH2 as a donor and rejects on donors > 2; corrected does not
SPLIT_ON_RO2 = "NCC(O)C[NH2_nuc]"
RECORDS = (
    SynthonRecord("C[NH2_nuc]", ("CN",), ("Amines_Amines",), 0),
    SynthonRecord("CC[NH2_nuc]", ("CCN", "CCNC(=O)OC(C)(C)C"), ("Amines_Amines",), 0),
    SynthonRecord(SPLIT_ON_RO2, ("NCC(O)CN",), ("Bifunctional_Amine_Amine",), 1),
)


@pytest.fixture
def stock_file(tmp_path):
    path = tmp_path / "stock.smi"
    write_synthon_stock(str(path), RECORDS)
    return str(path)


def canonical(smi):
    molecule = synthon_smiles(smi)
    molecule.canonicalize()
    return molecule


# --- text level -------------------------------------------------------------------------


def test_a_stock_file_round_trips(stock_file):
    assert tuple(read_synthon_records(stock_file)) == RECORDS


def test_a_stock_file_round_trips_charged_building_blocks(tmp_path):
    records = (
        SynthonRecord(
            "C[CH2_nuc]",
            ("C[NH3+]", "[Na+].[O-]C=O", "CCN"),
            ("PrimaryAmines",),
            0,
        ),
    )
    path = tmp_path / "charged.smi"
    write_synthon_stock(str(path), records)

    assert tuple(read_synthon_records(str(path))) == records


def test_a_synthon_made_by_two_blocks_keeps_both(stock_file):
    assert load_synthon_stock(stock_file)["CC[NH2_nuc]"] == {
        "CCN",
        "CCNC(=O)OC(C)(C)C",
    }


def test_whitespace_separated_records_parse_too(tmp_path):
    path = tmp_path / "spaces.smi"
    path.write_text("C[NH2_nuc] CN Amines_Amines 0\n")
    assert next(read_synthon_records(str(path))).building_blocks == ("CN",)


# --- the forward index ------------------------------------------------------------------


def test_the_label_key_reads_aromaticity_off_the_graph():
    assert label_keys(canonical("c1ccccc1[NH2_nuc]")) == [("N", False, "nuc")]
    assert label_keys(canonical("c1cc[cH_elec]cc1C")) == [("C", True, "elec")]


def test_the_forward_index_is_keyed_on_the_whole_triple():
    index = index_by_label(["C[NH2_nuc]", "c1cc[cH_elec]cc1C"])
    assert index[("N", False, "nuc")] == ["C[NH2_nuc]"]
    assert index[("C", True, "elec")] == ["c1cc[cH_elec]cc1C"]


# --- the rule of two --------------------------------------------------------------------


def test_ro2_filtration_off_is_the_identity():
    assert ro2_filter([SPLIT_ON_RO2], SynthonConfig()) == [SPLIT_ON_RO2]
    assert ro2_filter([SPLIT_ON_RO2]) == [SPLIT_ON_RO2]


@pytest.mark.parametrize("variant,kept", [("paper", False), ("corrected", True)])
def test_the_variant_decides_what_the_filter_keeps(variant, kept):
    config = SynthonConfig(ro2_filtration=True, ro2_variant=variant)
    assert bool(ro2_filter([SPLIT_ON_RO2], config)) is kept
    assert ro2_pass(canonical(SPLIT_ON_RO2), variant) is kept


def test_the_filter_reaches_the_stock_loader(stock_file):
    unfiltered = load_synthon_stock(stock_file)
    filtered = load_synthon_stock(
        stock_file, SynthonConfig(ro2_filtration=True, ro2_variant="paper")
    )
    assert SPLIT_ON_RO2 in unfiltered
    assert SPLIT_ON_RO2 not in filtered
    assert set(filtered) < set(unfiltered)


def test_a_synthon_rdkit_will_not_sanitise_is_rejected_not_raised():
    """Two rows in 6424 from a real catalogue carry a double-bonded `[O-]`. One bad row must not
    take the whole stock load down."""
    broken = "C[NH2_nuc]CC(=[O-])C"
    assert ro2_filter([broken], SynthonConfig(ro2_filtration=True)) == []


# --- the slot lookup --------------------------------------------------------------------


@pytest.fixture
def stock():
    return SynthonStock(
        {
            "CCC[NH2_nuc]": {"CCCN"},
            "CC[NH2_nuc]": {"CCN"},
            "CCCC[NH2_nuc]": {"CCCCN"},
            SPLIT_ON_RO2: {"NCC(O)CN"},
        }
    )


def test_a_slot_defaults_to_the_stocked_synthon_alone(stock):
    assert stock.slots(["CCC[NH2_nuc]"]) == {"CCC[NH2_nuc]": ["CCC[NH2_nuc]"]}


def test_a_synthon_the_stock_does_not_carry_gets_an_empty_slot(stock):
    assert stock.slots(["c1ccccc1[NH2_nuc]"]) == {"c1ccccc1[NH2_nuc]": []}


def test_find_analogues_adds_the_positional_analogues(stock):
    slots = stock.slots(["CCC[NH2_nuc]"], SynthonConfig(find_analogues=True))
    assert set(slots["CCC[NH2_nuc]"]) == {
        "CCC[NH2_nuc]",
        "CC[NH2_nuc]",
        "CCCC[NH2_nuc]",
    }


def test_the_removal_direction_reaches_the_slot(stock):
    """Upstream's removal branch is unsatisfiable, so an analogue may only ever GAIN an atom.
    The flag is what lets a user reproduce that."""
    config = SynthonConfig(find_analogues=True, pas_removal_direction=False)
    slots = stock.slots(["CCC[NH2_nuc]"], config)
    assert "CC[NH2_nuc]" not in slots["CCC[NH2_nuc]"]  # the shorter one is a removal
    assert "CCCC[NH2_nuc]" in slots["CCC[NH2_nuc]"]


def test_the_similarity_threshold_reaches_the_slot(stock):
    wide = stock.slots(
        ["CCC[NH2_nuc]"], SynthonConfig(find_analogues=True, similarity_threshold=0.0)
    )
    narrow = stock.slots(
        ["CCC[NH2_nuc]"], SynthonConfig(find_analogues=True, similarity_threshold=1.0)
    )
    assert set(narrow["CCC[NH2_nuc]"]) < set(wide["CCC[NH2_nuc]"])


def test_ro2_prunes_the_slot_candidates(stock):
    config = SynthonConfig(ro2_filtration=True, ro2_variant="paper")
    assert stock.slots([SPLIT_ON_RO2], config) == {SPLIT_ON_RO2: []}
    assert stock.slots([SPLIT_ON_RO2]) == {SPLIT_ON_RO2: [SPLIT_ON_RO2]}


def test_the_analogue_index_is_built_once(stock):
    stock.slots(["CCC[NH2_nuc]"], SynthonConfig(find_analogues=True))
    first = stock._analogue_index
    stock.slots(["CC[NH2_nuc]"], SynthonConfig(find_analogues=True))
    assert stock._analogue_index is first


# --- leaving groups ---------------------------------------------------------------------


def test_the_leaving_group_is_keyed_on_symbol_case_and_token():
    assert cap_leaving_group("C", False, "elec") == "Cl"
    assert (
        cap_leaving_group("C", True, "elec") == "Br"
    )  # aromatic carbon, a different group
    assert cap_leaving_group("C", False, "nowhere") == "H"


# --- config bounds ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        {"num_workers": 0},
        {"num_workers": -2},
        {"time_budget_s": 0.0},
        {"time_budget_s": -1.0},
        {"mw_lower": -1.0},
        {"mw_lower": 900.0, "mw_upper": 100.0},
    ],
)
def test_a_meaningless_setting_is_refused(field):
    with pytest.raises(ValidationError):
        SynthonConfig(**field)


def test_an_empty_mw_window_of_zero_width_is_allowed():
    assert SynthonConfig(mw_lower=250.0, mw_upper=250.0).mw_upper == 250.0
