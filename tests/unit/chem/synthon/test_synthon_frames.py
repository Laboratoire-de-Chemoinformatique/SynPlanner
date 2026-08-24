"""rules_frame and synthons_frame: the shipped census, rule order, and stock marking."""

import pytest

from synplan.chem.reaction.rules.synthon import (
    SYNTHON_SOURCE_NAME,
    synthon_priority_rules,
)
from synplan.chem.synthon.fragment import fragment_smiles
from synplan.chem.synthon.frames import rules_frame, synthons_frame


@pytest.fixture(scope="module")
def shipped():
    return rules_frame()


@pytest.fixture(scope="module")
def dag():
    return fragment_smiles("CC(=O)NCc1ccccc1")


def test_shipped_frame_reproduces_the_rule_census(shipped):
    """Every record, macrocyclic half included, split the way the tutorials report it."""
    assert len(shipped) == 154
    assert shipped.df.groupby(["kind", "provenance"]).size().to_dict() == {
        ("acyclic", "human"): 39,
        ("macro", "human"): 39,
        ("ring", "llm"): 76,
    }


def test_the_rule_column_depicts(shipped):
    assert "<svg" in shipped.head(1)._repr_html_()


def test_a_ring_rule_shows_its_reagent_form(shipped):
    """A ring record's raw SMARTS shows two cut bonds; retro_smarts names the reagents."""
    ring = shipped.df[shipped.df["kind"] == "ring"].iloc[0]
    assert "_" not in ring["smarts"].split(">>")[1]


def test_a_rule_list_restricts_the_frame_and_keeps_its_order():
    loaded = synthon_priority_rules()[SYNTHON_SOURCE_NAME]

    frame = rules_frame(loaded)

    assert list(frame.df["id"]) == [rule.rule_id for rule in loaded]
    assert len(frame) < 154


def test_synthons_frame_has_one_row_per_synthon(dag):
    frame = synthons_frame(dag)

    assert len(frame) == sum(len(pathway.key) for pathway in dag.pathways.values())
    assert list(frame.df["pathway"]) == sorted(frame.df["pathway"])
    assert "<svg" in frame._repr_html_()


def test_without_a_stock_availability_is_unknown_not_absent(dag):
    assert synthons_frame(dag).df["in_stock"].isna().all()


def test_a_stock_marks_only_what_it_holds(dag):
    stocked = next(iter(dag.pathways.values())).key[0]

    frame = synthons_frame(dag, {stocked: set()}).df

    assert frame.loc[frame["smiles"] == stocked, "in_stock"].all()
    assert not frame.loc[frame["smiles"] != stocked, "in_stock"].any()
