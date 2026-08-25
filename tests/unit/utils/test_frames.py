"""Tests for the ChemFrame wrapper and its cell depiction."""

import pandas as pd
import pytest
from chython import smiles

from synplan.utils.frames import ChemFrame, depict_value, tree_stats_frame


@pytest.fixture(scope="module")
def molecule():
    mol = smiles("c1ccccc1")
    mol.clean2d()
    return mol


@pytest.fixture
def frame(molecule):
    return ChemFrame(
        [
            {"name": "benzene", "mol": molecule},
            {"name": "other", "mol": molecule},
        ],
        depict_columns=["mol"],
    )


def test_depictable_column_renders_svg(frame):
    html = frame._repr_html_()

    assert "<svg" in html
    assert "benzene" in html


def test_plain_column_is_not_depicted(molecule):
    html = ChemFrame([{"name": "benzene", "mol": molecule}])._repr_html_()

    assert "<svg" not in html


def test_df_keeps_objects(frame, molecule):
    assert isinstance(frame.df, pd.DataFrame)
    assert frame.df["mol"].iloc[0] is molecule


def test_head_rewraps_and_keeps_depiction(frame):
    head = frame.head(1)

    assert isinstance(head, ChemFrame)
    assert len(head) == 1
    assert "<svg" in head._repr_html_()


def test_boolean_mask_rewraps_and_keeps_depiction(frame):
    masked = frame[frame.df["name"] == "benzene"]

    assert isinstance(masked, ChemFrame)
    assert len(masked) == 1
    assert "<svg" in masked._repr_html_()


def test_depict_value_uses_first_element_of_a_sequence(molecule):
    depicted = depict_value([molecule, "ignored"])

    assert depicted.startswith("<svg")
    assert "ignored" not in depicted
    assert depict_value(["x"]) == "x"


def test_depict_value_stringifies_a_scalar():
    assert depict_value(5) == "5"
    assert depict_value([]) == "[]"


def test_missing_depict_column_is_ignored(molecule):
    frame = ChemFrame(
        [{"name": "benzene", "mol": molecule}], depict_columns=["mol", "absent"]
    )

    assert "<svg" in frame._repr_html_()


class _Run:
    """The only thing tree_stats_frame asks of a tree."""

    def __init__(self, routes):
        self.routes = routes

    def to_stats_dict(self):
        return {"num_routes": self.routes}


def test_tree_stats_frame_takes_a_single_tree():
    stats = tree_stats_frame(_Run(1))

    assert isinstance(stats, pd.DataFrame)
    assert list(stats.index) == [0] and stats.index.name == "run"
    assert stats["num_routes"].tolist() == [1]


def test_tree_stats_frame_takes_an_iterable():
    stats = tree_stats_frame([_Run(1), _Run(2)])

    assert list(stats.index) == [0, 1]
    assert stats["num_routes"].tolist() == [1, 2]


def test_tree_stats_frame_takes_a_named_mapping():
    stats = tree_stats_frame({"priority": _Run(1), "policy only": _Run(2)})

    assert list(stats.index) == ["priority", "policy only"]
    assert stats.index.name == "run"
    assert stats.T["priority"].tolist() == [1]
