"""Tests for synplan.chem.reaction_rules.analysis."""

from __future__ import annotations

from pathlib import Path

import pytest
from chython.containers.reaction import ReactionContainer

# -- Fixture: minimal TSV file with 3 rules --------------------------------

SAMPLE_TSV = """\
rule_smarts\tpopularity\treaction_indices
[C;D3:1]-[O;D1:2]>>[C;D3:1]-[O;D2:2]-[C;D1:3]\t100\t1,2,3,4,5
[C;D3:1]-[N;D1:2]>>[C;D3:1]-[N;D3+:2](-[O;D1-:3])=[O;D1:4]\t50\t10,11,12
[C;D3:1]-[C;D3:2](=[O;D1:4])-[N;D2:6]-[C;D3:5]>>[C;D3:1]-[C;D3:2](=[O;D1:4])-[O;D1:3].[N;D1:6]-[C;D3:5]\t25\t20,21
"""


@pytest.fixture()
def sample_tsv(tmp_path: Path) -> Path:
    p = tmp_path / "rules.tsv"
    p.write_text(SAMPLE_TSV, encoding="utf-8")
    return p


# -- Tests ------------------------------------------------------------------


class TestRuleSetFromTsv:
    def test_loads_correct_count(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        assert len(rs) == 3

    def test_rules_are_reaction_containers(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        for rule in rs:
            assert isinstance(rule, ReactionContainer)

    def test_popularity_loaded(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        assert rs.popularity == (100, 50, 25)

    def test_reaction_indices_loaded(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        assert rs.reaction_indices[0] == (1, 2, 3, 4, 5)
        assert rs.reaction_indices[1] == (10, 11, 12)

    def test_smarts_strings_preserved(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        assert "[C;D3:1]-[O;D1:2]>>" in rs.smarts_strings[0]


class TestRuleSetIndexing:
    def test_int_index_returns_reaction(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        assert isinstance(rs[0], ReactionContainer)

    def test_slice_returns_ruleset(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        sliced = rs[:2]
        assert isinstance(sliced, RuleSet)
        assert len(sliced) == 2
        assert sliced.popularity == (100, 50)

    def test_negative_index(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        assert isinstance(rs[-1], ReactionContainer)

    def test_list_index_returns_ruleset(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        subset = rs[[0, 2]]
        assert isinstance(subset, RuleSet)
        assert len(subset) == 2
        assert subset.popularity == (100, 25)


class TestRuleSetRepr:
    def test_repr_html_returns_string(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        html = rs._repr_html_()
        assert isinstance(html, str)
        assert "<table" in html
        assert "<svg" in html

    def test_repr_html_shows_popularity(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        html = rs._repr_html_()
        assert "100" in html
        assert "50" in html


class TestRuleSetDataFrame:
    def test_to_dataframe_basic(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        df = rs.to_dataframe()
        assert len(df) == 3
        assert "smarts" in df.columns
        assert "popularity" in df.columns
        assert "n_reactions" in df.columns
        assert list(df["popularity"]) == [100, 50, 25]

    def test_to_dataframe_with_svg(self, sample_tsv: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        rs = RuleSet.from_tsv(sample_tsv)
        df = rs.to_dataframe(include_svg=True)
        assert "svg" in df.columns
        assert "<svg" in df["svg"].iloc[0]


class TestRuleSetErrorHandling:
    def test_skips_unparseable_smarts(self, tmp_path: Path):
        from synplan.chem.reaction_rules.analysis import RuleSet

        tsv = tmp_path / "bad.tsv"
        tsv.write_text(
            "rule_smarts\tpopularity\treaction_indices\n"
            "[C;D3:1]-[O;D1:2]>>[C;D3:1]-[O;D2:2]-[C;D1:3]\t10\t1\n"
            "INVALID_SMARTS_HERE\t5\t2\n",
            encoding="utf-8",
        )
        rs = RuleSet.from_tsv(tsv)
        assert len(rs) == 1
        assert rs.popularity == (10,)
