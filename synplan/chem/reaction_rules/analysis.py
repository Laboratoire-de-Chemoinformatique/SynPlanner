"""Utilities for loading and inspecting extracted reaction rules."""

import logging
from collections.abc import Sequence
from pathlib import Path

from chython import smarts
from chython.containers.reaction import ReactionContainer
from chython.exceptions import IncorrectSmiles
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class RuleSet:
    """A collection of reaction rules loaded from a TSV file.

    Provides iteration, indexing, slicing, Jupyter SVG rendering,
    and optional pandas DataFrame export.

    Attributes
    ----------
    rules : tuple[ReactionContainer, ...]
        Parsed reaction rules.
    smarts_strings : tuple[str, ...]
        Original SMARTS strings from the TSV.
    popularity : tuple[int, ...]
        Number of source reactions per rule.
    reaction_indices : tuple[tuple[int, ...], ...]
        Source reaction indices per rule.
    """

    def __init__(
        self,
        rules: Sequence[ReactionContainer],
        smarts_strings: Sequence[str],
        popularity: Sequence[int],
        reaction_indices: Sequence[tuple[int, ...]],
    ):
        self.rules = tuple(rules)
        self.smarts_strings = tuple(smarts_strings)
        self.popularity = tuple(popularity)
        self.reaction_indices = tuple(reaction_indices)

    @classmethod
    def from_tsv(cls, tsv_path: str | Path) -> "RuleSet":
        """Load rules from a SynPlanner extraction TSV file.

        TSV columns (tab-separated, with header):
            rule_smarts    popularity    reaction_indices

        Rules that fail to parse are skipped with a warning.

        :param tsv_path: Path to the TSV file.
        :return: RuleSet instance.
        """
        rules: list[ReactionContainer] = []
        smarts_strs: list[str] = []
        pops: list[int] = []
        indices: list[tuple[int, ...]] = []
        n_errors = 0

        with open(tsv_path, encoding="utf-8") as f:
            f.readline()  # skip header
            for line in tqdm(f, desc="Loading rules"):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                smarts_str = parts[0]
                pop = int(parts[1]) if len(parts) > 1 else 0
                idx = (
                    tuple(int(x) for x in parts[2].split(",") if x.strip())
                    if len(parts) > 2
                    else ()
                )
                try:
                    rule = smarts(smarts_str)
                    rules.append(rule)
                    smarts_strs.append(smarts_str)
                    pops.append(pop)
                    indices.append(idx)
                except IncorrectSmiles:
                    n_errors += 1

        if n_errors:
            logger.warning("Failed to parse %d rules from %s", n_errors, tsv_path)

        return cls(rules, smarts_strs, pops, indices)

    def __len__(self) -> int:
        return len(self.rules)

    def __getitem__(self, key):
        """Index, slice, or list-of-int. Returns ReactionContainer or new RuleSet."""
        if isinstance(key, int):
            return self.rules[key]
        if isinstance(key, slice):
            return RuleSet(
                self.rules[key],
                self.smarts_strings[key],
                self.popularity[key],
                self.reaction_indices[key],
            )
        if isinstance(key, (list, tuple)):
            return RuleSet(
                tuple(self.rules[i] for i in key),
                tuple(self.smarts_strings[i] for i in key),
                tuple(self.popularity[i] for i in key),
                tuple(self.reaction_indices[i] for i in key),
            )
        raise TypeError(f"Invalid key type: {type(key)}")

    def __iter__(self):
        return iter(self.rules)

    def _repr_html_(self) -> str:
        """Jupyter HTML table with inline SVG depictions."""
        max_preview = 20
        n = min(len(self.rules), max_preview)
        rows = []
        for i in range(n):
            rule = self.rules[i]
            rule.clean2d()
            svg = rule.depict()
            pop = self.popularity[i]
            rows.append(f"<tr><td>{i}</td><td>{svg}</td><td>{pop}</td></tr>")
        truncation = ""
        if len(self.rules) > max_preview:
            truncation = (
                f"<tr><td colspan='3' style='text-align:center'>"
                f"... and {len(self.rules) - max_preview} more rules</td></tr>"
            )
        return (
            f"<h4>RuleSet ({len(self.rules)} rules)</h4>"
            f"<table border='1' style='border-collapse:collapse'>"
            f"<tr><th>#</th><th>Rule</th><th>Popularity</th></tr>"
            f"{''.join(rows)}"
            f"{truncation}"
            f"</table>"
        )

    def to_dataframe(self, include_svg: bool = False):
        """Convert to pandas DataFrame.

        :param include_svg: If True, adds an 'svg' column with depiction HTML.
        :return: DataFrame with columns: smarts, popularity, n_reactions,
            and optionally svg.
        """
        import pandas as pd

        data = {
            "smarts": list(self.smarts_strings),
            "popularity": list(self.popularity),
            "n_reactions": [len(idx) for idx in self.reaction_indices],
        }
        if include_svg:
            svgs = []
            for rule in self.rules:
                rule.clean2d()
                svgs.append(rule.depict())
            data["svg"] = svgs

        return pd.DataFrame(data)
