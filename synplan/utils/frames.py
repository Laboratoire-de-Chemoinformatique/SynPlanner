"""Tabular views whose cells may hold chython objects, depicted inline in a notebook."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from html import escape
from typing import Any, ClassVar

import pandas as pd


def depict_value(value: Any) -> str:
    """SVG for anything that can draw itself, ``str`` otherwise.

    A cell holding a sequence is depicted by its first element, which is how the
    leaving-group and mark tables have always rendered a group of equivalent hits.
    """
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    return value.depict() if hasattr(value, "depict") else str(value)


def _escape_cell(value: Any) -> str:
    """Escape a non-depictable cell; ``to_html`` cannot, since the SVG columns need raw HTML."""
    if value is None or (isinstance(value, float) and value != value):
        return value
    return escape(str(value))


class ChemFrame:
    """A DataFrame that knows which of its columns hold depictable objects.

    Subclasses set :attr:`depict_columns` and usually add a ``from_*`` constructor;
    everything else — filtering, sorting, ``to_csv`` — is delegated to the wrapped
    frame, and a delegated call that returns a frame is re-wrapped so the depiction
    survives a ``.head()`` or a boolean mask.
    """

    depict_columns: ClassVar[tuple[str, ...]] = ()
    max_display_rows: ClassVar[int] = 20
    """Rows drawn before the view truncates; a depicted row costs roughly 7 kB of SVG."""

    def __init__(
        self,
        data: pd.DataFrame | Iterable[Mapping[str, Any]],
        depict_columns: Iterable[str] | None = None,
    ) -> None:
        self._df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(list(data))
        if depict_columns is not None:
            self.depict_columns = tuple(depict_columns)

    @property
    def df(self) -> pd.DataFrame:
        """The wrapped frame, objects intact — use it when you want plain pandas."""
        return self._df

    def _repr_html_(self) -> str:
        shown = self._df.head(self.max_display_rows).copy()
        hidden = len(self._df) - len(shown)
        for column in shown.columns:
            if column in self.depict_columns:
                shown[column] = shown[column].map(depict_value)
            else:
                shown[column] = shown[column].map(_escape_cell)
        # A recursive SMARTS spells `$(...)`, which MathJax reads as inline maths and
        # renders as a highlighted failure; the ignore classes keep it off the table.
        table = shown.to_html(escape=False)
        if hidden:
            table += f"<p>... and {hidden} more rows; raise max_display_rows or use .df</p>"
        return f'<div class="tex2jax_ignore mathjax_ignore">{table}</div>'

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._df, name)
        if not callable(attr):
            return self._rewrap(attr)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self._rewrap(attr(*args, **kwargs))

        return wrapped

    def __getitem__(self, key: Any) -> Any:
        return self._rewrap(self._df[key])

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return repr(self._df)

    def _rewrap(self, value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return type(self)(value, self.depict_columns)
        return value


def tree_stats_frame(trees: Any) -> pd.DataFrame:
    """One row of :meth:`~synplan.mcts.tree.Tree.to_stats_dict` per run, for A/B comparison.

    Plain pandas, not a :class:`ChemFrame`: nothing in the stats dict is depictable, and ``.T``
    is how a two-run comparison is read.

    :param trees: One tree, an iterable of trees, or an ordered ``run name -> tree`` mapping.
    :return: A frame indexed by run name — the mapping keys, or positions otherwise.
    """
    if hasattr(trees, "to_stats_dict"):
        trees = [trees]
    items = (
        list(trees.items()) if isinstance(trees, Mapping) else list(enumerate(trees))
    )
    return pd.DataFrame(
        [tree.to_stats_dict() for _, tree in items],
        index=pd.Index([name for name, _ in items], name="run"),
    )


def demo() -> None:
    from chython import smiles

    benzene = smiles("c1ccccc1")
    benzene.clean2d()
    frame = ChemFrame([{"name": "benzene", "mol": benzene}], depict_columns=["mol"])
    html = frame._repr_html_()
    assert "<svg" in html, "depictable column should render as SVG"
    assert "benzene" in html
    assert "<svg" not in frame.df.to_html(escape=False), "df stays object-valued"
    assert len(frame.head(1)) == 1, "delegation should work"
    wide = ChemFrame([{"n": i} for i in range(50)])
    assert "more rows" in wide._repr_html_(), "a long frame must truncate its view"
    assert len(wide) == 50, "truncation is display-only"
    assert isinstance(frame.head(1), ChemFrame), "a frame result stays a ChemFrame"
    assert depict_value(["x"]) == "x" and depict_value(5) == "5"

    class _Run:
        def to_stats_dict(self):
            return {"num_routes": 1, "solved": True}

    stats = tree_stats_frame({"a": _Run(), "b": _Run()})
    assert list(stats.index) == ["a", "b"] and stats.index.name == "run"
    assert len(tree_stats_frame(_Run())) == 1, "a bare tree is one row"
    assert list(tree_stats_frame([_Run()]).index) == [0], (
        "an iterable falls back to positions"
    )
    print("frames demo ok")


if __name__ == "__main__":
    demo()
