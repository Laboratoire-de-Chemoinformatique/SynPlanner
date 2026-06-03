"""Notebook plotting helpers for RouteCGR analysis."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from typing import Any

from chython import smiles as chython_smiles

from synplan.chem.utils import mol_from_smiles


def top_bb_usage_rows(
    bb_stats: Mapping[str, Mapping[str, Any]],
    *,
    top_n: int = 10,
    min_mol_size: int = 4,
) -> list[tuple[str, int, int]]:
    """Return `(smiles, route_count, occurrences)` rows for top BB usage."""

    rows = []
    for smiles, data in bb_stats.items():
        if len(chython_smiles(smiles)) < min_mol_size:
            continue
        rows.append((smiles, data["route_count"], data["occurrences"]))
    return sorted(rows, key=lambda row: (-row[1], -row[2], row[0]))[:top_n]


def plot_top_real_bbs(
    real_bb_stats: Mapping[str, Mapping[str, Any]],
    *,
    top_n: int = 10,
    min_mol_size: int = 4,
):
    """Plot top BB usage as a Matplotlib bar chart with SMILES tick labels."""

    import matplotlib.pyplot as plt

    rows = top_bb_usage_rows(
        real_bb_stats,
        top_n=top_n,
        min_mol_size=min_mol_size,
    )
    labels = [smiles for smiles, _, _ in rows]
    route_counts = [route_count for _, route_count, _ in rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(rows)), route_counts)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Route count")
    ax.set_title(f"Top {top_n} BBs, min_mol_size={min_mol_size}")
    fig.tight_layout()
    return fig, ax


def top_bbs_chython_svg_html(
    bb_stats: Mapping[str, Mapping[str, Any]],
    *,
    top_n: int = 10,
    min_mol_size: int = 4,
) -> str:
    """Return an HTML bar chart with Chython SVG molecule depictions."""

    rows = []
    for smiles, route_count, occurrences in top_bb_usage_rows(
        bb_stats,
        top_n=top_n,
        min_mol_size=min_mol_size,
    ):
        mol = mol_from_smiles(smiles, standardize=False, clean2d=True)
        rows.append((smiles, route_count, occurrences, mol))

    max_count = max((route_count for _, route_count, _, _ in rows), default=1)
    html = [
        """
<style>
  .bb-plot {
    display:flex;
    align-items:flex-end;
    gap:18px;
    min-height:430px;
    padding:8px 4px;
  }
  .bb-col {
    width:135px;
    text-align:center;
    font-family:Arial, sans-serif;
  }
  .bb-count { font-size:13px; margin-bottom:4px; }
  .bb-bar-wrap {
    height:190px;
    display:flex;
    align-items:flex-end;
    justify-content:center;
  }
  .bb-bar {
    width:42px;
    background:#4c78a8;
    border-radius:3px 3px 0 0;
  }
  .bb-mol {
    min-height:130px;
    display:flex;
    align-items:center;
    justify-content:center;
    overflow:visible;
  }
  .bb-mol svg {
    max-width:130px !important;
    max-height:120px !important;
    width:auto !important;
    height:auto !important;
    overflow:visible !important;
  }
  .bb-smi {
    font-size:10px;
    line-height:1.15;
    word-break:break-all;
    min-height:42px;
  }
</style>
<div class="bb-plot">
"""
    ]

    for smiles, route_count, _, mol in rows:
        mol.clean2d()
        bar_height = int(185 * route_count / max_count)
        html.append(
            f"""
  <div class="bb-col">
    <div class="bb-count">{route_count}</div>
    <div class="bb-bar-wrap">
      <div class="bb-bar" style="height:{bar_height}px;"></div>
    </div>
    <div class="bb-mol">{mol.depict()}</div>
    <div class="bb-smi">{escape(smiles)}</div>
  </div>
"""
        )

    html.append("</div>")
    return "".join(html)


def plot_top_real_bbs_with_chython_svg(
    real_bb_stats: Mapping[str, Mapping[str, Any]],
    *,
    top_n: int = 10,
    min_mol_size: int = 4,
) -> str:
    """Display top BB usage with Chython molecule SVG depictions.

    The HTML string is returned as well so callers can save or inspect it.
    """

    html = top_bbs_chython_svg_html(
        real_bb_stats,
        top_n=top_n,
        min_mol_size=min_mol_size,
    )
    try:
        from IPython.display import HTML, display
    except ImportError:
        return html
    display(HTML(html))
    return html


def plot_sb_cgr_cluster_venn(
    clusters_1: Mapping[Any, Mapping[str, Any]],
    clusters_2: Mapping[Any, Mapping[str, Any]],
    *,
    labels: tuple[str, str] = ("clusters_1", "clusters_2"),
):
    """Plot SB-CGR cluster overlap with matplotlib-venn."""

    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2

    sb_keys_1 = {
        str(cluster["sb_cgr"])
        for cluster in clusters_1.values()
        if cluster.get("sb_cgr") is not None
    }
    sb_keys_2 = {
        str(cluster["sb_cgr"])
        for cluster in clusters_2.values()
        if cluster.get("sb_cgr") is not None
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    venn2([sb_keys_1, sb_keys_2], set_labels=labels, ax=ax)
    ax.set_title(f"SB-CGR overlap: {len(sb_keys_1 & sb_keys_2)} shared")
    return fig, ax
