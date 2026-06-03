"""Notebook plotting helpers for RouteCGR analysis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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
