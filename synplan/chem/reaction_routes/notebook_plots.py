"""Compatibility wrapper for :mod:`synplan.routes.notebook_plots`."""

from synplan.routes.notebook_plots import (
    plot_sb_cgr_cluster_venn,
    plot_top_real_bbs,
    plot_top_real_bbs_with_chython_svg,
    top_bb_usage_rows,
    top_bbs_chython_svg_html,
)

__all__ = [
    "plot_sb_cgr_cluster_venn",
    "plot_top_real_bbs",
    "plot_top_real_bbs_with_chython_svg",
    "top_bb_usage_rows",
    "top_bbs_chython_svg_html",
]
