"""Analysis and visualisation of a built tree and the routes read out of it."""

from __future__ import annotations

from synplan.utils.visualisation.clustering import (
    group_lg_table_2_html_fixed,
    group_supporting_table_2_html_fixed,
    html_top_routes_cluster,
    lg_table_2_html,
    routes_clustering_report,
    routes_subclustering_report,
    supporting_table_2_html,
)
from synplan.utils.visualisation.routes import (
    extract_routes,
    get_child_nodes,
    route_rule_labels,
    routes_report_html,
)

__all__ = [
    "extract_routes",
    "get_child_nodes",
    "group_lg_table_2_html_fixed",
    "group_supporting_table_2_html_fixed",
    "html_top_routes_cluster",
    "lg_table_2_html",
    "route_rule_labels",
    "routes_clustering_report",
    "routes_report_html",
    "routes_subclustering_report",
    "supporting_table_2_html",
]
