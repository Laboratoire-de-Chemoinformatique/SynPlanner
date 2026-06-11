"""Command-line entry point for route clustering."""

import pickle
import re
from pathlib import Path

import click

from synplan.chem.reaction.routes.clustering import (
    cluster_routes,
    subcluster_all_clusters,
)
from synplan.chem.reaction.routes.io import (
    make_dict,
    make_json,
    read_routes_csv,
    read_routes_json,
)
from synplan.chem.reaction.routes.representation import (
    compose_all_route_cgrs,
    compose_all_sb_cgrs,
)
from synplan.utils.visualisation import (
    routes_clustering_report,
    routes_subclustering_report,
)


def run_cluster_cli(
    routes_file: str,
    cluster_results_dir: str,
    perform_subcluster: bool = False,
    subcluster_results_dir: Path | None = None,
):
    """
    Read routes from a CSV or JSON file, perform clustering, and optionally subclustering.

    Args:
        routes_file: Path to the input routes file (.csv or .json).
        cluster_results_dir: Directory where clustering results are stored.
        perform_subcluster: Whether to run subclustering on each cluster.
        subcluster_results_dir: Subdirectory for subclustering results (if enabled).
    """

    routes_file = Path(routes_file)
    match = re.search(r"_(\d+)\.", routes_file.name)
    if not match:
        raise ValueError(f"Could not extract index from filename: {routes_file.name}")
    file_index = int(match.group(1))
    ext = routes_file.suffix.lower()
    if ext == ".csv":
        routes_dict = read_routes_csv(str(routes_file))
        routes_json = make_json(routes_dict)
    elif ext == ".json":
        routes_json = read_routes_json(str(routes_file))
        routes_dict = make_dict(routes_json)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Compose condensed graph representations
    route_cgrs = compose_all_route_cgrs(routes_dict)
    click.echo("Generating RouteCGR")
    sb_cgrs = compose_all_sb_cgrs(route_cgrs)
    click.echo("Generating SB-CGR")

    # Perform clustering
    click.echo("\nClustering")
    clusters = cluster_routes(sb_cgrs, use_strat=False)

    click.echo(f"Total number of routes: {len(routes_dict)}")
    click.echo(f"Found number of clusters: {len(clusters)} ({list(clusters.keys())})")

    # Ensure output directory exists
    cluster_results_dir = Path(cluster_results_dir)
    cluster_results_dir.mkdir(parents=True, exist_ok=True)

    # Save clusters to pickle
    with open(cluster_results_dir / f"clusters_{file_index}.pickle", "wb") as f:
        pickle.dump(clusters, f)

    # Generate HTML reports for each cluster
    for idx in clusters:
        report_path = cluster_results_dir / f"{file_index}_cluster_{idx}.html"
        routes_clustering_report(
            routes_json, clusters, idx, sb_cgrs, html_path=str(report_path)
        )

    # Optional subclustering (Under development)
    if perform_subcluster and subcluster_results_dir:
        click.echo("\nSubClustering")
        sub_dir = cluster_results_dir / subcluster_results_dir
        sub_dir.mkdir(parents=True, exist_ok=True)

        subclusters = subcluster_all_clusters(clusters, sb_cgrs, route_cgrs)
        for cluster_idx, sub in subclusters.items():
            click.echo(f"Cluster {cluster_idx} has {len(sub)} subclusters")
            for sub_idx, subcluster in sub.items():
                subreport_path = (
                    sub_dir / f"{file_index}_subcluster_{cluster_idx}.{sub_idx}.html"
                )
                routes_subclustering_report(
                    routes_json,
                    subcluster,
                    cluster_idx,
                    sub_idx,
                    sb_cgrs,
                    aam=False,
                    html_path=str(subreport_path),
                )
