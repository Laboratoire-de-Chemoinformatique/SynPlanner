import pickle
import re
from collections import defaultdict
from pathlib import Path

from chython.containers import CGRContainer

from synplan.routes.io import (
    make_dict,
    make_json,
    read_routes_csv,
    read_routes_json,
)
from synplan.routes.route_cgr import compose_all_route_cgrs, compose_all_sb_cgrs
from synplan.utils.visualisation import routes_clustering_report


def run_cluster_cli(
    routes_file: str,
    cluster_results_dir: str,
    perform_subcluster: bool = False,
    subcluster_results_dir: Path = None,
):
    """
    Read routes from a CSV or JSON file, perform clustering, and optionally subclustering.

    Args:
        routes_file: Path to the input routes file (.csv or .json).
        cluster_results_dir: Directory where clustering results are stored.
        perform_subcluster: Whether to run subclustering on each cluster.
        subcluster_results_dir: Subdirectory for subclustering results (if enabled).
    """
    import click

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

    # Optional subclustering
    if perform_subcluster and subcluster_results_dir:
        from synplan.routes.clustering.subclustering import subcluster_all_clusters
        from synplan.utils.visualisation import routes_subclustering_report

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
                    aam=False,
                    html_path=str(subreport_path),
                )


def cluster_route_from_csv(routes_file: str):
    """
    Reads retrosynthetic routes from a CSV file, processes them, and performs clustering.

    This function orchestrates the process of loading retrosynthetic route data
    from a specified CSV file, converting the routes into Condensed Graph of
    Reactions (CGRs), reducing these CGRs to a simplified form (SB-CGRs),
    and finally clustering the routes based on these reduced representations.
    It uses strategic bonds for clustering by default (as indicated by `use_strat=False`
    in `cluster_routes`, which implies clustering based on the graph structure
    derived from the reduced CGRs, which often highlight strategic bonds).

    Args:
        routes_file (str): The path to the CSV file containing the retrosynthetic
                           route data.

    Returns:
        object: The result of the clustering process, typically a data structure
                representing the identified clusters. The exact type depends on
                the implementation of the `cluster_routes` function.
    """
    routes_dict = read_routes_csv(routes_file)
    route_cgrs_dict = compose_all_route_cgrs(routes_dict)
    sb_cgrs_dict = compose_all_sb_cgrs(route_cgrs_dict)
    clusters = cluster_routes(sb_cgrs_dict, use_strat=False)
    return clusters


def cluster_route_from_json(routes_file: str):
    """
    Reads retrosynthetic routes from a JSON file, processes them, and performs clustering.

    This function is similar to `cluster_route_from_csv` but loads the
    retrosynthetic route data from a specified JSON file. It reads the JSON,
    converts it into a suitable dictionary format, composes and reduces the
    Condensed Graph of Reactions (CGRs) for each route, and then clusters
    the routes based on these reduced representations, typically using
    strategic bonds as the basis for clustering.

    Args:
        routes_file (str): The path to the JSON file containing the retrosynthetic
                           route data.

    Returns:
        object: The result of the clustering process, typically a data structure
                representing the identified clusters. The exact type depends on
                the implementation of the `cluster_routes` function.
    """
    routes_json = read_routes_json(routes_file)
    routes_dict = make_dict(routes_json)
    route_cgrs_dict = compose_all_route_cgrs(routes_dict)
    sb_cgrs_dict = compose_all_sb_cgrs(route_cgrs_dict)
    clusters = cluster_routes(sb_cgrs_dict, use_strat=False)
    return clusters


def extract_strat_bonds(target_cgr: CGRContainer):
    """
    Extracts strategic bonds from a CGRContainer object.

    Strategic bonds are identified as bonds where the original bond order
    (`bond.order`) is None (indicating a bond that was not present in the
    reactants) but the primary bond order (`bond.p_order`) is not None
    (indicating a bond that was formed in the product). This function iterates
    through all bonds in the input CGR, identifies those matching the criteria
    for strategic bonds, and returns a sorted list of unique strategic bonds
    represented as tuples of sorted atom indices.

    Args:
        target_cgr (CGRContainer): The CGRContainer object from which to extract
                                   strategic bonds.

    Returns:
        list: A sorted list of tuples, where each tuple represents a strategic
              bond by the sorted integer indices of the two atoms involved in the bond.
    """
    result = []
    seen = set()
    for atom1, bond_set in target_cgr._bonds.items():
        for atom2, bond in bond_set.items():
            if atom1 >= atom2:
                continue
            if bond.order is None and bond.p_order is not None:
                bond_key = tuple(sorted((atom1, atom2)))
                if bond_key not in seen:
                    seen.add(bond_key)
                    result.append(bond_key)
    return sorted(result)


def cluster_routes(sb_cgrs: dict, use_strat=False):
    """
    Cluster routes objects based on their strategic bonds
      or CGRContainer object signature (not avoid mapping)

    Args:
        sb_cgrs: Dictionary mapping route_id to sb_cgr objects.

    Returns:
        Dictionary with groups keyed by '{length}.{index}' containing
        'sb_cgr', 'route_ids', and 'strat_bonds'.
    """
    temp_groups = defaultdict(
        lambda: {"route_ids": [], "sb_cgr": None, "strat_bonds": None}
    )

    # 1. Initial grouping based on the content of strategic bonds
    for route_id, sb_cgr in sb_cgrs.items():
        strat_bonds_list = extract_strat_bonds(sb_cgr)
        if use_strat == True:
            group_key = tuple(strat_bonds_list)
        else:
            group_key = str(sb_cgr)

        if not temp_groups[group_key]["route_ids"]:  # First time seeing this group
            temp_groups[group_key][
                "sb_cgr"
            ] = sb_cgr  # Store the first CGR as representative
            temp_groups[group_key][
                "strat_bonds"
            ] = strat_bonds_list  # Store the actual list

        temp_groups[group_key]["route_ids"].append(route_id)
        temp_groups[group_key][
            "route_ids"
        ].sort()  # Keep route_ids sorted for consistency

    for group_key in temp_groups.keys():
        temp_groups[group_key]["group_size"] = len(temp_groups[group_key]["route_ids"])

    # 2. Format the output dictionary with desired keys '{length}.{index}'
    final_grouped_results = {}
    group_indices = defaultdict(int)  # To track index for each length

    # Sort items by length of bonds first, then potentially by bonds themselves for consistent indexing
    # Sorting by the group_key (tuple of tuples) provides a deterministic order
    sorted_groups = sorted(
        temp_groups.items(), key=lambda item: (len(item[0]), item[0])
    )

    for group_key, group_data in sorted_groups:
        num_bonds = len(group_data["strat_bonds"])
        group_indices[num_bonds] += 1  # Increment index for this length (1-based)
        final_key = f"{num_bonds}.{group_indices[num_bonds]}"
        final_grouped_results[final_key] = group_data

    return final_grouped_results
