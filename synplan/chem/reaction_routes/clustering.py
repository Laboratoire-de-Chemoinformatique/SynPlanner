from collections import defaultdict
from pathlib import Path
import pickle
import re
from typing import Any, Dict

from CGRtools.containers import CGRContainer, ReactionContainer
from CGRtools.containers.bonds import DynamicBond

from synplan.chem.reaction_routes.io import (
    make_dict,
    make_json,
    read_routes_csv,
    read_routes_json,
)
from synplan.chem.reaction_routes.leaving_groups import *
from synplan.chem.reaction_routes.route_cgr import *
from synplan.chem.reaction_routes.visualisation import *
from synplan.utils.visualisation import (
    routes_clustering_report,
    routes_subclustering_report,
)


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


def lg_process_reset(lg_cgr: CGRContainer, atom_num: int):
    """
    Normalize bonds in an extracted leaving group (X) fragment and flag the attachment atom as a radical.

    Scans all bonds in `lg_cgr`, converting any bond with undefined `p_order`
    but defined `order` into a `DynamicBond` of matching integer order. Then sets
    the atom at `atom_num` to a radical.

    Parameters
    ----------
    target_cgr : CGRContainer
        The CGR representing the isolated leaving-group fragment.
    atom_num : int
        Index of the attachment atom to mark as a radical.

    Returns
    -------
    CGRContainer
        The modified `lg_cgr` with normalized bonds and the specified atom
        flagged as a radical.
    """
    bond_items = list(lg_cgr._bonds.items())
    for atom1, bond_set in bond_items:
        bond_set_items = list(bond_set.items())
        for atom2, bond in bond_set_items:
            if bond.p_order is None and bond.order is not None:
                order = int(bond.order)
                lg_cgr.delete_bond(atom1, atom2)
                lg_cgr.add_bond(atom1, atom2, DynamicBond(order, order))
    lg_cgr._atoms[atom_num].is_radical = True
    return lg_cgr


def lg_replacer(route_cgr: CGRContainer):
    """
    Extract dynamic leaving-groups from a CGR and mark attachment points.

    Scans the input CGRContainer for bonds lacking explicit p_order (i.e., leaving-group attachments),
    severs those bonds, captures each leaving-group as its own CGRContainer, and inserts DynamicX
    markers at the attachment sites. Finally, reindexes the markers to ensure unique labels.

    Parameters
    ----------
    route_cgr : CGRContainer
        A CGR representing the full synthethic route.

    Returns
    -------
    synthon_cgr : CGRContainer
        The core synthon CGR with DynamicX atoms marking each former leaving-group site.
    lg_groups : dict[int, tuple[CGRContainer, int]]
        Mapping from each marker label to a tuple of:
        - the extracted leaving-group CGRContainer
        - the atom index where it was attached.
    """
    lg_groups = {}

    cgr_prods = [route_cgr.substructure(c) for c in route_cgr.connected_components]
    target_cgr = cgr_prods[0]

    bond_items = list(target_cgr._bonds.items())
    reaction = ReactionContainer.from_cgr(target_cgr)
    target_mol = reaction.products[0]
    max_in_target_mol = max(target_mol._atoms)

    k = 1
    atom_nums = []
    checked_atoms = set()

    for atom1, bond_set in bond_items:
        bond_set_items = list(bond_set.items())
        for atom2, bond in bond_set_items:
            if (
                bond.p_order is None
                and bond.order is not None
                and tuple(sorted([atom1, atom2])) not in checked_atoms
            ):
                if atom1 <= max_in_target_mol:
                    lg = DynamicX()
                    lg.mark = k
                    lg.isotope = k
                    order = bond.order
                    p_order = bond.p_order
                    target_cgr.delete_bond(atom1, atom2)
                    lg_cgrs = [
                        target_cgr.substructure(c)
                        for c in target_cgr.connected_components
                    ]
                    checked_atoms.add(tuple(sorted([atom1, atom2])))
                    if len(lg_cgrs) == 2:
                        lg_cgr = lg_cgrs[1]
                        lg_cgr = lg_process_reset(lg_cgr, atom2)
                        lg_cgr.clean2d()
                    else:
                        continue
                    lg_groups[k] = (lg_cgr, atom2)
                    target_cgr = [
                        target_cgr.substructure(c)
                        for c in target_cgr.connected_components
                    ][0]
                    target_cgr.add_atom(lg, atom2)
                    if order == 4 and p_order == None:
                        order = 1
                    target_cgr.add_bond(atom1, atom2, DynamicBond(order, p_order))
                    target_cgr = [
                        target_cgr.substructure(c)
                        for c in target_cgr.connected_components
                    ][0]
                    k += 1
                    atom_nums.append(atom2)

    synthon_cgr = [target_cgr.substructure(c) for c in target_cgr.connected_components][
        0
    ]
    reaction = ReactionContainer.from_cgr(synthon_cgr)
    reactants = reaction.reactants

    atom_mark_map = {}  # To map atom numbers to their new marks
    g = 1
    for n, r in enumerate(reactants):
        for atom_num in atom_nums:
            if atom_num in r._atoms:
                synthon_cgr._atoms[atom_num].mark = g
                atom_mark_map[atom_num] = g
                g += 1

    new_lg_groups = {}
    for original_mark in lg_groups:
        cgr_obj, a_num = lg_groups[original_mark]
        new_mark = atom_mark_map.get(a_num)
        if new_mark is not None:
            new_lg_groups[new_mark] = (cgr_obj, a_num)
    lg_groups = new_lg_groups

    return synthon_cgr, lg_groups


def lg_reaction_replacer(
    synthon_reaction: ReactionContainer, lg_groups: dict, max_in_target_mol: int
):
    """
    Replace marked leaving-groups (X) into synthon reactants.

    For each reactant in `synthon_reaction`, finds placeholder atoms
    (indices > `max_in_target_mol`) that match entries in `lg_groups`,
    replaces them with `MarkedAt` atoms labeled by their leaving-group key (X),
    and preserves original bond connectivity.

    Parameters
    ----------
    synthon_reaction : ReactionContainer
        Reaction containing reactants with X placeholders.
    lg_groups : dict[int, tuple[CGRContainer, int]]
        Mapping from X label to (X CGR, attachment atom index).
    max_in_target_mol : int
        Highest atom index of the core product; any atom_num above this is a placeholder.

    Returns
    -------
    List[Molecule]
        Reactant molecules with `MarkedAt` atoms reinserted at X attachment sites.
    """
    new_reactants = []
    for reactant in synthon_reaction.reactants:
        atom_keys = list(reactant._atoms.keys())
        for atom_num in atom_keys:
            if atom_num > max_in_target_mol:
                for k, val in lg_groups.items():
                    lg = MarkedAt()
                    if atom_num == val[1]:
                        lg.mark = k
                        lg.isotope = k
                        atom1 = list(reactant._bonds[atom_num].keys())[0]
                        bond = reactant._bonds[atom_num][atom1]
                        reactant.delete_bond(atom1, atom_num)
                        reactant.delete_atom(atom_num)
                        reactant.add_atom(lg, atom_num)
                        reactant.add_bond(atom1, atom_num, bond)
        new_reactants.append(reactant)
    return new_reactants


class SubclusterError(Exception):
    """Raised when subcluster_one_cluster cannot complete successfully."""


def subcluster_one_cluster(group, sb_cgrs_dict, route_cgrs_dict):
    """
    Generate synthon data for each route in a single cluster.

    For each route (route ID) in `group['route_ids']`, replaces RouteCGRs with
    SynthonCGR, builds ReactionContainers before and after X replacement,
    and collects relevant data.

    Parameters
    ----------
    group : dict
        Must include `'route_ids'`, a list of route identifiers.
    sb_cgrs_dict : dict
        Maps route IDs to their SB-CGR.
    route_cgrs_dict : dict
        Maps route IDs to their RouteCGR.

    Returns
    -------
    dict or None
        If successful, returns a dict mapping each `route_id` to a tuple:
        `(sb_cgr, original_reaction, synthon_cgr, new_reaction, lg_groups)`.
        Or raises SubclusterError on any failure: if any step (X replacement or reaction
        parsing) fails for a route.

    """

    route_ids = group.get("route_ids")
    if not isinstance(route_ids, (list, tuple)):
        raise SubclusterError(
            f"'route_ids' must be a list or tuple, got {type(route_ids).__name__}"
        )

    result = {}
    for route_id in route_ids:
        sb_cgr = sb_cgrs_dict[route_id]
        route_cgr = route_cgrs_dict[route_id]

        # 1) Replace leaving groups in RouteCGR
        try:
            synthon_cgr, lg_groups = lg_replacer(route_cgr)
            lg_sizes = len(lg_groups)
        except (KeyError, ValueError) as e:
            raise SubclusterError(f"LG replacement failed for route {route_id}") from e

        # 2) Build ReactionContainer for Abstracted RouteCGR
        try:
            synthon_rxn = ReactionContainer.from_cgr(synthon_cgr)
        except:  # replace with the actual exception class
            raise SubclusterError(
                f"Failed to parse synthon CGR for route {route_id}"
            ) from e

        # 3) Prepare for X-based reaction replacement
        try:
            old_reactants = synthon_rxn.reactants
            target_mol = synthon_rxn.products[0]
            max_atom_idx = max(target_mol._atoms)
            new_reactants = lg_reaction_replacer(synthon_rxn, lg_groups, max_atom_idx)
            new_rxn = ReactionContainer(reactants=new_reactants, products=[target_mol])
        except (IndexError, TypeError) as e:
            raise SubclusterError(
                f"Leaving group (X) reaction replacement failed for route {route_id}"
            ) from e

        result[route_id] = (
            sb_cgr,
            ReactionContainer(reactants=old_reactants, products=[target_mol]),
            synthon_cgr,
            new_rxn,
            lg_groups,
            lg_sizes,
        )

    return result


def group_routes_by_synthon_detail(data_dict: Dict[Any, list]) -> Dict[str, dict]:
    """
    Groups routes based on synthon CGR (result_list[0]), reaction data, and lg_sizes.
    The final group index is formatted as "{lg_sizes}_{temp_index}".

    Args:
        data_dict: Dictionary {route_id: [sb_cgr, unlabeled_reaction, synthon_cgr, synthon_reaction,
                                         route_specific_data, lg_sizes, ...]}.

    Returns:
        Dictionary {
            group_index (str): {
                'sb_cgr': ...,
                'unlabeled_reaction': ...,
                'synthon_cgr': ...,
                'synthon_reaction': ...,
                'routes_data': {route_id: route_specific_data, ...},
                'lg_sizes': ...,
                'post_processed': False
            }
        }
    """
    # 1. Bucket route_ids by their grouping key
    temp_groups = defaultdict(list)
    for route_id, result_list in data_dict.items():
        # unpack values with defaults
        sb_cgr = result_list[0] if len(result_list) > 0 else None
        unlabeled_reaction = result_list[1] if len(result_list) > 1 else None
        synthon_cgr = result_list[2] if len(result_list) > 2 else None
        synthon_reaction = result_list[3] if len(result_list) > 3 else None
        lg_sizes = result_list[5] if len(result_list) > 5 else None

        # Attempt to use all parts of the key; skip if unhashable
        try:
            group_key = (
                sb_cgr,
                unlabeled_reaction,
                synthon_cgr,
                synthon_reaction,
                lg_sizes,
            )
        except TypeError:
            print(f"Warning: Skipping route {route_id} due to unhashable key element.")
            continue

        temp_groups[group_key].append(route_id)

    # 2. Sort groups for consistent ordering
    sorted_groups = sorted(temp_groups.items(), key=lambda kv: kv[1])

    # 3. Build final dict, numbering per lg_sizes
    final_groups = {}
    counters = defaultdict(int)  # counters per lg_sizes

    for group_key, route_ids in sorted_groups:
        sb_cgr, unlabeled_reaction, synthon_cgr, synthon_reaction, lg_sizes = group_key

        # Increment the counter for this lg_sizes
        counters[lg_sizes] += 1
        temp_index = counters[lg_sizes]
        group_index = f"{lg_sizes}_{temp_index}"

        # Collect the route-specific data (at index 4) for each route
        routes_data = {}
        for rid in sorted(route_ids):
            orig = data_dict.get(rid, [])
            routes_data[rid] = orig[4] if len(orig) > 4 else None

        final_groups[group_index] = {
            "sb_cgr": sb_cgr,
            "unlabeled_reaction": unlabeled_reaction,
            "synthon_cgr": synthon_cgr,
            "synthon_reaction": synthon_reaction,
            "routes_data": routes_data,
            "lg_sizes": lg_sizes,
            "post_processed": False,
        }

    return final_groups


def subcluster_all_clusters(groups, sb_cgrs_dict, route_cgrs_dict):
    """
    Subdivide each reaction cluster into detailed synthon-based subgroups.

    Iterates over all clusters in `groups`, applies `subcluster_one_cluster`
    to generate per-cluster synthons, then organizes routes by synthon detail.

    Parameters
    ----------
    groups : dict
        Mapping of cluster indices to cluster data.
    sb_cgrs_dict : dict
        Dictionary of SB-CGRs
    route_cgrs_dict : dict
        Dictionary of RoteCGRs

    Returns
    -------
    dict or None
        A dict mapping each cluster index to its subgroups dict,
        or None if any cluster fails to subcluster.
    """
    all_subgroups = {}
    for group_index, group in groups.items():
        group_synthons = subcluster_one_cluster(group, sb_cgrs_dict, route_cgrs_dict)
        if group_synthons is None:
            return None
        all_subgroups[group_index] = group_routes_by_synthon_detail(group_synthons)
    return all_subgroups


def all_lg_collect(subgroup):
    """
    Gather all leaving-group CGRContainers by route index.

    Scans `subgroup['routes_data']`, collects every CGRContainer per index,
    and returns a mapping from each index to the list of distinct containers.

    Parameters
    ----------
    subgroup : dict
        Must contain 'routes_data', a dict mapping pathway keys to
        dicts of {route_index: (CGRContainer, …)}.

    Returns
    -------
    dict[int, list[CGRContainer]]
        For each route index, a list of unique CGRContainer objects
        (duplicates by string are filtered out).
    """
    all_indices = set()
    for sub_dict in subgroup["routes_data"].values():
        all_indices.update(sub_dict.keys())

    # Dynamically initialize result and seen dictionaries
    result = {idx: [] for idx in all_indices}
    seen = {idx: set() for idx in all_indices}

    # Populate the result with unique CGRContainer objects
    for sub_dict in subgroup["routes_data"].values():
        for idx in sub_dict:
            cgr_container = sub_dict[idx][0]
            cgr_str = str(cgr_container)
            if cgr_str not in seen[idx]:
                seen[idx].add(cgr_str)
                result[idx].append(cgr_container)
    return result


def replace_leaving_groups_in_synthon(subgroup, to_remove):  # Under development
    """
    Replace specified leaving groups (LG) in a synthon CGR with new fragments and return the updated CGR
    along with a mapping from adjusted LG marks to their atom indices.

    Parameters:
        subgroup (dict): Must contain:
            - 'synthon_cgr': the CGR object representing the synthon graph
            - 'routes_data': mapping of route indices to LG replacement data
        to_remove (List[int]): List of LG marks to remove and replace.

    Returns:
        Tuple[CGR, Dict[int, int]]:
            - The updated CGR with replacements
            - A dict mapping new LG marks to their atom indices in the updated CGR
    """
    # Extract the original CGR and leaving group replacement table
    original_cgr = subgroup["synthon_cgr"]
    lg_table = next(iter(subgroup["routes_data"].values()))

    updated_cgr = original_cgr

    removed_count = 0
    new_lgs = {}

    # Iterate through all atoms (index, atom_obj) in the CGR
    for atom_idx, atom_obj in list(updated_cgr.atoms()):
        # Skip non-X atoms
        if atom_obj.__class__.__name__ != "DynamicX":
            continue

        current_mark = atom_obj.mark
        if current_mark in to_remove:
            # Remove old LG (X): delete bond and atom
            neighbors = list(updated_cgr._bonds[atom_idx].keys())
            if neighbors:
                neighbor_idx = neighbors[0]
                bond = updated_cgr._bonds[atom_idx][neighbor_idx]
                updated_cgr.delete_bond(atom_idx, neighbor_idx)
                updated_cgr.delete_atom(atom_idx)

                # Attach new LG(X) fragment from the table
                lg_fragment = lg_table[current_mark][0]
                updated_cgr = updated_cgr.union(lg_fragment)
                # Reset radical flag on the new atom and restore the bond
                updated_cgr._atoms[atom_idx].is_radical = False
                updated_cgr.add_bond(atom_idx, neighbor_idx, bond)

            removed_count += 1
        else:
            # Adjust the marks of remaining LGs to account for removed ones
            atom_obj.mark -= removed_count
            new_lgs[atom_obj.mark] = atom_idx

    # Reorder atoms dict and update 2D coordinates for depiction
    updated_cgr._atoms = dict(sorted(updated_cgr._atoms.items()))

    return updated_cgr, new_lgs


def new_lg_reaction_replacer(synthon_reaction, new_lgs, max_in_target_mol):
    """
    Replace placeholder atom indices with marked leaving-group atoms in reactants.

    Iterates through each reactant in a `ReactionContainer`, finds atom indices
    corresponding to newly detached leaving-groups (those greater than the
    core’s maximum index), and replaces them with `MarkedAt` atoms bearing
    the correct X labels and isotopes. Bonds to the original attachment points
    are preserved.

    Parameters
    ----------
    synthon_reaction : ReactionContainer
        A reaction container whose `reactants` list contains molecules with
        dummy atoms (by index) marking where leaving-groups were removed.
    new_lgs : dict[int, int]
        Mapping from leaving-group label (int) to the atom index (int) in each
        reactant that should be replaced.
    max_in_target_mol : int
        The highest atom index used by the core product. Any atom index in a
        reactant greater than this is treated as a leaving-group placeholder.

    Returns
    -------
    List[Molecule]
        A list of reactant molecules where each placeholder atom has been
        replaced by a `MarkedAt` atom with its `.mark` and `.isotope` set
        to the leaving-group label, and original bonds reattached.
    """
    new_reactants = []
    for reactant in synthon_reaction.reactants:
        atom_keys = list(reactant._atoms.keys())
        for atom_num in atom_keys:
            if atom_num > max_in_target_mol:
                for k, val in new_lgs.items():
                    lg = MarkedAt()
                    if atom_num == val:
                        lg.mark = k
                        lg.isotope = k
                        atom1 = list(reactant._bonds[atom_num].keys())[0]
                        bond = reactant._bonds[atom_num][atom1]
                        reactant.delete_bond(atom1, atom_num)
                        reactant.delete_atom(atom_num)
                        reactant.add_atom(lg, atom_num)
                        reactant.add_bond(atom1, atom_num, bond)
        new_reactants.append(reactant)

    return new_reactants


def post_process_subgroup(
    subgroup,
):  # Under development: Error in replace_leaving_groups_in_synthon , 'cuz synthon_reaction.clean2d crashes
    """
    Drop leaving-groups common to all pathways and rebuild a minimal synthon.

    Scans the subgroup for leaving-groups present in every route, removes those
    from the CGR, re-assembles a clean ReactionContainer with the original core,
    updates `routes_data`, and flags the dict as processed.

    Parameters
    ----------
    subgroup : dict
        Must include keys for `routes_data` and the helpers
        (`all_lg_collect`, `find_const_lg`, etc.). If already
        post_processed, returns immediately.

    Returns
    -------
    dict
        The same dict, now with:
        - `'synthon_reaction'`: cleaned ReactionContainer
        - `'routes_data'`: filtered route table
        - `'post_processed'`: True
    """
    if "post_processed" in subgroup.keys() and subgroup["post_processed"] == True:
        return subgroup
    result = all_lg_collect(subgroup)
    # to find constant lg that need to be removed
    to_remove = [ind for ind, cgr_set in result.items() if len(cgr_set) == 1]
    new_synthon_cgr, new_lgs = replace_leaving_groups_in_synthon(subgroup, to_remove)
    synthon_reaction = ReactionContainer.from_cgr(new_synthon_cgr)
    synthon_reaction.clean2d()
    old_reactants = ReactionContainer.from_cgr(new_synthon_cgr).reactants
    target_mol = synthon_reaction.products[0]  # TO DO: target_mol might be non 0
    max_in_target_mol = max(target_mol._atoms)
    new_reactants = new_lg_reaction_replacer(
        synthon_reaction, new_lgs, max_in_target_mol
    )
    new_synthon_reaction = ReactionContainer(
        reactants=new_reactants, products=[target_mol]
    )
    new_synthon_reaction.clean2d()
    subgroup["synthon_reaction"] = new_synthon_reaction
    subgroup["routes_data"] = remove_and_shift(subgroup["routes_data"], to_remove)
    subgroup["post_processed"] = True
    subgroup["group_lgs"] = group_by_identical_values(subgroup["routes_data"])
    return subgroup


def group_by_identical_values(routes_data):  # Under development
    """
    Groups entries in a nested dictionary based on identical sets of core values.

    Identifies route IDs whose inner dictionaries contain the
    same sequence of leaving groups, when ordered by subkey. These are collapsed into a single entry.

    Args:
        routes_data (dict): A dictionary mapping outer keys to inner dictionaries.
            Each inner dictionary maps subkeys to a tuple `(value_obj, other_info)`.
            `value_obj` is used for grouping, `other_info` is ignored.
            Example: {'route_1': {'pos_a': (1, 'infoA'), 'pos_b': (2, 'infoB')}, ...}

    Returns:
        dict: A dictionary where:
            - Keys are tuples of the original outer keys that were grouped.
            - Values are dictionaries mapping the original subkeys to the
              `value_obj` from the first outer key in the group's tuple.
            The dictionary is sorted descending by the number of grouped outer keys.
            Example: {('route_1', 'route_2'): {'pos_a': 1, 'pos_b': 2}, ...}
    """
    # Step 1: Build a signature for each outer key: the tuple of all first-elements in its inner dict
    signature_map = defaultdict(list)
    for outer_key, inner_dict in routes_data.items():
        # Sort inner_dict items by subkey to ensure consistent ordering
        sorted_items = sorted(inner_dict.items(), key=lambda kv: kv[0])
        # Extract only the first element of each (value_obj, other_info) tuple
        signature = tuple(val_tuple[0] for _, val_tuple in sorted_items)
        signature_map[signature].append(outer_key)

    # Step 2: Build the grouped result
    grouped = {}
    for signature, outer_keys in signature_map.items():
        # Use the representative inner dict from the first outer key in this group
        rep_inner = routes_data[outer_keys[0]]
        # Build mapping subkey -> value_obj
        rep_values = {subkey: val_tuple[0] for subkey, val_tuple in rep_inner.items()}
        # Store under tuple of grouped outer keys
        grouped_key = tuple(outer_keys)
        grouped[grouped_key] = rep_values

    sorted_grouped = dict(
        sorted(grouped.items(), key=lambda item: len(item[0]), reverse=True)
    )

    return sorted_grouped
