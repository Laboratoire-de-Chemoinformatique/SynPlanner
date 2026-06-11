"""Cluster and subcluster retrosynthetic routes by their strategic-bond CGRs."""

from collections import defaultdict
from typing import Any

from chython.containers import CGRContainer, ReactionContainer

from synplan.chem.reaction.routes.leaving_groups import (
    all_lg_collect,
    lg_reaction_replacer,
    lg_replacer,
    new_lg_reaction_replacer,
)
from synplan.chem.reaction.routes.visualisation import remove_and_shift


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
        if use_strat:
            group_key = tuple(strat_bonds_list)
        else:
            group_key = str(sb_cgr)

        if not temp_groups[group_key]["route_ids"]:  # First time seeing this group
            temp_groups[group_key]["sb_cgr"] = (
                sb_cgr  # Store the first CGR as representative
            )
            temp_groups[group_key]["strat_bonds"] = (
                strat_bonds_list  # Store the actual list
            )

        temp_groups[group_key]["route_ids"].append(route_id)
        temp_groups[group_key][
            "route_ids"
        ].sort()  # Keep route_ids sorted for consistency

    for group_key in temp_groups:
        temp_groups[group_key]["group_size"] = len(temp_groups[group_key]["route_ids"])

    # 2. Format the output dictionary with desired keys '{length}.{index}'
    final_grouped_results = {}
    group_indices = defaultdict(int)  # To track index for each length

    # Sort items by length of bonds first, then potentially by bonds themselves for consistent indexing
    # Sorting by the group_key (tuple of tuples) provides a deterministic order
    sorted_groups = sorted(
        temp_groups.items(), key=lambda item: (len(item[0]), item[0])
    )

    for _group_key, group_data in sorted_groups:
        num_bonds = len(group_data["strat_bonds"])
        group_indices[num_bonds] += 1  # Increment index for this length (1-based)
        final_key = f"{num_bonds}.{group_indices[num_bonds]}"
        final_grouped_results[final_key] = group_data

    return final_grouped_results


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
        except Exception as e:
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


def group_routes_by_synthon_detail(data_dict: dict[Any, list]) -> dict[str, dict]:
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
                # is_radical is read-only in chython; modify container dicts directly.
                updated_cgr._radicals[atom_idx] = False
                updated_cgr._p_radicals[atom_idx] = False
                updated_cgr.flush_cache()
                updated_cgr.add_bond(atom_idx, neighbor_idx, bond)

            removed_count += 1
        else:
            # Adjust the marks of remaining LGs to account for removed ones
            atom_obj.mark -= removed_count
            new_lgs[atom_obj.mark] = atom_idx

    # Reorder atoms dict and update 2D coordinates for depiction
    updated_cgr._atoms = dict(sorted(updated_cgr._atoms.items()))

    return updated_cgr, new_lgs


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
    if subgroup.get("post_processed"):
        return subgroup
    result = all_lg_collect(subgroup)
    # to find constant lg that need to be removed
    to_remove = [ind for ind, cgr_set in result.items() if len(cgr_set) == 1]
    new_synthon_cgr, new_lgs = replace_leaving_groups_in_synthon(subgroup, to_remove)
    synthon_reaction = ReactionContainer.from_cgr(new_synthon_cgr)
    synthon_reaction.clean2d()
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
    for _signature, outer_keys in signature_map.items():
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
