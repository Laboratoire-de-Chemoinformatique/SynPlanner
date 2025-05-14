from collections import defaultdict

from CGRtools.containers import ReactionContainer, CGRContainer
from CGRtools.containers.bonds import DynamicBond

from synplan.chem.reaction_routes.leaving_groups import *
from synplan.chem.reaction_routes.visualisation import *
from synplan.chem.reaction_routes.route_cgr import *
from synplan.chem.reaction_routes.io import read_routes_csv, read_routes_json, make_dict


def cluster_route_from_csv(routes_file):
    routes_dict = read_routes_csv(routes_file)
    route_cgrs_dict = compose_all_route_cgrs(routes_dict)
    reduced_route_cgrs_dict = compose_all_reduced_route_cgrs(route_cgrs_dict)
    clusters = cluster_routes(reduced_route_cgrs_dict, use_strat=False)
    return clusters


def cluster_route_from_json(routes_file):
    routes_json = read_routes_json(routes_file)
    routes_dict = make_dict(routes_json)
    route_cgrs_dict = compose_all_route_cgrs(routes_dict)
    reduced_route_cgrs_dict = compose_all_reduced_route_cgrs(route_cgrs_dict)
    clusters = cluster_routes(reduced_route_cgrs_dict, use_strat=False)
    return clusters


def extract_strat_bonds(target_cgr):
    """Extracts strategic bonds (order=None, p_order!=None)."""
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

def cluster_routes(clusters_dict: dict, use_strat = True):
    """
    Cluster routes objects based on their strategic bonds
      or CGRContainer object signature (not avoid mapping)

    Args:
        clusters_dict: Dictionary mapping node_id to rg_cgr objects.

    Returns:
        Dictionary with groups keyed by '{length}.{index}' containing
        'rg_cgr', 'node_ids', and 'strat_bonds'.
    """
    temp_groups = defaultdict(lambda: {'node_ids': [], 'rg_cgr': None, 'strat_bonds': None})

    # 1. Initial grouping based on the content of strategic bonds
    for node_id, rg_cgr in clusters_dict.items():
        strat_bonds_list = extract_strat_bonds(rg_cgr)
        if use_strat == True:
            group_key = tuple(strat_bonds_list)
        else:
            group_key = str(rg_cgr)

        if not temp_groups[group_key]['node_ids']: # First time seeing this group
            temp_groups[group_key]['rg_cgr'] = rg_cgr # Store the first CGR as representative
            temp_groups[group_key]['strat_bonds'] = strat_bonds_list # Store the actual list

        temp_groups[group_key]['node_ids'].append(node_id)
        temp_groups[group_key]['node_ids'].sort() # Keep node_ids sorted for consistency
        # temp_groups[group_key]['group_size'] = len(temp_groups[group_key]['node_ids'])
    for group_key in temp_groups.keys():
        temp_groups[group_key]['group_size'] = len(temp_groups[group_key]['node_ids'])

    # 2. Format the output dictionary with desired keys '{length}.{index}'
    final_grouped_results = {}
    group_indices = defaultdict(int) # To track index for each length

    # Sort items by length of bonds first, then potentially by bonds themselves for consistent indexing
    # Sorting by the group_key (tuple of tuples) provides a deterministic order
    sorted_groups = sorted(temp_groups.items(), key=lambda item: (len(item[0]), item[0]))

    for group_key, group_data in sorted_groups:
        num_bonds = len(group_data['strat_bonds'])
        group_indices[num_bonds] += 1 # Increment index for this length (1-based)
        final_key = f"{num_bonds}.{group_indices[num_bonds]}"
        final_grouped_results[final_key] = group_data

    return final_grouped_results

def lg_process_reset(lg_cgr, atom_num):
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

def lg_replacer(g_cgr: CGRContainer):
    """
    Extract dynamic leaving-groups from a CGR and mark attachment points.

    Scans the input CGRContainer for bonds lacking explicit p_order (i.e., leaving-group attachments),
    severs those bonds, captures each leaving-group as its own CGRContainer, and inserts DynamicX
    markers at the attachment sites. Finally, reindexes the markers to ensure unique labels.

    Parameters
    ----------
    g_cgr : CGRContainer
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
    
    cgr_prods = [g_cgr.substructure(c) for c in g_cgr.connected_components]
    target_cgr = cgr_prods[0]

    
    bond_items = list(target_cgr._bonds.items())
    reaction = ReactionContainer.from_cgr(target_cgr)
    target_mol = reaction.products[0]
    max_in_target_mol = max(target_mol._atoms)
    
    k = 1
    atom_nums = []
    
    for atom1, bond_set in bond_items:
        bond_set_items = list(bond_set.items())
        for atom2, bond in bond_set_items:
            if bond.p_order is None and bond.order is not None:
                if atom1 <= max_in_target_mol:
                    lg = DynamicX()
                    lg.mark = k
                    lg.isotope = k
                    order = bond.order
                    p_order = bond.p_order
                    target_cgr.delete_bond(atom1, atom2)
                    lg_cgrs = [target_cgr.substructure(c) for c in target_cgr.connected_components]
                    if len(lg_cgrs) == 2:
                        lg_cgr = lg_cgrs[1]
                        lg_cgr = lg_process_reset(lg_cgr, atom2)
                        lg_cgr.clean2d()
                    else:
                        continue
                    lg_groups[k] = (lg_cgr, atom2)
                    target_cgr = [target_cgr.substructure(c) for c in target_cgr.connected_components][0]
                    target_cgr.add_atom(lg, atom2)
                    if order == 4 and p_order == None:
                        order = 1
                    target_cgr.add_bond(atom1, atom2, DynamicBond(order, p_order))
                    target_cgr = [target_cgr.substructure(c) for c in target_cgr.connected_components][0]
                    k += 1
                    atom_nums.append(atom2)

    synthon_cgr = [target_cgr.substructure(c) for c in target_cgr.connected_components][0]
    reaction = ReactionContainer.from_cgr(synthon_cgr)
    reactants = reaction.reactants
    
    atom_mark_map = {}  # To map atom numbers to their new marks
    g = 1
    for n, r in enumerate(reactants):
        for atom_num in atom_nums:
            if atom_num in r._atoms:
                synthon_cgr._atoms[atom_num].mark = g
                atom_mark_map[atom_num] = g
                # print('reasssigned', atom_num, 'mark', g)
                g += 1
            
    new_lg_groups = {}
    for original_mark in lg_groups:
        cgr_obj, a_num = lg_groups[original_mark]
        new_mark = atom_mark_map.get(a_num)
        if new_mark is not None:
            new_lg_groups[new_mark] = (cgr_obj, a_num)
    lg_groups = new_lg_groups
    
    return synthon_cgr, lg_groups


def lg_reaction_replacer(synthon_reaction, lg_groups, max_in_target_mol):
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

def subcluster_one_cluster(group, rg_cgrs_dict, g_cgrs_dict):
    """
    Generate synthon data for each route in a single cluster.

    For each route (node ID) in `group['node_ids']`, replaces RouteCGRs with
    SynthonCGR, builds ReactionContainers before and after X replacement,
    and collects relevant data.

    Parameters
    ----------
    group : dict
        Must include `'node_ids'`, a list of node identifiers.
    rg_cgrs_dict : dict
        Maps node IDs to their reference RG-CGR strings.
    g_cgrs_dict : dict
        Maps node IDs to their generic RouteCGR strings.

    Returns
    -------
    dict or None
        If successful, returns a dict mapping each `node_id` to a tuple:
        `(rg_cgr, original_reaction, synthon_cgr, new_reaction, lg_groups)`.
        Returns `None` immediately if any step (X replacement or reaction
        parsing) fails for a node.

    """
    group_synthons = {}
    for node_id in group['node_ids']:
        rg_cgr = rg_cgrs_dict[node_id]
        g_cgr = g_cgrs_dict[node_id]
        try:
            synthon_cgr, lg_groups = lg_replacer(g_cgr)
        except:
            print('replacer', node_id)
            return None
        try:
            synthon_reaction = ReactionContainer.from_cgr(synthon_cgr)
        except:
            print('reaction', node_id)
            return None
        old_reactants = ReactionContainer.from_cgr(synthon_cgr).reactants
        target_mol = synthon_reaction.products[0]
        max_in_target_mol = max(target_mol._atoms)
        new_reactants = lg_reaction_replacer(synthon_reaction, lg_groups, max_in_target_mol)
        new_synthon_reaction = ReactionContainer(reactants=new_reactants, products=[target_mol])
        group_synthons[node_id] = (rg_cgr, ReactionContainer(reactants=old_reactants, products=[target_mol]),  synthon_cgr, new_synthon_reaction, lg_groups)

    return group_synthons


def group_nodes_by_synthon_detail(data_dict):
    """
    Groups nodes based on synthon CGR (result[0]) and reaction (result[1]).
    The output includes a dictionary mapping node IDs to their result[2] value.

    Args:
        data_dict: Dictionary {node_id: [synthon_cgr, synthon_reaction, node_data, ...]}.

    Returns:
        Dictionary {group_index: {'rg_cgr': ... ,'synthon_cgr': ..., 'synthon_reaction': ...,
                                  'nodes_data': {node_id1: node_data1, ...}}}.
    """
    temp_groups = defaultdict(list)


    for node_id, result_list in data_dict.items():
        if len(result_list) < 4:
            group_key = (result_list[0], None) # Handle missing reaction
        else:
            try:
                group_key = (result_list[0], result_list[1], result_list[2], result_list[3])
            except TypeError:
                 print(f"Warning: Skipping node {node_id} because reaction data is not hashable: {type(result_list[1])}")
                 continue

        temp_groups[group_key].append(node_id)

    # 2. Format the output dictionary with sequential integer keys
    #    and include the node-specific data (result[2]) in a sub-dictionary.
    final_grouped_results = {}
    group_index = 1

    sorted_temp_groups = sorted(temp_groups.items(), key=lambda item: item[1])
    for group_key, node_ids in sorted_temp_groups:
        
        rg_cgr, unlabeled_reaction, synthon_cgr, synthon_reaction = group_key
        nodes_data_dict = {} 

        # Iterate through the node IDs belonging to this group
        for node_id in sorted(node_ids): # Sort node IDs for consistent dict order
            original_result = data_dict.get(node_id, []) # Get original list for this node
            node_specific_data = None # Default value if index 2 is missing
            if len(original_result) > 4:
                node_specific_data = original_result[4] # Get the third element

            nodes_data_dict[node_id] = node_specific_data # Add to the sub-dictionary

        final_grouped_results[group_index] = {
            'rg_cgr': rg_cgr,
            'unlabeled_reaction': unlabeled_reaction,
            'synthon_cgr': synthon_cgr,
            'synthon_reaction': synthon_reaction,
            'nodes_data': nodes_data_dict,
            'post_processed' : False
        }
        group_index += 1

    return final_grouped_results


def subcluster_all_clusters(groups, rg_cgrs_dict, g_cgrs_dict):
    """
    Subdivide each reaction cluster into detailed synthon-based subgroups.

    Iterates over all clusters in `groups`, applies `subcluster_one_cluster`
    to generate per-cluster synthons, then organizes nodes by synthon detail.

    Parameters
    ----------
    groups : dict
        Mapping of cluster indices to cluster data.
    rg_cgrs_dict : dict
        Reference CGRs for reactant groups.
    g_cgrs_dict : dict
        Reference CGRs for generic groups.

    Returns
    -------
    dict or None
        A dict mapping each cluster index to its subgroups dict,
        or None if any cluster fails to subcluster.
    """
    all_subgroups = {}
    for group_index, group in groups.items():
        group_synthons = subcluster_one_cluster(group, rg_cgrs_dict, g_cgrs_dict)
        if group_synthons is None:
            return None
        subgroup = group_nodes_by_synthon_detail(group_synthons)
        all_subgroups[group_index] = subgroup
    return all_subgroups


def all_lg_collect(subgroup):
    """
    Gather all leaving-group CGRContainers by node index.

    Scans `subgroup['nodes_data']`, collects every CGRContainer per index,
    and returns a mapping from each index to the list of distinct containers.

    Parameters
    ----------
    subgroup : dict
        Must contain 'nodes_data', a dict mapping pathway keys to
        dicts of {node_index: (CGRContainer, …)}.

    Returns
    -------
    dict[int, list[CGRContainer]]
        For each node index, a list of unique CGRContainer objects
        (duplicates by string are filtered out).
    """
    all_indices = set()
    for sub_dict in subgroup['nodes_data'].values():
        all_indices.update(sub_dict.keys())
    
    # Dynamically initialize result and seen dictionaries
    result = {idx: [] for idx in all_indices}
    seen = {idx: set() for idx in all_indices}
    
    # Populate the result with unique CGRContainer objects
    for sub_dict in subgroup['nodes_data'].values():
        for idx in sub_dict:
            cgr_container = sub_dict[idx][0]
            cgr_str = str(cgr_container)
            if cgr_str not in seen[idx]:
                seen[idx].add(cgr_str)
                result[idx].append(cgr_container)
    return result

def replace_leaving_groups_in_synthon(subgroup, to_remove):
    """
    Replace specified leaving groups (LG) in a synthon CGR with new fragments and return the updated CGR
    along with a mapping from adjusted LG marks to their atom indices.

    Parameters:
        subgroup (dict): Must contain:
            - 'synthon_cgr': the CGR object representing the synthon graph
            - 'nodes_data': mapping of node indices to LG replacement data
        to_remove (List[int]): List of LG marks to remove and replace.

    Returns:
        Tuple[CGR, Dict[int, int]]: 
            - The updated CGR with replacements
            - A dict mapping new LG marks to their atom indices in the updated CGR
    """
    # Extract the original CGR and leaving group replacement table
    original_cgr = subgroup['synthon_cgr']
    lg_table = next(iter(subgroup['nodes_data'].values()))

    updated_cgr = original_cgr

    removed_count = 0
    new_lgs = {}

    # Iterate through all atoms (index, atom_obj) in the CGR
    for atom_idx, atom_obj in list(updated_cgr.atoms()):
        # Skip non-LG atoms
        if atom_obj.__class__.__name__ != 'DynamicX':
            continue

        current_mark = atom_obj.mark
        if current_mark in to_remove:
            # Remove old LG(X): delete bond and atom
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

def post_process_subgroup(subgroup):
    """
    Drop leaving-groups common to all pathways and rebuild a minimal synthon.

    Scans the subgroup for leaving-groups present in every route, removes those
    from the CGR, re-assembles a clean ReactionContainer with the original core,
    updates `nodes_data`, and flags the dict as processed.

    Parameters
    ----------
    subgroup : dict
        Must include keys for `nodes_data` and the helpers
        (`all_lg_collect`, `find_const_lg`, etc.). If already
        post_processed, returns immediately.

    Returns
    -------
    dict
        The same dict, now with:
        - `'synthon_reaction'`: cleaned ReactionContainer
        - `'nodes_data'`: filtered node table
        - `'post_processed'`: True
    """
    if 'post_processed' in subgroup.keys() and subgroup['post_processed'] == True:
        return subgroup
    result = all_lg_collect(subgroup)
    # to find constant lg that need to be removed
    to_remove = [ind for ind, cgr_set in result.items() if len(cgr_set) == 1]
    new_synthon_cgr, new_lgs = replace_leaving_groups_in_synthon(subgroup, to_remove)
    synthon_reaction = ReactionContainer.from_cgr(new_synthon_cgr)
    synthon_reaction.clean2d()
    old_reactants = ReactionContainer.from_cgr(new_synthon_cgr).reactants
    target_mol = synthon_reaction.products[0] # TO DO: target_mol might be non 0
    max_in_target_mol = max(target_mol._atoms)
    new_reactants = new_lg_reaction_replacer(synthon_reaction, new_lgs, max_in_target_mol)
    new_synthon_reaction = ReactionContainer(reactants=new_reactants, products=[target_mol])
    new_synthon_reaction.clean2d()
    subgroup['synthon_reaction'] = new_synthon_reaction
    subgroup['nodes_data'] = remove_and_shift(subgroup['nodes_data'], to_remove)
    subgroup['post_processed'] = True
    subgroup['group_lgs'] = group_by_identical_values(subgroup['nodes_data'])
    return subgroup

def group_by_identical_values(nodes_data):
    """
    Groups entries in a nested dictionary based on identical sets of core values.

    Identifies route IDs whose inner dictionaries contain the
    same sequence of leaving groups, when ordered by subkey. These are collapsed into a single entry.

    Args:
        nodes_data (dict): A dictionary mapping outer keys to inner dictionaries.
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
    for outer_key, inner_dict in nodes_data.items():
        # Sort inner_dict items by subkey to ensure consistent ordering
        sorted_items = sorted(inner_dict.items(), key=lambda kv: kv[0])
        # Extract only the first element of each (value_obj, other_info) tuple
        signature = tuple(val_tuple[0] for _, val_tuple in sorted_items)
        signature_map[signature].append(outer_key)

    # Step 2: Build the grouped result
    grouped = {}
    for signature, outer_keys in signature_map.items():
        # Use the representative inner dict from the first outer key in this group
        rep_inner = nodes_data[outer_keys[0]]
        # Build mapping subkey -> value_obj
        rep_values = {subkey: val_tuple[0] for subkey, val_tuple in rep_inner.items()}
        # Store under tuple of grouped outer keys
        grouped_key = tuple(outer_keys)
        grouped[grouped_key] = rep_values

    sorted_grouped = dict(
        sorted(
            grouped.items(),
            key=lambda item: len(item[0]),
            reverse=True
        )
    )

    return sorted_grouped