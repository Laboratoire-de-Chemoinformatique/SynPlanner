
from CGRtools.containers.bonds import DynamicBond
from CGRtools.containers import ReactionContainer

def find_next_atom_num(reactions):
    """Find the next available atom number."""
    max_num = 0
    for reaction in reactions:
        cgr = reaction.compose()
        max_num = max(max_num, max(cgr._atoms.keys()))
    return max_num + 1

def get_clean_mapping(curr_prod, prod, reverse=False):
    """Get clean mapping between molecules while avoiding number conflicts."""
    dict_map = {}
    m = list(curr_prod.get_mapping(prod))
    
    if len(m) == 0:
        return dict_map

    curr_atoms = set(curr_prod._atoms.keys())
    prod_atoms = set(prod._atoms.keys())
    
    rr = m[0]
    
    # Build mapping while checking for conflicts
    for key, value in rr.items():
        if key != value:
            if value in rr.keys() and rr[value] != key:
                continue
                
            source = value if reverse else key
            target = key if reverse else value
            
            if reverse and target in curr_atoms:
                continue
            if not reverse and target in prod_atoms:
                continue
                
            dict_map[source] = target
    
    return dict_map

def validate_molecule_components(curr_mol, node_id):
    """Validate that molecule has only one connected component."""
    new_rmol = [curr_mol.substructure(c) for c in curr_mol.connected_components]
    if len(new_rmol) > 1:
        print(f'Error tree {node_id}: We have more than one molecule in one node')

def get_leaving_groups(products):
    """Extract leaving group atom numbers from products."""
    lg_atom_nums = []
    for i, prod in enumerate(products):
        if i != 0:  # Skip first product (main product)
            lg_atom_nums.extend(prod._atoms.keys())
    return lg_atom_nums

def process_first_reaction(first_react, tree, node_id):
    """Process first reaction in the route and initialize building block set."""
    bb_set = set()
    
    for curr_mol in first_react.reactants:
        react_key = tuple(curr_mol._atoms)
        react_key_set = set(react_key)
        
        if len(curr_mol) <= tree.config.min_mol_size or str(curr_mol) in tree.building_blocks:
            bb_set = react_key_set
        
        validate_molecule_components(curr_mol, node_id)
    
    return bb_set

def update_reaction_dict(reaction, node_id, mapping, react_dict, tree, bb_set, prev_remap=None):
    """Update reaction dictionary with new mappings."""
    for curr_mol in reaction.reactants:
        react_key = tuple(curr_mol._atoms)
        react_key_set = set(react_key)
        
        validate_molecule_components(curr_mol, node_id)
        
        if len(curr_mol) <= tree.config.min_mol_size or str(curr_mol) in tree.building_blocks:
            bb_set = bb_set.union(react_key_set)

        # Filter the mapping to include only keys present in the current react_key
        filtered_mapping = {k: v for k, v in mapping.items() if k in react_key_set}
        if prev_remap:
            prev_remappping = {k: v for k, v in prev_remap.items() if k in react_key_set}
            filtered_mapping.update(prev_remappping)
        react_dict[react_key] = filtered_mapping
    
    return react_dict, bb_set

def process_target_blocks(curr_products, curr_prod, lg_atom_nums, curr_lg_atom_nums, bb_set):
    """Process and collect target blocks for remapping."""
    target_block = []
    if len(curr_products) > 1:
        for prod in curr_products:
            dict_map = get_clean_mapping(curr_prod, prod)
            if prod._atoms.keys() != curr_prod._atoms.keys():
                for key in list(prod._atoms.keys()):
                    if key in lg_atom_nums or key in curr_lg_atom_nums:
                        target_block.append(key)
                    if key in bb_set:
                        target_block.append(key)
    return target_block

def compose_route_cgr(tree_or_routes, node_id):
    """
    Process a single synthesis route maintaining consistent state.

    Parameters
    ----------
    tree_or_routes : synplan.mcts.tree.Tree
        or dict mapping route_id -> {step_id: ReactionContainer}
    node_id : int
        the route index (in the Tree’s winning_nodes, or the dict’s keys)

    Returns
    -------
    dict or None
      - if successful: { 'cgr': <composed CGR>, 'reactions_dict': {step: ReactionContainer,…} }
      - on error: None
    """
    # ----------- dict-based branch ------------
    if isinstance(tree_or_routes, dict):
        routes_dict = tree_or_routes
        if node_id not in routes_dict:
            raise KeyError(f"Route {node_id} not in provided dict.")
        # grab and sort the ReactionContainers in chronological order
        step_map   = routes_dict[node_id]
        sorted_ids = sorted(step_map)
        reactions  = [step_map[i] for i in sorted_ids]

        # start from the last (final) reaction
        accum_cgr       = reactions[-1].compose()
        reactions_dict  = { len(reactions)-1: reactions[-1] }
        # now fold backwards through the earlier steps
        for idx in range(len(reactions)-2, -1, -1):
            rxn = reactions[idx]
            curr_cgr = rxn.compose()
            accum_cgr      = curr_cgr.compose(accum_cgr)
            reactions_dict[idx] = rxn

        return {'cgr': accum_cgr, 'reactions_dict': reactions_dict }


    # ----------- tree-based branch ------------
    tree = tree_or_routes
    try:
        # original tree-based logic:
        reactions = tree.synthesis_route(node_id)

        first_react = reactions[-1]
        reactions_dict = { len(reactions)-1: first_react }

        accum_cgr = first_react.compose()
        bb_set    = process_first_reaction(first_react, tree, node_id)
        react_dict = {}
        max_num    = find_next_atom_num(reactions)

        for step in range(len(reactions)-2, -1, -1):
            reaction = reactions[step]
            curr_cgr = reaction.compose()
            curr_prod = reaction.products[0]

            accum_products = accum_cgr.decompose()[1].split()
            lg_atom_nums   = get_leaving_groups(accum_products)
            curr_products  = curr_cgr.decompose()[1].split()

            tuple_atoms = tuple(curr_prod._atoms)
            prev_remap  = react_dict.get(tuple_atoms, {})

            if prev_remap:
                curr_cgr = curr_cgr.remap(prev_remap, copy=True)

            # identify new atom‐numbers for any overlap
            target_block = process_target_blocks(curr_products, curr_prod,
                                                 lg_atom_nums,
                                                 [list(p._atoms.keys()) for p in curr_products[1:]],
                                                 bb_set)
            mapping = {}
            for atom_num in sorted(target_block):
                if atom_num in accum_cgr._atoms and atom_num not in mapping:
                    mapping[atom_num] = max_num
                    max_num += 1

            # carry forward any clean remap on the product itself
            dict_map = {}
            for ap in accum_products:
                clean_map = get_clean_mapping(curr_prod, ap, reverse=True)
                if clean_map:
                    dict_map = clean_map
                    break
            if dict_map:
                curr_cgr = curr_cgr.remap(dict_map, copy=False)

            # update our react_dict & bb_set
            react_dict, bb_set = update_reaction_dict(
                reaction, node_id, mapping, react_dict, tree,
                bb_set, prev_remap
            )

            # apply the new overlap‐mapping
            if mapping:
                curr_cgr = curr_cgr.remap(mapping, copy=False)

            reactions_dict[step] = ReactionContainer.from_cgr(curr_cgr)
            accum_cgr = curr_cgr.compose(accum_cgr)

        return { 'cgr': accum_cgr, 'reactions_dict': reactions_dict }

    except Exception as e:
        print(f"Error processing node {node_id}: {e}")
        return None

def compose_all_route_cgrs(tree_or_routes, node_id=None):
    """
    Process routes (reassign atom mappings) to compose RouteCGR.

    Parameters
    ----------
    tree_or_routes : synplan.mcts.tree.Tree
        or dict mapping route_id -> {step_id: ReactionContainer}
    node_id : int or None
        if None, do *all* winning routes (or all keys of the dict);
        otherwise only that specific route.

    Returns
    -------
    dict or None
      - if node_id is None: {route_id: CGR, …}
      - if node_id is given: {node_id: CGR}
      - returns None on error
    """
    # dict-based branch
    if isinstance(tree_or_routes, dict):
        routes_dict = tree_or_routes

        def _single(rid):
            res = compose_route_cgr(routes_dict, rid)
            return res['cgr'] if res else None

        if node_id is not None:
            if node_id not in routes_dict:
                raise KeyError(f"Route {node_id} not in provided dict.")
            return { node_id: _single(node_id) }

        # all routes
        result = { rid: _single(rid) for rid in sorted(routes_dict) }
        return result

    # tree-based branch
    tree = tree_or_routes
    route_cgrs = {}

    if node_id is not None:
        res = compose_route_cgr(tree, node_id)
        if res:
            route_cgrs[node_id] = res['cgr']
        else:
            return None
        return route_cgrs

    for rid in sorted(set(tree.winning_nodes)):
        res = compose_route_cgr(tree, rid)
        if res:
            route_cgrs[rid] = res['cgr']

    return route_cgrs

def extract_reactions(tree, node_id=None):
    """
    Collect mapped reaction sequences from a synthesis tree.

    Traverses either a single branch (if `node_id` is given) or all winning routes,
    composing CGR-based reactions for each, and returns a dict of reaction mappings.
    Ensures that in every extracted reaction, atom indices are uniquely mapped (no overlaps)

    Parameters
    ----------
    tree : ReactionTree
        A retrosynthetic tree object with a `.winning_nodes` attribute and
        supporting `compose_route_cgr(...)`.
    node_id : hashable, optional
        If provided, only extract reactions for this specific node/route.

    Returns
    -------
    dict[node_id, dict]
        Maps each route terminal node ID to its `reactions_dict` (as returned
        by `compose_route_cgr`). Returns `None` if the specified `node_id` fails
        to produce valid reactions.
    """
    react_dict = {}
    if node_id is not None:
        result = compose_route_cgr(tree, node_id)
        if result:
            react_dict[node_id] = result['reactions_dict']
        else:
            return None
        return react_dict

    for node_id in set(tree.winning_nodes):
        result = compose_route_cgr(tree, node_id)
        if result:
            react_dict[node_id] = result['reactions_dict']
    
    return dict(sorted(react_dict.items()))

def compose_reduced_route_cgr(route_cgr):
    """
    Reduces a Routes Condensed Graph of reaction (G-CGR) by performing the following steps:
    
    1. Extracts substructures corresponding to connected components from the input G-CGR.
    2. Selects the first substructure as the target to work on.
    3. Iterates over all bonds in the target G-CGR:
       - If a bond is identified as a "leaving group" (its primary order is None while its original order is defined),
         the bond is removed.
       - If a bond has a modified order (both primary and original orders are integers) and the primary order is less than the original,
         the bond is deleted and then re-added with a new dynamic bond using the primary order (this updates the bond to the reduced form).
    4. After bond modifications, re-extracts the substructure from the target G-CGR (now called the reduced G-CGR or RG-CGR).
    5. If the charge distributions (_p_charges vs. _charges) differ, neutralizes the charges by setting them to zero.
    
    Finally, returns the reduced RouteCGR.
    """
    # Get all connected components of the G-CGR as separate substructures.
    cgr_prods = [route_cgr.substructure(c) for c in route_cgr.connected_components]
    target_cgr = cgr_prods[0]  # Choose the first substructure (main product) for further reduction.
    
    # Iterate over each bond in the target G-CGR.
    bond_items = list(target_cgr._bonds.items())
    for atom1, bond_set in bond_items:
        bond_set_items = list(bond_set.items())
        for atom2, bond in bond_set_items:

            # Removing bonds corresponding to leaving groups:
            # If product bond order is None (indicating a leaving group) but an original bond order exists,
            # delete the bond.
            if bond.p_order is None and bond.order is not None:
                target_cgr.delete_bond(atom1, atom2)

            # For bonds that have been modified (not leaving groups) where the new (primary) order is less than the original:
            # Remove the bond and re-add it using the DynamicBond with the primary order for both bond orders.
            elif type(bond.p_order) is int and type(bond.order) is int and bond.p_order != bond.order:
                p_order = int(bond.p_order)
                target_cgr.delete_bond(atom1, atom2)
                target_cgr.add_bond(atom1, atom2, DynamicBond(p_order, p_order))
                
    # After modifying bonds, extract the reduced G-CGR from the target's connected components.
    reduced_route_cgr = [target_cgr.substructure(c) for c in target_cgr.connected_components][0]

    # Neutralize charges if the primary charges and current charges differ.
    if reduced_route_cgr._p_charges != reduced_route_cgr._charges:
        for num, charge in reduced_route_cgr._charges.items():
            if charge != 0:
                reduced_route_cgr._atoms[num].charge = 0

    return reduced_route_cgr

def compose_all_reduced_route_cgrs(route_cgrs_dict):
    """
    Processes a collection (dictionary) of RouteCGRs to generate their reduced forms (ReducedRouteCGRs).
    
    Iterates over each G-CGR in the provided dictionary and applies the compose_reduced_route_cgr function.
    
    Returns:
        A dictionary where each key corresponds to the RG-CGR obtained from the input G-CGR.
    """
    all_reduced_route_cgrs = dict()
    for num, cgr in route_cgrs_dict.items():
        all_reduced_route_cgrs[num] = compose_reduced_route_cgr(cgr)
    return all_reduced_route_cgrs