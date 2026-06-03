from chython.containers import CGRContainer, MoleculeContainer, ReactionContainer
from chython.containers.bonds import DynamicBond
from synplan.chem.reaction_routes.route_cgr_container import (
    enable_route_cgr_container,
)
from synplan.chem.reaction_routes.route_cgr_state import (
    RouteDynamicBond,
    remove_transient_bonds,
    route_atom,
    transient_bond,
)

from synplan.mcts.tree import Tree


def _next_atom_number(*containers):
    max_num = 0
    for container in containers:
        atoms = getattr(container, "_atoms", {})
        if atoms:
            max_num = max(max_num, max(atoms))
    return max_num + 1


def _bond_key(atom1, atom2):
    return tuple(sorted((atom1, atom2)))


def _route_order_depths(reactions):
    final_step = len(reactions) - 1
    reactant_atoms = [
        {atom for reactant in reaction.reactants for atom in reactant._atoms}
        for reaction in reactions
    ]
    product_atoms = [
        {atom for product in reaction.products for atom in product._atoms}
        for reaction in reactions
    ]
    successors = {idx: set() for idx in range(len(reactions))}

    for idx, atoms in enumerate(product_atoms):
        for next_idx in range(idx + 1, len(reactions)):
            if atoms & reactant_atoms[next_idx]:
                successors[idx].add(next_idx)

    depths = {}

    def depth(idx):
        if idx in depths:
            return depths[idx]
        if idx == final_step:
            depths[idx] = 1
        else:
            next_depths = [depth(next_idx) for next_idx in successors[idx]]
            depths[idx] = (
                min(next_depths) + 1 if next_depths else final_step - idx + 1
            )
        return depths[idx]

    return [depth(idx) for idx in range(len(reactions))]


def _record_route_orders(cgr, route_order, bond_route_orders, atom_route_orders):
    for atom_num, atom in cgr._atoms.items():
        if getattr(atom, "is_dynamic", False):
            atom_route_orders.setdefault(atom_num, set()).add(route_order)

    for atom1, atom2, bond in cgr.bonds():
        if bond.order == bond.p_order:
            continue
        key = _bond_key(atom1, atom2)
        bond_route_orders.setdefault(key, set()).add(route_order)
        atom_route_orders.setdefault(atom1, set()).add(route_order)
        atom_route_orders.setdefault(atom2, set()).add(route_order)


def _set_bond(cgr, atom1, atom2, bond):
    cgr._bonds[atom1][atom2] = cgr._bonds[atom2][atom1] = bond


def _apply_route_orders(cgr, bond_route_orders, atom_route_orders):
    for atom1, atom2, bond in list(cgr.bonds()):
        if bond.order == bond.p_order and bond.order is not None:
            continue
        route_orders = bond_route_orders.get(_bond_key(atom1, atom2))
        if not route_orders:
            atom1_orders = atom_route_orders.get(atom1, set())
            atom2_orders = atom_route_orders.get(atom2, set())
            route_orders = (atom1_orders & atom2_orders) or (
                atom1_orders | atom2_orders
            )
        if not route_orders:
            continue
        route_order = min(route_orders)
        if isinstance(bond, RouteDynamicBond):
            bond.route_order = route_order
        else:
            _set_bond(
                cgr,
                atom1,
                atom2,
                RouteDynamicBond.from_bond(bond, route_order),
            )

    for atom_num, route_orders in atom_route_orders.items():
        if atom_num in cgr._atoms:
            cgr._atoms[atom_num] = route_atom(cgr._atoms[atom_num], route_orders)

    cgr.flush_cache()
    return cgr


def get_clean_mapping(
    curr_prod: MoleculeContainer, prod: MoleculeContainer, reverse: bool = False
):
    """
    Get a 'clean' atom mapping between two molecules, avoiding conflicts.

    This function attempts to establish a mapping between the atoms of two
    MoleculeContainer objects (`curr_prod` and `prod`). It uses an internal
    mapping mechanism and then filters the result to create a "clean" mapping.
    The cleaning process specifically avoids adding entries to the mapping
    where the source and target indices are the same, or where the target
    index already exists as a source in the mapping with a different target.
    It also checks for potential conflicts based on the atom keys present
    in the original molecules.

    Args:
        curr_prod (MoleculeContainer): The first MoleculeContainer object.
        prod (MoleculeContainer): The second MoleculeContainer object.
        reverse (bool, optional): If True, the mapping is generated in the
                                  reverse direction (from `prod` to `curr_prod`).
                                  Defaults to False (mapping from `curr_prod` to `prod`).

    Returns:
        dict: A dictionary representing the clean atom mapping. Keys are atom
              indices from the source molecule, and values are the corresponding
              atom indices in the target molecule. Returns an empty dictionary
              if no mapping is found or if the initial mapping is empty.
    """
    dict_map = {}
    rr = next(iter(curr_prod.get_mapping(prod)), None)
    if rr is None:
        return dict_map

    curr_atoms = set(curr_prod._atoms.keys())
    prod_atoms = set(prod._atoms.keys())

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


def validate_molecule_components(curr_mol: MoleculeContainer, route_id: int):
    """
    Validate that a molecule has only one connected component.

    This function checks if a given MoleculeContainer object represents a
    single connected molecule or multiple disconnected fragments. It extracts
    the connected components and prints an error message if more than one
    component is found, indicating a potential issue with the molecule
    representation within a specific tree node.

    Args:
        curr_mol (MoleculeContainer): The MoleculeContainer object to validate.
        route_id (int): The ID of the tree route associated with this molecule,
                       used for reporting purposes in the error message.
    """
    new_rmol = [curr_mol.substructure(c) for c in curr_mol.connected_components]
    if len(new_rmol) > 1:
        print(f"Error tree {route_id}: We have more than one molecule in one node")
        return 0

    return 1


def get_leaving_groups(products: list):
    """Extract leaving-group atom numbers from reaction products.

    The first product is treated as the main product. Atoms from every later
    product are collected as leaving-group atoms.

    Args:
        products: Product ``MoleculeContainer`` objects.

    Returns:
        Atom numbers from the leaving-group product fragments.
    """
    lg_atom_nums = []
    for i, prod in enumerate(products):
        if i != 0:  # Skip first product (main product)
            lg_atom_nums.extend(prod._atoms.keys())
    return lg_atom_nums


def process_first_reaction(first_react: ReactionContainer, tree: Tree, route_id: int):
    """
    Process the first reaction in a retrosynthetic route and initialize the building block set.

    This function takes the first reaction in a route, iterates through its
    reactants, validates that each reactant is a single connected component,
    and identifies potential building blocks. A reactant is considered a
    potential building block if its size is less than or equal to the
    minimum molecule size defined in the tree's configuration or if its
    SMILES string is present in the tree's building blocks set. The atom
    indices of such building blocks are collected into a set.

    Args:
        first_react (ReactionContainer): The first ReactionContainer object in the route.
        tree (Tree): The Tree object containing the retrosynthetic search tree
                     and configuration (including `min_mol_size` and `building_blocks`).
        route_id (int): The ID of the tree node associated with this reaction,
                       used for validation reporting.

    Returns:
        set: A set of integer atom indices corresponding to the atoms
             identified as part of building blocks in the first reaction's reactants.
    """
    bb_set = set()

    for curr_mol in first_react.reactants:
        react_key = tuple(curr_mol._atoms)
        react_key_set = set(react_key)

        if (
            len(curr_mol) <= tree.config.min_mol_size
            or str(curr_mol) in tree.building_blocks
        ):
            bb_set = bb_set.union(react_key_set)

        if validate_molecule_components(curr_mol, route_id) == 0:
            return set()

    return bb_set


def update_reaction_dict(
    reaction: ReactionContainer,
    route_id: int,
    mapping: dict,
    react_dict: dict,
    tree: Tree,
    bb_set: set,
    prev_remap: dict = None,
):
    """
    Update a reaction dictionary with atom mappings and identify building blocks.

    This function processes the reactants of a given reaction, validates their
    structure (single connected component), updates a dictionary (`react_dict`)
    with atom mappings for each reactant, and expands a set of building block
    atom indices (`bb_set`). The mapping is filtered based on the atoms present
    in the current reactant, and can optionally include a previous remapping.
    Reactants are identified as building blocks based on size or presence in
    the tree's building blocks set.

    Args:
        reaction (ReactionContainer): The ReactionContainer object representing the reaction.
        route_id (int): The ID of the tree node associated with this synthetic route,
                       used for validation reporting.
        mapping (dict): The primary atom mapping dictionary to filter and apply.
        react_dict (dict): The dictionary to update with filtered mappings for each reactant.
                           Keys are tuples of atom indices for each reactant molecule.
        tree (Tree): The Tree object containing the retrosynthetic search tree
                     and configuration (including `min_mol_size` and `building_blocks`).
        bb_set (set): The set of building block atom indices to update.
        prev_remap (dict, optional): An optional dictionary representing a previous
                                     remapping to include in the filtered mapping.
                                     Defaults to None.

    Returns:
        tuple: A tuple containing:
               - dict: The updated `react_dict` with filtered mappings for each reactant.
               - set: The updated `bb_set` including atom indices from newly identified
                      building blocks.
    """
    for curr_mol in reaction.reactants:
        react_key = tuple(curr_mol._atoms)
        react_key_set = set(react_key)

        if validate_molecule_components(curr_mol, route_id) == 0:
            return dict(), set()

        if (
            len(curr_mol) <= tree.config.min_mol_size
            or str(curr_mol) in tree.building_blocks
        ):
            bb_set = bb_set.union(react_key_set)

        # Filter the mapping to include only keys present in the current react_key
        filtered_mapping = {k: v for k, v in mapping.items() if k in react_key_set}
        if prev_remap:
            prev_remapping = {
                k: v for k, v in prev_remap.items() if k in react_key_set
            }
            filtered_mapping.update(prev_remapping)
        react_dict[react_key] = filtered_mapping

    return react_dict, bb_set


def process_target_blocks(
    curr_products: list,
    curr_prod: MoleculeContainer,
    lg_atom_nums: list,
    curr_lg_atom_nums: list,
    bb_set: set,
):
    """
    Identifies and collects atom indices for target blocks based on leaving groups and building blocks.

    This function iterates through a list of current product molecules, compares their atoms
    to a reference molecule (`curr_prod`), and collects the indices of atoms that correspond
    to atoms in the provided leaving group lists (`lg_atom_nums`, `curr_lg_atom_nums`) or
    the building block set (`bb_set`). This is typically used to identify parts of molecules
    that should be treated as 'target blocks' during a remapping or analysis process.

    Args:
        curr_products (list): A list of MoleculeContainer objects representing the current products.
        curr_prod (MoleculeContainer): A reference MoleculeContainer object, likely the main product,
                                       used for mapping atom indices.
        lg_atom_nums (list): A list of integer atom indices identified as leaving group atoms
                             in a relevant context.
        curr_lg_atom_nums (list): Another list of integer atom indices identified as leaving
                                   group atoms, potentially from a different context than `lg_atom_nums`.
        bb_set (set): A set of integer atom indices identified as building block atoms.

    Returns:
        list: A list of integer atom indices that are identified as 'target blocks' based on
              their presence in the leaving group lists or building block set after mapping
              to the reference molecule.
    """
    target_block = set()
    target_atoms = set(lg_atom_nums) | set(curr_lg_atom_nums) | set(bb_set)
    if len(curr_products) > 1:
        for prod in curr_products:
            if prod._atoms.keys() != curr_prod._atoms.keys():
                for key in prod._atoms:
                    if key in target_atoms:
                        target_block.add(key)
    return list(target_block)


def _compose_cgrs(
    curr_cgr: CGRContainer,
    accum_cgr: CGRContainer,
    preserve_transient_bonds: bool,
):
    composed_cgr = curr_cgr.compose(accum_cgr)
    if not preserve_transient_bonds:
        return composed_cgr

    for atom1, atom2, bond in curr_cgr.bonds():
        next_bond = accum_cgr._bonds.get(atom1, {}).get(atom2)
        if (
            bond.order is None
            and (
                bond.p_order is None
                or (
                    next_bond is not None
                    and next_bond.order == bond.p_order
                    and next_bond.p_order is None
                )
            )
            and atom2 not in composed_cgr._bonds.get(atom1, {})
        ):
            if bond.p_order is None:
                composed_cgr.add_bond(atom1, atom2, bond)
                continue
            composed_cgr.add_bond(atom1, atom2, transient_bond())

    for atom1, atom2, bond in accum_cgr.bonds():
        if (
            bond.order is None
            and bond.p_order is None
            and atom2 not in composed_cgr._bonds.get(atom1, {})
        ):
            composed_cgr.add_bond(atom1, atom2, bond)

    return composed_cgr


def compose_route_cgr(tree_or_routes, route_id, preserve_transient_bonds=True):
    """
    Process a single synthesis route maintaining consistent state.

    Parameters
    ----------
    tree_or_routes : synplan.mcts.tree.Tree
        or dict mapping route_id -> {step_id: ReactionContainer}
    route_id : int
        the route index (in the Tree’s winning_nodes, or the dict’s keys)
    preserve_transient_bonds : bool
        If True, preserve transient route bonds that are formed in an earlier
        step and broken in a later step as DynamicBond(None, None). The default
        is True because RouteCGR hashing and comparison include transient route
        history by default.

    Returns
    -------
    dict or None
      - if successful: { 'cgr': <RouteCGRContainer>, 'reactions_dict': {step: ReactionContainer,…} }
      - on error: None
    """
    def remap_composition_conflicts(curr_cgr, accum_cgr, start_num):
        curr_atoms = curr_cgr._atoms
        accum_atoms = accum_cgr._atoms
        used_nums = set(curr_atoms) | set(accum_atoms)
        remap = {}
        next_num = start_num

        for atom_num in sorted(set(curr_atoms) & set(accum_atoms)):
            curr_atom = curr_atoms[atom_num]
            accum_atom = accum_atoms[atom_num]
            curr_identity = (
                curr_atom.atomic_number,
                getattr(curr_atom, "isotope", None),
            )
            accum_identity = (
                accum_atom.atomic_number,
                getattr(accum_atom, "isotope", None),
            )
            if curr_identity == accum_identity:
                continue
            while next_num in used_nums or next_num in remap.values():
                next_num += 1
            remap[atom_num] = next_num
            used_nums.add(next_num)
            next_num += 1

        if remap:
            curr_cgr = curr_cgr.remap(remap, copy=True)
        return curr_cgr, next_num, remap

    def update_react_remaps_for_conflicts(react_dict, reaction, remap):
        if not remap:
            return
        for curr_mol in reaction.reactants:
            react_key = tuple(curr_mol._atoms)
            if react_key not in react_dict:
                continue
            stored_remap = react_dict[react_key]
            for atom_num in react_key:
                mapped_atom_num = stored_remap.get(atom_num, atom_num)
                if mapped_atom_num in remap:
                    stored_remap[atom_num] = remap[mapped_atom_num]

    # ----------- dict-based route ------------
    if isinstance(tree_or_routes, dict):
        routes_dict = tree_or_routes
        if route_id not in routes_dict:
            raise KeyError(f"Route {route_id} not in provided dict.")
        # grab and sort the ReactionContainers in chronological order
        step_map = routes_dict[route_id]
        sorted_ids = sorted(step_map)
        reactions = [step_map[i] for i in sorted_ids]
        cgrs = [rxn.compose() for rxn in reactions]
        route_orders = _route_order_depths(reactions)
        bond_route_orders = {}
        atom_route_orders = {}
        _record_route_orders(
            cgrs[-1], route_orders[-1], bond_route_orders, atom_route_orders
        )

        # start from the last (final) reaction
        accum_cgr = cgrs[-1]
        reactions_dict = {
            len(reactions) - 1: ReactionContainer.from_cgr(cgrs[-1])
        }
        max_num = _next_atom_number(*cgrs)
        # now fold backwards through the earlier steps
        for idx in range(len(reactions) - 2, -1, -1):
            curr_cgr = cgrs[idx]
            curr_cgr, max_num, _ = remap_composition_conflicts(
                curr_cgr, accum_cgr, max_num
            )
            _record_route_orders(
                curr_cgr, route_orders[idx], bond_route_orders, atom_route_orders
            )
            accum_cgr = _compose_cgrs(
                curr_cgr, accum_cgr, preserve_transient_bonds
            )
            reactions_dict[idx] = ReactionContainer.from_cgr(curr_cgr)

        _apply_route_orders(accum_cgr, bond_route_orders, atom_route_orders)
        accum_cgr = enable_route_cgr_container(accum_cgr)
        return {"cgr": accum_cgr, "reactions_dict": reactions_dict}

    # ----------- tree-based route ------------
    tree = tree_or_routes
    try:
        # original tree-based logic:
        reactions = tree.synthesis_route(route_id)
        cgrs = [rxn.compose() for rxn in reactions]
        route_orders = _route_order_depths(reactions)
        bond_route_orders = {}
        atom_route_orders = {}
        _record_route_orders(
            cgrs[-1], route_orders[-1], bond_route_orders, atom_route_orders
        )

        first_react = reactions[-1]
        reactions_dict = {
            len(reactions) - 1: ReactionContainer.from_cgr(cgrs[-1])
        }

        accum_cgr = cgrs[-1]
        bb_set = process_first_reaction(first_react, tree, route_id)
        react_dict = {}
        max_num = _next_atom_number(*cgrs)

        for step in range(len(reactions) - 2, -1, -1):
            reaction = reactions[step]
            curr_cgr = cgrs[step]
            curr_prod = reaction.products[0]

            accum_products = accum_cgr.decompose()[1].split()
            lg_atom_nums = get_leaving_groups(accum_products)
            curr_products = curr_cgr.decompose()[1].split()

            tuple_atoms = tuple(curr_prod._atoms)
            prev_remap = react_dict.get(tuple_atoms, {})

            if prev_remap:
                curr_cgr = curr_cgr.remap(prev_remap, copy=True)

            # identify new atom-numbers for any overlap
            target_block = process_target_blocks(
                curr_products,
                curr_prod,
                lg_atom_nums,
                get_leaving_groups(curr_products),
                bb_set,
            )
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
                curr_cgr.remap(dict_map, copy=False)

            # update our react_dict & bb_set
            react_dict, bb_set = update_reaction_dict(
                reaction, route_id, mapping, react_dict, tree, bb_set, prev_remap
            )
            if not react_dict and not bb_set:
                return None

            # apply the new overlap-mapping
            if mapping:
                curr_cgr.remap(mapping, copy=False)

            curr_cgr, max_num, conflict_mapping = remap_composition_conflicts(
                curr_cgr, accum_cgr, max_num
            )
            update_react_remaps_for_conflicts(
                react_dict, reaction, conflict_mapping
            )
            _record_route_orders(
                curr_cgr, route_orders[step], bond_route_orders, atom_route_orders
            )

            reactions_dict[step] = ReactionContainer.from_cgr(curr_cgr)
            accum_cgr = _compose_cgrs(
                curr_cgr, accum_cgr, preserve_transient_bonds
            )

        _apply_route_orders(accum_cgr, bond_route_orders, atom_route_orders)
        accum_cgr = enable_route_cgr_container(accum_cgr)
        return {"cgr": accum_cgr, "reactions_dict": reactions_dict}

    except Exception as e:
        print(f"Error processing route {route_id}: {e}")
        return None


def compose_all_route_cgrs(
    tree_or_routes,
    route_id=None,
    preserve_transient_bonds=True,
):
    """
    Process routes (reassign atom mappings) to compose RouteCGR.

    Parameters
    ----------
    tree_or_routes : synplan.mcts.tree.Tree
        or dict mapping route_id -> {step_id: ReactionContainer}
    route_id : int or None
        if None, do *all* winning nodes (or all keys of the dict);
        otherwise only that specific route.
    preserve_transient_bonds : bool
        Forwarded to ``compose_route_cgr``. The default is True.

    Returns
    -------
    dict or None
      - if route_id is None: {route_id: CGR, …}
      - if route_id is given: {route_id: CGR}
      - returns None on error
    """
    # dict-based branch
    if isinstance(tree_or_routes, dict):
        routes_dict = tree_or_routes

        def _single(route_id):
            res = compose_route_cgr(
                routes_dict,
                route_id,
                preserve_transient_bonds=preserve_transient_bonds,
            )
            return res["cgr"] if res else None

        if route_id is not None:
            if route_id not in routes_dict:
                raise KeyError(f"Route {route_id} not in provided dict.")
            return {route_id: _single(route_id)}

        # all routes
        result = {route_id: _single(route_id) for route_id in sorted(routes_dict)}
        return result

    # tree-based branch
    tree = tree_or_routes
    route_cgrs = {}

    if route_id is not None:
        res = compose_route_cgr(
            tree,
            route_id,
            preserve_transient_bonds=preserve_transient_bonds,
        )
        if res:
            route_cgrs[route_id] = res["cgr"]
        else:
            return None
        return route_cgrs

    for route_id in sorted(set(tree.winning_nodes)):
        res = compose_route_cgr(
            tree,
            route_id,
            preserve_transient_bonds=preserve_transient_bonds,
        )
        if res:
            route_cgrs[route_id] = res["cgr"]

    return route_cgrs


def extract_reactions(tree: Tree, route_id=None, preserve_transient_bonds=True):
    """
    Collect mapped reaction sequences from a synthesis tree (basically routes_dict, which might be later converted to routes_json).

    Traverses either a single branch (if `route_id` is given) or all winning nodes,
    composing CGR-based reactions for each, and returns a dict of reaction mappings.
    Ensures that in every extracted reaction, atom indices are uniquely mapped (no overlaps)

    Parameters
    ----------
    tree : ReactionTree
        A retrosynthetic tree object with a `.winning_nodes` attribute and
        supporting `compose_route_cgr(...)`.
    route_id : hashable, optional
        If provided, only extract reactions for this specific route/route.
    preserve_transient_bonds : bool
        Forwarded to ``compose_route_cgr``. The default is True.

    Returns
    -------
    dict[route_id, dict]
        Maps each route terminal route ID to its `reactions_dict` (as returned
        by `compose_route_cgr`). Returns `None` if the specified `route_id` fails
        to produce valid reactions.
    """
    react_dict = {}
    if route_id is not None:
        result = compose_route_cgr(
            tree,
            route_id,
            preserve_transient_bonds=preserve_transient_bonds,
        )
        if result:
            react_dict[route_id] = result["reactions_dict"]
        else:
            return None
        return react_dict

    for route_id in set(tree.winning_nodes):
        result = compose_route_cgr(
            tree,
            route_id,
            preserve_transient_bonds=preserve_transient_bonds,
        )
        if result:
            react_dict[route_id] = result["reactions_dict"]

    return dict(sorted(react_dict.items()))


def compose_sb_cgr(route_cgr: CGRContainer):
    """Reduce a RouteCGR to the synthon/building-block CGR used for clustering.

    The reduction keeps the target RouteCGR component, removes transient bonds,
    removes leaving-group bonds, simplifies selected dynamic bonds, and keeps
    product-side charge/radical deltas synchronized with Chython's atom objects.

    Args:
        route_cgr: RouteCGR object to reduce.

    Returns:
        The reduced RouteCGR object.
    """
    route_cgr = remove_transient_bonds(route_cgr.copy())

    # Get the RouteCGR component with the target product.  Route construction
    # keeps target atoms at the lowest atom numbers, while later leaving-group
    # atoms can be remapped above that range.
    cgr_prods = [route_cgr.substructure(c) for c in route_cgr.connected_components]
    target_cgr = min(cgr_prods, key=lambda cgr: min(cgr._atoms))

    # `ReactionContainer.from_cgr` may split off several product fragments
    # (leaving groups, salts, etc.).  The target product is the fragment with
    # the lowest atom numbers; after deleting leaving-group bonds we should
    # return exactly these atoms, not whichever connected component happens to
    # be yielded first.
    reaction = ReactionContainer.from_cgr(target_cgr)
    target_product = min(reaction.products, key=lambda mol: min(mol._atoms))
    target_atom_nums = set(target_product._atoms)

    # Iterate over each bond in the target RouteCGR.
    checked_bonds = set()
    bond_items = list(target_cgr._bonds.items())
    for atom1, bond_set in bond_items:
        bond_set_items = list(bond_set.items())
        for atom2, bond in bond_set_items:
            bond_key = tuple(sorted((atom1, atom2)))
            if bond_key in checked_bonds:
                continue
            checked_bonds.add(bond_key)

            # Removing bonds corresponding to leaving groups:
            # If product bond order is None (indicating a leaving group) but an original bond order exists,
            # delete the bond.
            if bond.p_order is None and bond.order is not None:
                target_cgr.delete_bond(atom1, atom2)

            # For bonds that have been modified (not leaving groups) where the new (primary) order is less than the original:
            # Remove the bond and re-add it using the DynamicBond with the primary order for both bond orders.
            elif (
                type(bond.p_order) is int
                and type(bond.order) is int
                and bond.p_order != bond.order
            ):
                p_order = int(bond.p_order)
                target_cgr.delete_bond(atom1, atom2)
                target_cgr.add_bond(atom1, atom2, DynamicBond(p_order, p_order))

    # After modifying bonds, extract the target atom substructure.  Selecting
    # connected_components[0] here is order-dependent and can return a detached
    # leaving group instead of the target scaffold.
    sb_cgr = target_cgr.substructure(target_atom_nums)

    # Neutralize current-side charge deltas while preserving unchanged charged
    # atoms and product-side deltas.
    if sb_cgr._p_charges != sb_cgr._charges:
        for num, charge in sb_cgr._charges.items():
            if charge != 0 and charge != sb_cgr._p_charges[num]:
                sb_cgr._charges[num] = 0

    # Chython copies DynamicElement objects during substructure extraction.
    # Keep atom objects in sync with the CGR state dictionaries so SMILES
    # serialization does not see stale charges or radicals after reduction.
    for atom_num, atom in sb_cgr._atoms.items():
        atom._charge = sb_cgr._charges[atom_num]
        atom._p_charge = sb_cgr._p_charges[atom_num]
        atom._is_radical = sb_cgr._radicals[atom_num]
        atom._p_is_radical = sb_cgr._p_radicals[atom_num]
    sb_cgr.flush_cache()

    return sb_cgr


def compose_all_sb_cgrs(route_cgrs_dict: dict):
    """
    Processes a collection (dictionary) of RouteCGRs to generate their reduced forms (ReducedRouteCGRs).

    Iterates over each RouteCGR in the provided dictionary and applies the compose_sb_cgr function.

    Args:
        route_cgrs_dict (dict): A dictionary where keys are identifiers (e.g., route numbers)
                                and values are RouteCGR objects.

    Returns:
        dict: A dictionary where each key corresponds to the original identifier from
              `route_cgrs_dict` and the value is the corresponding ReducedRouteCGR object.
    """
    all_sb_cgrs = dict()
    for num, cgr in route_cgrs_dict.items():
        if cgr is None:
            continue
        all_sb_cgrs[num] = compose_sb_cgr(cgr)
    return all_sb_cgrs
