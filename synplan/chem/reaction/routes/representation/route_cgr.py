import logging
from typing import TYPE_CHECKING

from chython.containers import CGRContainer, MoleculeContainer, ReactionContainer

from synplan.chem.reaction.routes.contracts import (
    RouteCGRBuildResult,
    RouteDiagnostic,
)
from synplan.chem.reaction.routes.representation.container import (
    enable_route_cgr_container,
)
from synplan.chem.reaction.routes.representation.state import (
    RouteDynamicBond,
    _set_symmetric_bond,
    bond_key,
    route_atom,
    transient_bond,
)

if TYPE_CHECKING:
    from synplan.mcts.tree import Tree

logger = logging.getLogger(__name__)


def _next_atom_number(*containers):
    max_num = 0
    for container in containers:
        atoms = getattr(container, "_atoms", {})
        if atoms:
            max_num = max(max_num, max(atoms))
    return max_num + 1


def find_next_atom_num(reactions: list):
    """Next free atom number across a list of reactions (back-compat helper).

    Composes each reaction to its CGR and returns one more than the largest
    atom index seen, mirroring the historical ``find_next_atom_num`` API.
    """
    return _next_atom_number(*(reaction.compose() for reaction in reactions))


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
            depths[idx] = min(next_depths) + 1 if next_depths else final_step - idx + 1
        return depths[idx]

    return [depth(idx) for idx in range(len(reactions))]


def _record_route_orders(
    cgr,
    route_order,
    route_step_order,
    bond_route_orders,
    atom_route_orders,
    bond_route_step_orders,
    atom_route_step_orders,
):
    """Collect route depth and chronological step metadata after remapping."""

    for atom_num, atom in cgr._atoms.items():
        if getattr(atom, "is_dynamic", False):
            atom_route_orders.setdefault(atom_num, set()).add(route_order)
            atom_route_step_orders.setdefault(atom_num, set()).add(route_step_order)

    for atom1, atom2, bond in cgr.bonds():
        if bond.order == bond.p_order:
            continue
        key = bond_key(atom1, atom2)
        bond_route_orders.setdefault(key, set()).add(route_order)
        bond_route_step_orders.setdefault(key, set()).add(route_step_order)
        atom_route_orders.setdefault(atom1, set()).add(route_order)
        atom_route_orders.setdefault(atom2, set()).add(route_order)
        atom_route_step_orders.setdefault(atom1, set()).add(route_step_order)
        atom_route_step_orders.setdefault(atom2, set()).add(route_step_order)


def _atom_step_state(atom):
    return (atom.charge, atom.p_charge, atom.is_radical, atom.p_is_radical)


def _bond_step_state(bond):
    return (bond.order, bond.p_order)


def _record_deconvolution_labels(
    cgr,
    route_step_order,
    atom_step_states,
    bond_step_states,
):
    """Collect per-step CGR states needed for native RouteCGR deconvolution."""

    for atom_num, atom in cgr._atoms.items():
        atom_step_states.setdefault(atom_num, {})[route_step_order] = _atom_step_state(
            atom
        )

    for atom1, atom2, bond in cgr.bonds():
        bond_step_states.setdefault(bond_key(atom1, atom2), {})[route_step_order] = (
            _bond_step_state(bond)
        )


def _apply_route_orders(
    cgr,
    bond_route_orders,
    atom_route_orders,
    bond_route_step_orders,
    atom_route_step_orders,
    atom_step_states,
    bond_step_states,
    preserve_transient_bonds,
):
    if preserve_transient_bonds:
        for atom1, atom2 in sorted(
            set(bond_step_states) - {bond_key(a1, a2) for a1, a2, _ in cgr.bonds()}
        ):
            if atom1 in cgr._atoms and atom2 in cgr._atoms:
                cgr.add_bond(atom1, atom2, transient_bond())

    for atom1, atom2, bond in list(cgr.bonds()):
        key = bond_key(atom1, atom2)
        has_step_states = key in bond_step_states
        route_orders = bond_route_orders.get(key)
        if not route_orders:
            atom1_orders = atom_route_orders.get(atom1, set())
            atom2_orders = atom_route_orders.get(atom2, set())
            route_orders = (atom1_orders & atom2_orders) or (
                atom1_orders | atom2_orders
            )
        route_step_orders = bond_route_step_orders.get(key)
        if not route_step_orders:
            atom1_steps = atom_route_step_orders.get(atom1, set())
            atom2_steps = atom_route_step_orders.get(atom2, set())
            route_step_orders = (atom1_steps & atom2_steps) or (
                atom1_steps | atom2_steps
            )
        if not route_orders and not route_step_orders and not has_step_states:
            continue
        route_order = min(route_orders) if route_orders else None
        if isinstance(bond, RouteDynamicBond):
            bond.route_order = route_order
            bond.route_step_order = set(route_step_orders)
        else:
            bond = RouteDynamicBond.from_bond(
                bond,
                route_order,
                route_step_orders,
            )
            _set_symmetric_bond(cgr, atom1, atom2, bond)
        bond.route_bond_step_states = dict(bond_step_states.get(key, {}))

    for atom_num in sorted(
        set(atom_route_orders) | set(atom_route_step_orders) | set(atom_step_states)
    ):
        if atom_num in cgr._atoms:
            cgr._atoms[atom_num] = route_atom(
                cgr._atoms[atom_num],
                atom_route_orders.get(atom_num, set()),
                atom_route_step_orders.get(atom_num, set()),
            )
            cgr._atoms[atom_num].route_atom_step_states = dict(
                atom_step_states.get(atom_num, {})
            )

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
    # Build mapping while checking for conflicts
    for key, value in rr.items():
        if key != value:
            if value in rr and rr[value] != key:
                continue

            source = value if reverse else key
            target = key if reverse else value

            if source in curr_atoms and target in curr_atoms:
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
        logger.warning("Route %s: more than one molecule in one node", route_id)
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


def process_first_reaction(first_react: ReactionContainer, tree: "Tree", route_id: int):
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
    tree: "Tree",
    bb_set: set,
    prev_remap: dict | None = None,
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
            prev_remapping = {k: v for k, v in prev_remap.items() if k in react_key_set}
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


def _compose_route_cgr_legacy(
    tree_or_routes,
    route_id,
    preserve_transient_bonds=True,
    return_reactions_dict=False,
):
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
    return_reactions_dict : bool
        If True, also return a debug/compatibility ``reactions_dict``. The
        default keeps composition fast; use ``routes_dict_from_route_cgrs`` to
        deconvolute reactions from the returned RouteCGR when needed.

    Returns
    -------
    dict or None
      - if successful: { 'cgr': <RouteCGRContainer> }
      - if return_reactions_dict: also includes {step: ReactionContainer, ...}
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
        # Depth is kept for route interpretation; step order preserves exact
        # chronological route identity for hashing.
        route_step_orders = list(range(1, len(reactions) + 1))
        bond_route_orders = {}
        atom_route_orders = {}
        bond_route_step_orders = {}
        atom_route_step_orders = {}
        atom_step_states = {}
        bond_step_states = {}
        _record_deconvolution_labels(
            cgrs[-1],
            route_step_orders[-1],
            atom_step_states,
            bond_step_states,
        )
        _record_route_orders(
            cgrs[-1],
            route_orders[-1],
            route_step_orders[-1],
            bond_route_orders,
            atom_route_orders,
            bond_route_step_orders,
            atom_route_step_orders,
        )

        # start from the last (final) reaction
        accum_cgr = cgrs[-1]
        reactions_dict = (
            {len(reactions) - 1: ReactionContainer.from_cgr(cgrs[-1])}
            if return_reactions_dict
            else None
        )
        max_num = _next_atom_number(*cgrs)
        # now fold backwards through the earlier steps
        for idx in range(len(reactions) - 2, -1, -1):
            curr_cgr = cgrs[idx]
            curr_cgr, max_num, _ = remap_composition_conflicts(
                curr_cgr, accum_cgr, max_num
            )
            _record_deconvolution_labels(
                curr_cgr,
                route_step_orders[idx],
                atom_step_states,
                bond_step_states,
            )
            _record_route_orders(
                curr_cgr,
                route_orders[idx],
                route_step_orders[idx],
                bond_route_orders,
                atom_route_orders,
                bond_route_step_orders,
                atom_route_step_orders,
            )
            accum_cgr = _compose_cgrs(curr_cgr, accum_cgr, preserve_transient_bonds)
            if return_reactions_dict:
                reactions_dict[idx] = ReactionContainer.from_cgr(curr_cgr)

        _apply_route_orders(
            accum_cgr,
            bond_route_orders,
            atom_route_orders,
            bond_route_step_orders,
            atom_route_step_orders,
            atom_step_states,
            bond_step_states,
            preserve_transient_bonds,
        )
        accum_cgr = enable_route_cgr_container(accum_cgr)
        result = {"cgr": accum_cgr}
        if return_reactions_dict:
            result["reactions_dict"] = reactions_dict
        return result

    # ----------- tree-based route ------------
    tree = tree_or_routes
    try:
        # original tree-based logic:
        reactions = tree.synthesis_route(route_id)
        cgrs = [rxn.compose() for rxn in reactions]
        route_orders = _route_order_depths(reactions)
        # Depth is kept for route interpretation; step order preserves exact
        # chronological route identity for hashing.
        route_step_orders = list(range(1, len(reactions) + 1))
        bond_route_orders = {}
        atom_route_orders = {}
        bond_route_step_orders = {}
        atom_route_step_orders = {}
        atom_step_states = {}
        bond_step_states = {}
        _record_deconvolution_labels(
            cgrs[-1],
            route_step_orders[-1],
            atom_step_states,
            bond_step_states,
        )
        _record_route_orders(
            cgrs[-1],
            route_orders[-1],
            route_step_orders[-1],
            bond_route_orders,
            atom_route_orders,
            bond_route_step_orders,
            atom_route_step_orders,
        )

        first_react = reactions[-1]
        reactions_dict = (
            {len(reactions) - 1: ReactionContainer.from_cgr(cgrs[-1])}
            if return_reactions_dict
            else None
        )

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
                dict_map = {
                    source: target
                    for source, target in dict_map.items()
                    if source in curr_cgr._atoms and target not in curr_cgr._atoms
                }
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
            update_react_remaps_for_conflicts(react_dict, reaction, conflict_mapping)
            _record_deconvolution_labels(
                curr_cgr,
                route_step_orders[step],
                atom_step_states,
                bond_step_states,
            )
            _record_route_orders(
                curr_cgr,
                route_orders[step],
                route_step_orders[step],
                bond_route_orders,
                atom_route_orders,
                bond_route_step_orders,
                atom_route_step_orders,
            )

            if return_reactions_dict:
                reactions_dict[step] = ReactionContainer.from_cgr(curr_cgr)
            accum_cgr = _compose_cgrs(curr_cgr, accum_cgr, preserve_transient_bonds)

        _apply_route_orders(
            accum_cgr,
            bond_route_orders,
            atom_route_orders,
            bond_route_step_orders,
            atom_route_step_orders,
            atom_step_states,
            bond_step_states,
            preserve_transient_bonds,
        )
        accum_cgr = enable_route_cgr_container(accum_cgr)
        result = {"cgr": accum_cgr}
        if return_reactions_dict:
            result["reactions_dict"] = reactions_dict
        return result

    except Exception as e:
        logger.warning("Error processing route %s: %s", route_id, e)
        return None


def build_route_cgr(
    tree_or_routes,
    route_id,
    preserve_transient_bonds: bool = True,
    *,
    include_reactions: bool = False,
) -> RouteCGRBuildResult:
    """Build a RouteCGR and report recoverable composition failures explicitly."""

    result = _compose_route_cgr_legacy(
        tree_or_routes,
        route_id,
        preserve_transient_bonds=preserve_transient_bonds,
        return_reactions_dict=include_reactions,
    )
    if result is None:
        return RouteCGRBuildResult(
            route_id=route_id,
            diagnostic=RouteDiagnostic(
                route_id=route_id,
                stage="route_cgr_composition",
                message="RouteCGR composition did not produce a valid route",
            ),
        )
    return RouteCGRBuildResult(
        route_id=route_id,
        cgr=result["cgr"],
        reactions_dict=result.get("reactions_dict"),
    )


def compose_route_cgr(
    tree_or_routes,
    route_id,
    preserve_transient_bonds: bool = True,
    return_reactions_dict: bool = False,
):
    """Compatibility adapter returning the historical RouteCGR result mapping."""

    return build_route_cgr(
        tree_or_routes,
        route_id,
        preserve_transient_bonds,
        include_reactions=return_reactions_dict,
    ).as_legacy_dict(include_reactions=return_reactions_dict)


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
            result = build_route_cgr(
                routes_dict,
                route_id,
                preserve_transient_bonds=preserve_transient_bonds,
            )
            return result.cgr if result.ok else None

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
        result = build_route_cgr(
            tree,
            route_id,
            preserve_transient_bonds=preserve_transient_bonds,
        )
        if result.ok:
            route_cgrs[route_id] = result.cgr
        else:
            return None
        return route_cgrs

    for route_id in sorted(set(tree.winning_nodes)):
        result = build_route_cgr(
            tree,
            route_id,
            preserve_transient_bonds=preserve_transient_bonds,
        )
        if result.ok:
            route_cgrs[route_id] = result.cgr

    return route_cgrs


def extract_reactions(tree: "Tree", route_id=None, preserve_transient_bonds=True):
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
        result = build_route_cgr(
            tree,
            route_id,
            preserve_transient_bonds=preserve_transient_bonds,
            include_reactions=True,
        )
        if result.ok and result.reactions_dict is not None:
            react_dict[route_id] = result.reactions_dict
        else:
            return None
        return react_dict

    for route_id in set(tree.winning_nodes):
        result = build_route_cgr(
            tree,
            route_id,
            preserve_transient_bonds=preserve_transient_bonds,
            include_reactions=True,
        )
        if result.ok and result.reactions_dict is not None:
            react_dict[route_id] = result.reactions_dict

    return dict(sorted(react_dict.items()))
