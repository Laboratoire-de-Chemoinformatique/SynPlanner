"""Strategic-bond CGR (SB-CGR) representation derived from a RouteCGR."""

from chython.containers import CGRContainer, ReactionContainer
from chython.containers.bonds import DynamicBond

from synplan.chem.reaction.routes.representation.state import remove_transient_bonds


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
