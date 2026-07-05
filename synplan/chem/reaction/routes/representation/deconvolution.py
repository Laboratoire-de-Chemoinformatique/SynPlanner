"""Reconstruct route reaction dictionaries from native RouteCGR labels."""

from __future__ import annotations

from chython.containers import CGRContainer, ReactionContainer

from synplan.chem.reaction.routes.representation.state import RouteDynamicBond


def _bond_key(atom1: int, atom2: int) -> tuple[int, int]:
    return tuple(sorted((atom1, atom2)))


def _step_ids(route_cgr: CGRContainer) -> list[int]:
    steps = set()
    for atom in route_cgr._atoms.values():
        steps.update(getattr(atom, "route_atom_step_states", {}))
    for _, _, bond in route_cgr.bonds():
        steps.update(getattr(bond, "route_bond_step_states", {}))
    if not steps:
        raise ValueError(
            "RouteCGR does not carry native deconvolution labels. "
            "Build it with compose_route_cgr(..., preserve_transient_bonds=True)."
        )
    return sorted(int(step) for step in steps)


def _set_atom_state(
    cgr: CGRContainer, atom_num: int, state: tuple[int, int, bool, bool]
) -> None:
    charge, p_charge, is_radical, p_is_radical = state
    atom = cgr._atoms[atom_num]
    atom._charge = charge
    atom._p_charge = p_charge
    atom._is_radical = is_radical
    atom._p_is_radical = p_is_radical
    cgr._charges[atom_num] = charge
    cgr._p_charges[atom_num] = p_charge
    cgr._radicals[atom_num] = is_radical
    cgr._p_radicals[atom_num] = p_is_radical


def _set_bond(
    cgr: CGRContainer, atom1: int, atom2: int, bond: RouteDynamicBond
) -> None:
    cgr._bonds.setdefault(atom1, {})[atom2] = bond
    cgr._bonds.setdefault(atom2, {})[atom1] = bond


def _step_cgr(route_cgr: CGRContainer, step: int) -> CGRContainer:
    atom_nums = [
        atom_num
        for atom_num, atom in route_cgr._atoms.items()
        if step in getattr(atom, "route_atom_step_states", {})
    ]
    if not atom_nums:
        raise ValueError(f"RouteCGR deconvolution labels have no atoms for step {step}")

    step_cgr = route_cgr.substructure(atom_nums)

    for atom_num in atom_nums:
        state = route_cgr._atoms[atom_num].route_atom_step_states[step]
        _set_atom_state(step_cgr, atom_num, state)

    step_bonds = {}
    for atom1, atom2, bond in route_cgr.bonds():
        states = getattr(bond, "route_bond_step_states", {})
        if step in states:
            step_bonds[_bond_key(atom1, atom2)] = states[step]

    for atom1, atom2, _ in list(step_cgr.bonds()):
        if _bond_key(atom1, atom2) not in step_bonds:
            step_cgr.delete_bond(atom1, atom2)

    for atom1, atom2 in sorted(step_bonds):
        order, p_order = step_bonds[(atom1, atom2)]
        if order is None and p_order is None:
            continue
        _set_bond(step_cgr, atom1, atom2, RouteDynamicBond(order, p_order))

    step_cgr.flush_cache()
    return step_cgr


def reactions_from_route_cgr(route_cgr: CGRContainer) -> dict[int, ReactionContainer]:
    """Reconstruct mapped reaction steps from native RouteCGR labels."""

    return {
        step - 1: ReactionContainer.from_cgr(_step_cgr(route_cgr, step))
        for step in _step_ids(route_cgr)
    }


def routes_dict_from_route_cgrs(
    route_cgrs: dict[int, CGRContainer],
) -> dict[int, dict[int, ReactionContainer]]:
    """Convert ``route_id -> RouteCGR`` into ``route_id -> step_id -> Reaction``."""

    return {
        int(route_id): reactions_from_route_cgr(route_cgr)
        for route_id, route_cgr in sorted(route_cgrs.items())
    }
