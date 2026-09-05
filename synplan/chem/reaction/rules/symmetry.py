"""Symmetry helpers for reaction-rule SMARTS."""

from collections.abc import Iterator
from itertools import chain

# Chython 1.105 exposes automorphisms only for molecules, so query graphs must
# supply their own comparison labels to the shared private graph search.
from chython.algorithms.isomorphism import (
    _get_automorphism_mapping as chython_automorphism_mapping,
)
from chython.containers import QueryContainer, ReactionContainer

_TargetMap = tuple[tuple[int, int], ...]
_IDENTITY_TARGET_MAP: _TargetMap = ()
_HUB_ATOM_LABEL = ("__synplan_product_hub__",)
_HUB_BOND_LABEL = ("__synplan_product_hub_bond__",)


def needs_decollapsed_matches(rule: ReactionContainer) -> bool:
    """Return True when LHS symmetry is broken on the product side.

    Chython may deduplicate matches on the same target atoms. This is safe only
    when the RHS realizes every compatible LHS permutation.
    """
    targets = _shared_atom_numbers(rule.reactants, rule.products)
    if len(targets) < 2:
        return False

    required_maps = _lhs_target_maps(rule.reactants, targets)
    if not required_maps:
        return False
    # Chython's atom-set filter can keep a stereo-invalid orientation and
    # discard the valid one, even when the RHS topology is invariant.
    for reactant in rule.reactants:
        atoms = (atom for _, atom in reactant.atoms())
        bonds = (bond for _, _, bond in reactant.bonds())
        if any(item.stereo is not None for item in chain(atoms, bonds)):
            return True
    return not _rhs_realizes_target_maps(rule.products, targets, required_maps)


def _predicate_values(value) -> tuple:
    """Query predicates are tuples; concrete molecule properties are scalars."""
    if value is None:
        return ()
    return (value,) if isinstance(value, int) else tuple(value)


def _lhs_target_maps(
    reactants: tuple[QueryContainer, ...], targets: frozenset[int]
) -> set[_TargetMap]:
    """Collect compatible LHS permutations projected onto shared atom maps."""
    return {
        target_map
        for reactant in reactants
        for mapping in _query_automorphisms(reactant)
        if (target_map := _normalized_target_map(targets, mapping))
        != _IDENTITY_TARGET_MAP
    }


def _rhs_realizes_target_maps(
    products: tuple[QueryContainer, ...],
    targets: frozenset[int],
    required_maps: set[_TargetMap],
) -> bool:
    """Return whether RHS automorphisms realize every required target map.

    A uniquely labelled hub lets Chython exchange equivalent disconnected
    components. Synthetic vertex ids prevent component-local number collisions.
    """
    atom_invariants: dict[int, tuple] = {}
    bonds: dict[int, dict[int, object]] = {}
    target_vertices: dict[int, int] = {}
    next_vertex = 0

    for product in products:
        vertices = {}
        for atom_number, atom in product.atoms():
            vertex = next_vertex
            next_vertex += 1
            vertices[atom_number] = vertex
            is_target = atom_number in targets
            atom_invariants[vertex] = (
                "target" if is_target else "external",
                _query_atom_invariant(atom),
            )
            bonds[vertex] = {}
            if is_target:
                target_vertices[vertex] = atom_number

        for atom_1, atom_2, bond in product.bonds():
            vertex_1, vertex_2 = vertices[atom_1], vertices[atom_2]
            bond_label = (bond.order, bond.in_ring, bond.stereo)
            bonds[vertex_1][vertex_2] = bond_label
            bonds[vertex_2][vertex_1] = bond_label

    hub = next_vertex
    atom_invariants[hub] = _HUB_ATOM_LABEL
    bonds[hub] = {}
    for vertex in range(next_vertex):
        bonds[hub][vertex] = _HUB_BOND_LABEL
        bonds[vertex][hub] = _HUB_BOND_LABEL

    missing_maps = required_maps.copy()
    for mapping in chython_automorphism_mapping(atom_invariants, bonds):
        target_map = _induced_target_map(mapping, target_vertices)
        if target_map is None:
            continue
        missing_maps.discard(target_map)
        if not missing_maps:
            return True
    return False


def _induced_target_map(
    mapping: dict[int, int], target_vertices: dict[int, int]
) -> _TargetMap | None:
    """Project an RHS graph mapping onto consistent reaction-map numbers."""
    induced = {}
    for vertex, source_target in target_vertices.items():
        destination_target = target_vertices[mapping[vertex]]
        if induced.setdefault(source_target, destination_target) != destination_target:
            return None
    return _normalized_target_map(frozenset(induced), induced)


def _query_automorphisms(query: QueryContainer) -> Iterator[dict[int, int]]:
    """Yield topology automorphisms whose atom predicates can overlap."""
    atoms = dict(query.atoms())
    bonds: dict[int, dict[int, object]] = {atom_number: {} for atom_number in atoms}
    for atom_1, atom_2, _ in query.bonds():
        bonds[atom_1][atom_2] = bonds[atom_2][atom_1] = None

    for mapping in chython_automorphism_mapping(dict.fromkeys(atoms, 0), bonds):
        if all(
            _query_atoms_overlap(atom, atoms[mapping[atom_number]])
            for atom_number, atom in atoms.items()
        ) and all(
            (
                8 in _predicate_values(bond.order)
                or 8 in _predicate_values(mapped_bond.order)
                or not set(_predicate_values(bond.order)).isdisjoint(
                    _predicate_values(mapped_bond.order)
                )
            )
            and (
                bond.in_ring is None
                or mapped_bond.in_ring is None
                or bond.in_ring == mapped_bond.in_ring
            )
            for atom_1, atom_2, bond in query.bonds()
            for mapped_bond in (query._bonds[mapping[atom_1]][mapping[atom_2]],)
        ):
            yield mapping


def _query_atoms_overlap(left, right) -> bool:
    """Return whether two LHS atom predicates can match the same atom."""
    if left.is_radical != right.is_radical:
        return False
    symbols = left.atomic_symbol, right.atomic_symbol
    if "M" in symbols and symbols[0] != symbols[1]:
        return False
    elements = []
    for atom in (left, right):
        atom_elements = getattr(atom, "atomic_numbers", None)
        if atom_elements is None:
            atomic_number = getattr(atom, "atomic_number", None)
            atom_elements = () if atomic_number is None else (atomic_number,)
        elements.append(set(atom_elements))
    if elements[0] and elements[1] and elements[0].isdisjoint(elements[1]):
        return False

    left_isotope = getattr(left, "isotope", None)
    right_isotope = getattr(right, "isotope", None)
    if None not in (left_isotope, right_isotope) and left_isotope != right_isotope:
        return False

    charges = []
    for atom in (left, right):
        charge = getattr(atom, "_charge", None)
        values = {charge} if charge is not None else set(range(-4, 5))
        charge_not = getattr(atom, "charge_not", None)
        if charge_not == "positive":
            values = {value for value in values if value <= 0}
        elif charge_not == "negative":
            values = {value for value in values if value >= 0}
        charges.append(values)
    if charges[0].isdisjoint(charges[1]):
        return False

    for field in (
        "neighbors",
        "hybridization",
        "implicit_hydrogens",
        "heteroatoms",
        "total_connectivity",
        "rings_count",
        "valence",
        "ring_connectivity",
        "ring_sizes",
    ):
        left_values = _predicate_values(getattr(left, field, ()))
        right_values = _predicate_values(getattr(right, field, ()))
        if left_values and right_values and set(left_values).isdisjoint(right_values):
            return False
    return True


def _shared_atom_numbers(
    reactants: tuple[QueryContainer, ...],
    products: tuple[QueryContainer, ...],
) -> frozenset[int]:
    """Return atom-map numbers present on both sides of the reaction rule."""
    return frozenset(
        {number for reactant in reactants for number in reactant.atoms_numbers}
        & {number for product in products for number in product.atoms_numbers}
    )


def _normalized_target_map(
    targets: frozenset[int], mapping: dict[int, int]
) -> _TargetMap:
    """Project a graph mapping onto targets, using an empty tuple for identity."""
    mapped_targets = tuple(
        sorted((target, mapping.get(target, target)) for target in targets)
    )
    if all(target == mapped for target, mapped in mapped_targets):
        return _IDENTITY_TARGET_MAP
    return mapped_targets


def _query_atom_invariant(atom) -> tuple:
    return (
        atom.__class__,
        getattr(atom, "atomic_number", None),
        atom.atomic_symbol,
        getattr(atom, "isotope", None),
        getattr(atom, "charge", None),
        getattr(atom, "charge_not", None),
        getattr(atom, "is_radical", None),
        atom.neighbors,
        atom.hybridization,
        getattr(atom, "implicit_hydrogens", ()),
        getattr(atom, "ring_sizes", ()),
        getattr(atom, "rings_count", ()),
        getattr(atom, "total_connectivity", ()),
        getattr(atom, "heteroatoms", ()),
        getattr(atom, "excluded_elements", None),
        bool(getattr(atom, "recursive_smarts", None)),
        getattr(atom, "masked", False),
        getattr(atom, "stereo", None),
    )


__all__ = ["needs_decollapsed_matches"]
