"""Symmetry helpers for reaction-rule SMARTS."""

from collections import Counter
from collections.abc import Iterator

from chython.containers import QueryContainer, ReactionContainer

_IDENTITY_TARGET_MAP: tuple[tuple[int, int], ...] = ()


class _ProductGraph:
    __slots__ = ("adjacency", "edge_labels", "fingerprint", "node_labels")

    def __init__(
        self,
        *,
        node_labels: tuple[tuple, ...],
        edge_labels: tuple[tuple[int, int, tuple], ...],
    ) -> None:
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.adjacency = _graph_adjacency(len(node_labels), edge_labels)
        self.fingerprint = _graph_fingerprint(node_labels, self.adjacency, edge_labels)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fingerprint={self.fingerprint!r})"


def needs_decollapsed_matches(rule: ReactionContainer) -> bool:
    """Return True when LHS symmetry is broken on the product side.

    Chython's ``automorphism_filter`` deduplicates matches that hit the same set
    of target atoms. That is only safe when every non-identity automorphism of
    a rule's left-hand side preserves the whole right-hand side rule patch.
    Non-automorphic overlapping query mappings are outside this predicate's
    scope.
    """
    targets = _shared_atom_numbers(rule.reactants, rule.products)
    if len(targets) < 2:
        return False

    target_maps = set()
    for reactant in rule.reactants:
        for mapping in _query_automorphisms(reactant):
            target_map = _normalized_target_map(targets, mapping)
            if target_map == _IDENTITY_TARGET_MAP:
                continue
            target_maps.add(target_map)
    if not target_maps:
        return False

    product_graph_cache: dict[
        tuple[tuple[int, int], ...], tuple[_ProductGraph, ...]
    ] = {_IDENTITY_TARGET_MAP: _rhs_product_graphs(rule, targets, _IDENTITY_TARGET_MAP)}

    for target_map in target_maps:
        mapped_products = product_graph_cache.setdefault(
            target_map, _rhs_product_graphs(rule, targets, target_map)
        )
        if not _product_graph_multisets_equal(
            mapped_products, product_graph_cache[_IDENTITY_TARGET_MAP]
        ):
            return True
    return False


def _rhs_product_graphs(
    rule: ReactionContainer,
    targets: frozenset[int],
    target_map: tuple[tuple[int, int], ...],
) -> tuple[_ProductGraph, ...]:
    target_map_dict = dict(target_map)
    return tuple(
        _rhs_product_graph(product, targets, target_map_dict)
        for product in rule.products
    )


def _rhs_product_graph(
    product: QueryContainer,
    targets: frozenset[int],
    target_map: dict[int, int],
) -> _ProductGraph:
    atom_numbers = tuple(sorted(product.atoms_numbers))
    atom_index = {atom: index for index, atom in enumerate(atom_numbers)}
    edge_labels = []
    for atom_1, atom_2, bond in product.bonds():
        local_atom_1 = atom_index[atom_1]
        local_atom_2 = atom_index[atom_2]
        edge_labels.append(
            (
                min(local_atom_1, local_atom_2),
                max(local_atom_1, local_atom_2),
                _bond_signature(bond),
            )
        )

    return _ProductGraph(
        node_labels=tuple(
            _rhs_atom_label(product.atom(atom), atom, targets, target_map)
            for atom in atom_numbers
        ),
        edge_labels=tuple(sorted(edge_labels, key=repr)),
    )


def _rhs_atom_label(
    atom,
    atom_number: int,
    targets: frozenset[int],
    target_map: dict[int, int],
) -> tuple:
    if atom_number in targets:
        return (
            "target",
            target_map.get(atom_number, atom_number),
            _query_atom_invariant(atom),
        )
    return ("external", _query_atom_invariant(atom))


def _query_automorphisms(query: QueryContainer) -> Iterator[dict[int, int]]:
    """Yield non-identity automorphisms for a chython query graph.

    Chython exposes the graph automorphism engine used for molecule containers,
    but query containers do not have a public ``get_automorphism_mapping`` API.
    Use chython's graph search with query-atom invariants built locally.
    """
    try:
        from chython.algorithms.isomorphism import (
            _get_automorphism_mapping as chython_automorphism_mapping,
        )
    except ImportError as err:
        raise RuntimeError(
            "Cannot enumerate reaction-rule query automorphisms because the "
            "installed chython-synplan package does not expose "
            "chython.algorithms.isomorphism._get_automorphism_mapping."
        ) from err

    atom_invariants = {
        atom_number: _query_atom_invariant(atom) for atom_number, atom in query.atoms()
    }
    bonds = {atom_number: {} for atom_number in atom_invariants}
    for atom_1, atom_2, bond in query.bonds():
        bonds[atom_1][atom_2] = bond
        bonds[atom_2][atom_1] = bond
    yield from chython_automorphism_mapping(atom_invariants, bonds)


def _shared_atom_numbers(
    reactants: tuple[QueryContainer, ...],
    products: tuple[QueryContainer, ...],
) -> frozenset[int]:
    reactant_atoms = {
        atom_number for reactant in reactants for atom_number in reactant.atoms_numbers
    }
    product_atoms = {
        atom_number for product in products for atom_number in product.atoms_numbers
    }
    return frozenset(reactant_atoms & product_atoms)


def _normalized_target_map(
    targets: frozenset[int], mapping: dict[int, int]
) -> tuple[tuple[int, int], ...]:
    mapped_targets = tuple(
        sorted((target, mapping.get(target, target)) for target in targets)
    )
    if all(target == mapped for target, mapped in mapped_targets):
        return _IDENTITY_TARGET_MAP
    return mapped_targets


def _graph_adjacency(
    atom_count: int, edge_labels: tuple[tuple[int, int, tuple], ...]
) -> tuple[dict[int, tuple], ...]:
    adjacency = [{} for _ in range(atom_count)]
    for atom_1, atom_2, edge_label in edge_labels:
        adjacency[atom_1][atom_2] = edge_label
        adjacency[atom_2][atom_1] = edge_label
    return tuple(adjacency)


def _graph_fingerprint(
    node_labels: tuple[tuple, ...],
    adjacency: tuple[dict[int, tuple], ...],
    edge_labels: tuple[tuple[int, int, tuple], ...],
) -> tuple:
    colors = node_labels
    for _ in range(len(node_labels)):
        raw_colors = tuple(
            (
                colors[atom],
                tuple(
                    sorted(
                        (
                            (bond_signature, colors[neighbor])
                            for neighbor, bond_signature in neighbors.items()
                        ),
                        key=repr,
                    )
                ),
            )
            for atom, neighbors in enumerate(adjacency)
        )
        palette = {
            raw_color: color_id
            for color_id, raw_color in enumerate(sorted(set(raw_colors), key=repr))
        }
        refined_colors = tuple(palette[raw_color] for raw_color in raw_colors)
        if refined_colors == colors:
            colors = refined_colors
            break
        colors = refined_colors

    node_counts = Counter(colors)
    edge_counts = Counter(
        (
            min(colors[atom_1], colors[atom_2]),
            max(colors[atom_1], colors[atom_2]),
            edge_label,
        )
        for atom_1, atom_2, edge_label in edge_labels
    )

    return (
        len(node_labels),
        tuple(sorted(node_counts.items(), key=repr)),
        tuple(sorted(edge_counts.items(), key=repr)),
    )


def _product_graph_multisets_equal(
    left_graphs: tuple[_ProductGraph, ...],
    right_graphs: tuple[_ProductGraph, ...],
    *,
    use_fingerprint_prefilter: bool = True,
) -> bool:
    if len(left_graphs) != len(right_graphs):
        return False

    matches = tuple(
        tuple(
            _product_graphs_equal(
                left_graph,
                right_graph,
                use_fingerprint_prefilter=use_fingerprint_prefilter,
            )
            for right_graph in right_graphs
        )
        for left_graph in left_graphs
    )
    order = tuple(
        sorted(
            range(len(left_graphs)),
            key=lambda index: (
                sum(matches[index]),
                repr(left_graphs[index].fingerprint),
            ),
        )
    )
    used: set[int] = set()

    def backtrack(position: int) -> bool:
        if position == len(order):
            return True
        left_index = order[position]
        for right_index in range(len(right_graphs)):
            if right_index in used:
                continue
            if not matches[left_index][right_index]:
                continue
            used.add(right_index)
            if backtrack(position + 1):
                return True
            used.remove(right_index)
        return False

    return backtrack(0)


def _product_graphs_equal(
    left: _ProductGraph,
    right: _ProductGraph,
    *,
    use_fingerprint_prefilter: bool = True,
) -> bool:
    if len(left.node_labels) != len(right.node_labels):
        return False
    if len(left.edge_labels) != len(right.edge_labels):
        return False
    if use_fingerprint_prefilter and left.fingerprint != right.fingerprint:
        return False

    candidates = _isomorphism_candidates(left, right)
    if candidates is None:
        return False

    order = tuple(
        sorted(
            range(len(left.node_labels)),
            key=lambda atom: (
                len(candidates[atom]),
                -len(left.adjacency[atom]),
                repr(left.node_labels[atom]),
            ),
        )
    )
    mapping: dict[int, int] = {}
    used: set[int] = set()

    def backtrack(position: int) -> bool:
        if position == len(order):
            return True
        atom = order[position]
        for candidate in candidates[atom]:
            if candidate in used:
                continue
            if not _isomorphism_candidate_is_compatible(
                atom, candidate, left, right, mapping
            ):
                continue
            mapping[atom] = candidate
            used.add(candidate)
            if backtrack(position + 1):
                return True
            used.remove(candidate)
            del mapping[atom]
        return False

    return backtrack(0)


def _isomorphism_candidates(
    left: _ProductGraph,
    right: _ProductGraph,
) -> tuple[tuple[int, ...], ...] | None:
    candidates = []
    for atom, node_label in enumerate(left.node_labels):
        atom_candidates = tuple(
            candidate
            for candidate, candidate_label in enumerate(right.node_labels)
            if candidate_label == node_label
            and len(right.adjacency[candidate]) == len(left.adjacency[atom])
        )
        if not atom_candidates:
            return None
        candidates.append(atom_candidates)
    return tuple(candidates)


def _isomorphism_candidate_is_compatible(
    atom: int,
    candidate: int,
    left: _ProductGraph,
    right: _ProductGraph,
    mapping: dict[int, int],
) -> bool:
    for mapped_atom, mapped_candidate in mapping.items():
        left_bond = left.adjacency[atom].get(mapped_atom)
        right_bond = right.adjacency[candidate].get(mapped_candidate)
        if left_bond != right_bond:
            return False
    return True


def _bond_signature(bond) -> tuple:
    return (
        bond.order,
        bond.in_ring,
        bond.stereo,
    )


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


__all__ = [
    "needs_decollapsed_matches",
]
