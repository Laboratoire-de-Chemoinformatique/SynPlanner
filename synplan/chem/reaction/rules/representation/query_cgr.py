"""Torch-free QueryCGR numbering-invariant labels and canonical keys for reaction rules."""

from itertools import permutations, product
from math import factorial

from chython.containers import (
    CGRContainer,
    MoleculeContainer,
    QueryCGRContainer,
    QueryContainer,
    ReactionContainer,
)

# Cap on intra-class permutations before canonical_query_cgr_key falls back to
# greedy ordering; the product of class-size factorials can blow up.
_MAX_CANONICAL_PERMUTATIONS = 5000


def query_to_mol(query: QueryContainer) -> MoleculeContainer:
    """Converts a QueryContainer object into a MoleculeContainer object.

    :param query: A QueryContainer object representing the query structure.
    :return: A MoleculeContainer object that replicates the structure of the query.
    """
    new_mol = MoleculeContainer()
    for n, atom in query.atoms():
        new_mol.add_atom(
            atom.atomic_symbol, n, charge=atom.charge, is_radical=atom.is_radical
        )
    for i, j, bond in query.bonds():
        new_mol.add_bond(i, j, int(bond))
    return new_mol


def reaction_query_to_reaction(reaction_rule: ReactionContainer) -> ReactionContainer:
    """Converts a ReactionContainer object with query structures into a
    ReactionContainer with molecular structures.

    :param reaction_rule: A ReactionContainer object where reactants and products are
        QueryContainer objects.
    :return: A new ReactionContainer object where reactants and products are
        MoleculeContainer objects.
    """
    reactants = [query_to_mol(q) for q in reaction_rule.reactants]
    products = [query_to_mol(q) for q in reaction_rule.products]
    reagents = [
        query_to_mol(q) for q in reaction_rule.reagents
    ]  # Assuming reagents are also part of the rule
    reaction = ReactionContainer(reactants, products, reagents, reaction_rule.meta)
    reaction.name = reaction_rule.name
    return reaction


def cgr_from_reaction_rule(reaction_rule: ReactionContainer) -> CGRContainer:
    """Creates a CGR from the given reaction rule.

    :param reaction_rule: The reaction rule to be converted.
    :return: The resulting CGR.
    """

    reaction_rule = reaction_query_to_reaction(reaction_rule)
    cgr_rule = ~reaction_rule

    return cgr_rule


def query_cgr_atom_label(query_cgr: QueryCGRContainer, atom_number: int) -> tuple:
    """Numbering-invariant label for a single QueryCGR atom.

    Touches several private chython dicts (``_charges``, ``_p_charges``,
    ``_neighbors``, ``_p_neighbors``, ``_hybridizations``,
    ``_p_hybridizations``). QueryCGRContainer's ``__slots__`` declares these,
    so the access is stable across chython versions; chython does not currently
    expose a public canonical-form API.
    """
    atom = query_cgr.atom(atom_number)
    return (
        getattr(atom, "atomic_number", None),
        getattr(atom, "atomic_symbol", None),
        getattr(atom, "isotope", None),
        query_cgr._charges.get(atom_number, 0),
        query_cgr._p_charges.get(atom_number, 0),
        query_cgr._radicals.get(atom_number, False),
        query_cgr._p_radicals.get(atom_number, False),
        tuple(query_cgr._neighbors.get(atom_number, ())),
        tuple(query_cgr._p_neighbors.get(atom_number, ())),
        tuple(query_cgr._hybridizations.get(atom_number, ())),
        tuple(query_cgr._p_hybridizations.get(atom_number, ())),
    )


def query_cgr_bond_label(
    query_cgr: QueryCGRContainer, atom_1: int, atom_2: int
) -> tuple:
    """Numbering-invariant label for a single QueryCGR bond."""
    bond = query_cgr._bonds[atom_1][atom_2]
    return bond.order, bond.p_order


def compress_labels(labels: dict[int, tuple]) -> dict[int, int]:
    """Replace structural labels with dense integer ids (``repr`` for sort key
    because labels contain mixed types)."""
    label_to_order = {
        label: index
        for index, label in enumerate(sorted(set(labels.values()), key=repr))
    }
    return {atom: label_to_order[label] for atom, label in labels.items()}


def _refined_query_cgr_colors(query_cgr: QueryCGRContainer) -> dict[int, int]:
    """Run 1-WL colour refinement on the QueryCGR atom graph.

    Returns a stable colour per atom: atoms in the same final class are
    structurally indistinguishable under the chosen labels and so must be
    enumerated as a permutation group when deriving a canonical key.
    """
    atoms = tuple(query_cgr._atoms)
    colors = compress_labels(
        {atom: query_cgr_atom_label(query_cgr, atom) for atom in atoms}
    )

    for _ in range(len(atoms)):
        signatures = {}
        for atom in atoms:
            neighborhood = tuple(
                sorted(
                    [
                        (
                            query_cgr_bond_label(query_cgr, atom, neighbor),
                            colors[neighbor],
                        )
                        for neighbor in query_cgr._bonds[atom]
                    ],
                    key=repr,
                )
            )
            signatures[atom] = (colors[atom], neighborhood)
        refined = compress_labels(signatures)
        if refined == colors:
            return refined
        colors = refined
    return colors


def _query_cgr_order_encoding(
    query_cgr: QueryCGRContainer, order: tuple[int, ...]
) -> tuple:
    """Encode the QueryCGR as atom-labels and bond-labels in the given order.

    Bond labels are emitted with positional (not atom-mapping) endpoints so
    two graphs with identical chemistry but different atom numbers produce the
    same encoding under their respective canonical orderings.
    """
    atom_positions = {atom: index for index, atom in enumerate(order)}
    atom_labels = tuple(query_cgr_atom_label(query_cgr, atom) for atom in order)
    bond_labels = []
    for atom_1 in order:
        position_1 = atom_positions[atom_1]
        for atom_2 in query_cgr._bonds[atom_1]:
            position_2 = atom_positions[atom_2]
            if position_1 < position_2:
                bond_labels.append(
                    (
                        position_1,
                        position_2,
                        query_cgr_bond_label(query_cgr, atom_1, atom_2),
                    )
                )
    return atom_labels, tuple(sorted(bond_labels, key=repr))


def canonical_query_cgr_key(query_cgr: QueryCGRContainer) -> str:
    """Atom-numbering-invariant canonical key for a QueryCGRContainer.

    Use when you need to deduplicate query rules that are chemically the same
    but came out of extraction with different atom numbering. Chython's
    ``QueryCGRContainer.__str__`` honours insertion/atom order, so two graphs
    that differ only in numbering serialise to different strings and would be
    counted as distinct rules during extraction.

    The key preserves every query label, including ``neighbors`` and
    ``p_neighbors``, so rules that differ only in those filters remain
    distinguishable.

    Algorithm: 1-WL colour refinement to partition atoms into automorphism
    classes, then either exhaustive enumeration of intra-class permutations
    (when the product of class factorials is ≤ ``_MAX_CANONICAL_PERMUTATIONS``)
    or a deterministic greedy ordering as a fallback. The greedy fallback is
    deterministic but not provably canonical for highly symmetric graphs, so
    such rules may miss dedup opportunities.
    """
    atoms = tuple(query_cgr._atoms)
    if not atoms:
        return repr(((), ()))

    colors = _refined_query_cgr_colors(query_cgr)
    color_groups = []
    for color in sorted(set(colors.values())):
        color_groups.append(tuple(atom for atom in atoms if colors[atom] == color))

    permutation_count = 1
    for group in color_groups:
        permutation_count *= factorial(len(group))

    if permutation_count <= _MAX_CANONICAL_PERMUTATIONS:
        encodings = (
            _query_cgr_order_encoding(
                query_cgr,
                tuple(atom for group_order in group_orders for atom in group_order),
            )
            for group_orders in product(
                *(permutations(group) for group in color_groups)
            )
        )
        return repr(min(encodings, key=repr))

    order = tuple(
        atom
        for group in color_groups
        for atom in sorted(
            group,
            key=lambda atom: (
                repr(query_cgr_atom_label(query_cgr, atom)),
                len(query_cgr._bonds[atom]),
                atom,
            ),
        )
    )
    return repr(_query_cgr_order_encoding(query_cgr, order))


__all__ = [
    "canonical_query_cgr_key",
    "cgr_from_reaction_rule",
    "compress_labels",
    "query_cgr_atom_label",
    "query_cgr_bond_label",
    "query_to_mol",
    "reaction_query_to_reaction",
]
