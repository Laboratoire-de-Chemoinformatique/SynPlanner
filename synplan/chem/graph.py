"""Graph operations missing from Chython 1.106's public container API.

Keep the storage-specific operations here. In particular, ordering a molecule
must not replace its native atom map with a Python dictionary. Query/CGR
neighbor order is significant for stereo and cannot be rebuilt from bonds().
"""

from chython._engine import native_graph
from chython._engine.compat import substructure_native
from chython.containers import MoleculeContainer
from chython.exceptions import BondNotFound


def neighbors(graph, number):
    """Adjacent atom numbers in stereo-reference order, without loading atoms."""
    backend = native_graph(graph)
    if backend is not None:
        return backend.row(backend.indices[number]).keys()
    if isinstance(graph, MoleculeContainer):
        return graph.environment(number, include_atom=False, include_bond=False)
    return graph._bonds[number].keys()


def bond_items(graph, number):
    """Adjacent (atom number, bond) pairs in stereo-reference order."""
    if isinstance(graph, MoleculeContainer):
        return graph.environment(number, include_atom=False)
    return graph._bonds[number].items()


def get_bond(graph, left, right, default=None):
    """Find a bond, including on CGRs which do not implement has_bond()."""
    try:
        return graph.bond(left, right)
    except BondNotFound:
        return default


def ordered_copy(graph, *, bonds=False):
    """Copy in numeric atom order, optionally sorting each neighbor row too.

    Sorting neighbors requires the caller to restore mapped stereo orientation.
    Native subgraph copying retains labels and coordinates without Python atoms.
    """
    result = graph.copy()
    if isinstance(graph, MoleculeContainer) and native_graph(graph) is not None:
        numbers = sorted(graph)
        if list(graph) != numbers:
            substructure_native(graph, result, numbers, False)
            result.calc_labels()
        if bonds:
            backend = native_graph(result)
            backend.set_neighbor_order(
                [sorted(backend.row(backend.indices[n])) for n in result]
            )
    else:
        result._atoms = dict(sorted(result.atoms()))
        if bonds:
            result._bonds = {n: dict(sorted(bond_items(result, n))) for n in result}
    result.flush_cache()
    return result


def replace_atom(graph, number, atom):
    """Replace a route atom with its metadata subclass, retaining its bonds."""
    graph.atom(number)  # reject missing atoms before changing topology
    graph._atoms[number] = atom


def set_bond(graph, left, right, bond):
    """Store a shared route bond without discarding its transient-state metadata."""
    graph.atom(left)
    graph.atom(right)
    graph._bonds[left][right] = graph._bonds[right][left] = bond


def pop_bond(graph, left, right):
    """Temporarily hide a route bond while retaining cached depiction geometry."""
    bond = graph.bond(left, right)
    del graph._bonds[left][right]
    del graph._bonds[right][left]
    return bond
