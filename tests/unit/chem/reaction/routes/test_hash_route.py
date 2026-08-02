import pytest
from chython import smiles

from synplan.chem.reaction.routes.representation import compose_route_cgr
from synplan.chem.reaction.routes.representation.hash import (
    HASH_SCHEMA,
    compare_route_cgr_dicts,
    hash_route_cgrs,
    route_cgr_bucket_hash,
    route_cgr_hash,
    route_cgr_hash_without_route_order,
    route_cgrs_equal,
    route_order_variant_sets,
)


class _Atom:
    atomic_number = 6
    isotope = None
    charge = 0
    p_charge = 0
    is_radical = False
    p_is_radical = False
    route_order = None
    route_step_order = None


class _Bond:
    order = 1
    p_order = 1
    route_order = None
    route_step_order = None


class _RouteCGR:
    def __init__(self, edges):
        self._atoms = {atom_id: _Atom() for atom_id in range(1, 7)}
        self._bonds = {}
        self._edges = edges
        self.connected_components = [set(self._atoms)]

    def atoms(self):
        return self._atoms.items()

    def bonds(self):
        for atom1, atom2 in self._edges:
            yield atom1, atom2, _Bond()


def _transient_route_cgr():
    routes = {
        1: {
            0: smiles("[CH3:1].[CH3:2][Cl:3]>>[CH3:1][CH3:2].[ClH:3]"),
            1: smiles("[CH3:1][CH3:2]>>[CH4:1]"),
        }
    }
    return compose_route_cgr(routes, 1, preserve_transient_bonds=True)["cgr"]


def test_route_cgr_hash_ignores_atom_map_numbering():
    route_cgr = _transient_route_cgr()
    remapped = route_cgr.remap({1: 300, 2: 200, 3: 100}, copy=True)

    assert route_cgr_hash(route_cgr) == route_cgr_hash(remapped)
    assert route_cgrs_equal(route_cgr, remapped)


def test_route_cgr_hash_includes_transient_bonds():
    routes = {
        1: {
            0: smiles("[CH3:1].[CH3:2][Cl:3]>>[CH3:1][CH3:2].[ClH:3]"),
            1: smiles("[CH3:1][CH3:2]>>[CH4:1]"),
        }
    }
    default_cgr = compose_route_cgr(routes, 1, preserve_transient_bonds=False)["cgr"]
    transient_cgr = compose_route_cgr(routes, 1, preserve_transient_bonds=True)["cgr"]

    assert route_cgr_hash(default_cgr) != route_cgr_hash(transient_cgr)


def _bonds(route_cgr):
    return [bond for _, _, bond in route_cgr.bonds()]


def _atoms(route_cgr):
    return [atom for _, atom in route_cgr.atoms()]


@pytest.mark.parametrize(
    ("elements", "attribute", "mutate"),
    [
        (
            _bonds,
            "route_order",
            lambda bond: setattr(bond, "route_order", bond.route_order + 1),
        ),
        (_atoms, "route_order", lambda atom: atom.route_order.add(99)),
        (_bonds, "route_step_order", lambda bond: bond.route_step_order.add(99)),
        (_atoms, "route_step_order", lambda atom: atom.route_step_order.add(99)),
    ],
    ids=[
        "bond_route_order",
        "atom_route_order",
        "bond_route_step_order",
        "atom_route_step_order",
    ],
)
def test_route_cgr_hash_includes_route_order(elements, attribute, mutate):
    route_cgr = _transient_route_cgr()
    changed = route_cgr.copy()

    for element in elements(changed):
        if getattr(element, attribute, None) is not None:
            mutate(element)
            break

    assert route_cgr_hash(route_cgr) != route_cgr_hash(changed)


def test_route_cgr_hash_without_route_order_ignores_route_order():
    route_cgr = _transient_route_cgr()
    changed = route_cgr.copy()

    for _, _, bond in changed.bonds():
        if getattr(bond, "route_order", None) is not None:
            bond.route_order += 1
            bond.route_step_order.add(99)
            break

    assert route_cgr_hash(route_cgr) != route_cgr_hash(changed)
    assert route_cgr_hash_without_route_order(route_cgr) == (
        route_cgr_hash_without_route_order(changed)
    )


def test_route_cgr_hash_uses_container_charge_state():
    route_cgr = _transient_route_cgr()
    changed = route_cgr.copy()
    atom_num = next(iter(changed._atoms))

    changed._charges[atom_num] = changed._charges.get(atom_num, 0) + 1

    assert changed._atoms[atom_num].charge != changed._charges[atom_num]
    assert route_cgr_hash(route_cgr) != route_cgr_hash(changed)


def test_route_order_variant_sets_detects_route_order_only_changes():
    route_cgr = _transient_route_cgr()
    remapped = route_cgr.remap({1: 300, 2: 200, 3: 100}, copy=True)
    route_order_changed = route_cgr.copy()
    chemistry_changed = route_cgr.copy()
    atom_num = next(iter(chemistry_changed._atoms))

    for _, _, bond in route_order_changed.bonds():
        if getattr(bond, "route_order", None) is not None:
            bond.route_order += 1
            break

    chemistry_changed._charges[atom_num] = (
        chemistry_changed._charges.get(atom_num, 0) + 1
    )

    assert route_order_variant_sets(
        {
            10: route_cgr,
            20: route_order_changed,
            30: remapped,
            40: chemistry_changed,
        }
    ) == [[[10, 30], [20]]]


def test_route_cgr_hash_exactly_splits_wl_bucket_collisions():
    triangular_prism = _RouteCGR(
        [(1, 2), (1, 3), (1, 5), (2, 4), (2, 6), (3, 4), (3, 6), (4, 5), (5, 6)]
    )
    k33 = _RouteCGR(
        [(1, 3), (1, 4), (1, 6), (2, 3), (2, 5), (2, 6), (3, 4), (4, 5), (5, 6)]
    )

    assert route_cgr_bucket_hash(triangular_prism) == route_cgr_bucket_hash(k33)
    assert route_cgr_hash(triangular_prism) != route_cgr_hash(k33)
    assert not route_cgrs_equal(triangular_prism, k33)


def test_hash_route_cgrs_groups_duplicate_route_cgrs():
    route_cgr = _transient_route_cgr()
    remapped = route_cgr.remap({1: 300, 2: 200, 3: 100}, copy=True)

    result = hash_route_cgrs({2: remapped, 1: route_cgr})

    assert result["hash_schema"] == HASH_SCHEMA
    assert result["route_count"] == 2
    assert result["unique_hash_count"] == 1
    assert list(result["route_ids_by_hash"].values()) == [[1, 2]]


def test_compare_route_cgr_dicts_reports_overlap_and_uniques_by_hash():
    route_cgr = _transient_route_cgr()
    remapped = route_cgr.remap({1: 300, 2: 200, 3: 100}, copy=True)
    changed = route_cgr.copy()

    for _, _, bond in changed.bonds():
        if getattr(bond, "route_order", None) is not None:
            bond.route_order += 1
            break

    result = compare_route_cgr_dicts(
        {33: route_cgr, 36: changed},
        {101: remapped},
    )

    overlap = result["route_ids_overlap"]
    unique_1 = result["route_ids_unique_1"]
    unique_2 = result["route_ids_unique_2"]

    assert result["overlap_exact_count"] == 1
    assert result["overlap_exact_count_1"] == 1
    assert result["overlap_exact_count_2"] == 1
    assert len(overlap) == 1
    assert list(overlap.values()) == [
        {"route_cgr_dict_1": [33], "route_cgr_dict_2": [101]}
    ]
    assert list(unique_1.values()) == [[36]]
    assert next(iter(unique_1)).startswith("bucket:")
    assert list(result["route_ids_unique_1_by_bucket_hash"].values()) == [[36]]
    assert result["route_ids_unique_1_by_exact_hash"] == {}
    assert unique_2 == {}


def test_compare_route_cgr_dicts_exactly_splits_shared_wl_bucket_uniques():
    triangular_prism = _RouteCGR(
        [(1, 2), (1, 3), (1, 5), (2, 4), (2, 6), (3, 4), (3, 6), (4, 5), (5, 6)]
    )
    k33 = _RouteCGR(
        [(1, 3), (1, 4), (1, 6), (2, 3), (2, 5), (2, 6), (3, 4), (4, 5), (5, 6)]
    )

    result = compare_route_cgr_dicts({1: triangular_prism}, {2: k33})

    assert result["overlap_bucket_count"] == 1
    assert result["overlap_exact_count"] == 0
    assert result["route_ids_unique_1_by_bucket_hash"] == {}
    assert result["route_ids_unique_2_by_bucket_hash"] == {}
    assert list(result["route_ids_unique_1_by_exact_hash"].values()) == [[1]]
    assert list(result["route_ids_unique_2_by_exact_hash"].values()) == [[2]]
    assert next(iter(result["route_ids_unique_1"])).startswith("exact:")
    assert next(iter(result["route_ids_unique_2"])).startswith("exact:")
