#!/usr/bin/env python
"""Hash RouteCGRs with transient bonds and route ordering."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations, product
from typing import Any

from synplan.routes.route_cgr.builder import _bond_key

HASH_SCHEMA = "route-cgr-exact-v2"
BUCKET_HASH_SCHEMA = "route-cgr-wl-bucket-v2"
ROUTE_ORDER_AGNOSTIC_HASH_SCHEMA = "route-cgr-exact-without-route-order-v2"
ROUTE_ORDER_AGNOSTIC_BUCKET_HASH_SCHEMA = (
    "route-cgr-wl-bucket-without-route-order-v2"
)
HASH_INCLUDES = (
    "atom element, isotope, charge, radical state",
    "dynamic atom product charge and radical state",
    "atom route_order depth values",
    "atom route_step_order chronological values",
    "bond order and product order",
    "bond route_order depth value",
    "bond route_step_order chronological values",
    "transient bonds encoded as (None, None)",
)
HASH_EXCLUDES = (
    "stereochemistry",
    "implicit hydrogens",
    "atom-map numbers except as temporary graph node identifiers",
    "atom-map-only metadata",
    "cached canonical strings or cached Chython hash state",
    "Chython atom/bond attributes not explicitly listed in HASH_INCLUDES",
)

__all__ = [
    "BUCKET_HASH_SCHEMA",
    "HASH_EXCLUDES",
    "HASH_INCLUDES",
    "HASH_SCHEMA",
    "ROUTE_ORDER_AGNOSTIC_BUCKET_HASH_SCHEMA",
    "ROUTE_ORDER_AGNOSTIC_HASH_SCHEMA",
    "RouteCGRGraph",
    "atom_label",
    "bond_label",
    "compare_route_cgr_dicts",
    "hash_route_cgrs",
    "route_cgr_bucket_fingerprint",
    "route_cgr_bucket_hash",
    "route_cgr_fingerprint",
    "route_cgr_fingerprint_without_route_order",
    "route_cgr_graph",
    "route_cgr_hash",
    "route_cgr_hash_without_route_order",
    "route_cgr_metadata",
    "route_cgrs_equal",
    "route_order_variant_sets",
]


@dataclass(frozen=True)
class RouteCGRGraph:
    """Small labeled undirected graph representation for RouteCGR hashing."""

    node_labels: Mapping[int, str]
    edge_labels: Mapping[tuple[int, int], str]
    adjacency: Mapping[int, tuple[tuple[int, str], ...]]

    @property
    def number_of_nodes(self) -> int:
        return len(self.node_labels)

    @property
    def number_of_edges(self) -> int:
        return len(self.edge_labels)


@dataclass(frozen=True)
class _PreparedRouteCGR:
    """RouteCGR graph state shared by bucket and exact hashing."""

    graph: RouteCGRGraph
    colors: Mapping[int, str]
    history_hashes: tuple[str, ...]
    components: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _RouteCGRHashResult:
    bucket_hash: str
    exact_hash: str


@dataclass(frozen=True)
class _RouteCGRBucketResult:
    bucket_hash: str
    numbered_hash: str


def _json_label(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(_json_label(value).encode()).hexdigest()


def _sort_key(value: Any) -> str:
    return _json_label(value)


def _route_orders(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (set, frozenset, list, tuple)):
        return sorted(value, key=_sort_key)
    return [value]


def _atom_state(route_cgr: Any, state_name: str, atom_id: int, fallback: Any) -> Any:
    state = getattr(route_cgr, state_name, None)
    if isinstance(state, Mapping):
        return state.get(atom_id, fallback)
    return fallback


def atom_label(
    route_cgr: Any,
    atom_id: int,
    atom: Any,
    *,
    include_route_order: bool = True,
) -> str:
    """Return the atom properties that participate in a RouteCGR hash."""

    return _json_label(
        {
            "atomic_number": atom.atomic_number,
            "isotope": atom.isotope,
            "charge": _atom_state(route_cgr, "_charges", atom_id, atom.charge),
            "p_charge": _atom_state(
                route_cgr, "_p_charges", atom_id, atom.p_charge
            ),
            "is_radical": _atom_state(
                route_cgr, "_radicals", atom_id, atom.is_radical
            ),
            "p_is_radical": _atom_state(
                route_cgr, "_p_radicals", atom_id, atom.p_is_radical
            ),
            "route_orders": (
                _route_orders(getattr(atom, "route_order", None))
                if include_route_order
                else []
            ),
            "route_step_orders": (
                _route_orders(getattr(atom, "route_step_order", None))
                if include_route_order
                else []
            ),
        }
    )


def bond_label(bond: Any, *, include_route_order: bool = True) -> str:
    """Return bond properties, preserving transient and route-order metadata."""

    return _json_label(
        {
            "order": bond.order,
            "p_order": bond.p_order,
            "route_orders": (
                _route_orders(getattr(bond, "route_order", None))
                if include_route_order
                else []
            ),
            "route_step_orders": (
                _route_orders(getattr(bond, "route_step_order", None))
                if include_route_order
                else []
            ),
            "transient": bond.order is None and bond.p_order is None,
        }
    )


def route_cgr_graph(
    route_cgr: Any,
    *,
    include_route_order: bool = True,
) -> RouteCGRGraph:
    """Build a labeled RouteCGR graph without depending on atom-map numbering."""

    node_labels = {
        atom_id: atom_label(
            route_cgr,
            atom_id,
            atom,
            include_route_order=include_route_order,
        )
        for atom_id, atom in route_cgr.atoms()
    }
    edge_labels = {}
    adjacency = defaultdict(list)

    for atom_id in node_labels:
        adjacency.setdefault(atom_id, [])

    for atom1, atom2, bond in route_cgr.bonds():
        label = bond_label(bond, include_route_order=include_route_order)
        edge_labels[_bond_key(atom1, atom2)] = label
        adjacency[atom1].append((atom2, label))
        adjacency[atom2].append((atom1, label))

    return RouteCGRGraph(
        node_labels=node_labels,
        edge_labels=edge_labels,
        adjacency={
            atom_id: tuple(sorted(neighbors, key=_sort_key))
            for atom_id, neighbors in adjacency.items()
        },
    )


def _route_cgr_components(route_cgr: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            (tuple(sorted(component)) for component in route_cgr.connected_components),
            key=_sort_key,
        )
    )


def _prepare_route_cgr(
    route_cgr: Any,
    *,
    include_route_order: bool = True,
) -> _PreparedRouteCGR:
    graph = route_cgr_graph(route_cgr, include_route_order=include_route_order)
    colors, history_hashes = _wl_refinement(graph)
    return _PreparedRouteCGR(
        graph=graph,
        colors=colors,
        history_hashes=tuple(history_hashes),
        components=_route_cgr_components(route_cgr),
    )


def _count_labels(labels: list[str]) -> list[list[Any]]:
    return [[label, count] for label, count in sorted(Counter(labels).items())]


def _count_values(values: list[Any]) -> list[list[Any]]:
    labels = [_json_label(value) for value in values]
    return _count_labels(labels)


def _wl_refinement(
    graph: RouteCGRGraph,
) -> tuple[dict[int, str], list[str]]:
    """Return final node colors and refinement-history hashes."""

    colors = dict(graph.node_labels)
    history_hashes = [_stable_digest(_count_labels(list(colors.values())))]

    for _ in range(max(3, graph.number_of_nodes)):
        colors = {
            atom_id: _stable_digest(
                [
                    colors[atom_id],
                    sorted(
                        [
                            [edge_label, colors[neighbor_id]]
                            for neighbor_id, edge_label in graph.adjacency.get(
                                atom_id, ()
                            )
                        ],
                        key=_sort_key,
                    ),
                ]
            )
            for atom_id in sorted(graph.node_labels)
        }
        history_hashes.append(_stable_digest(_count_labels(list(colors.values()))))

    return colors, history_hashes


def _component_signatures(
    graph: RouteCGRGraph,
    colors: Mapping[int, str],
    components: list[tuple[int, ...]],
) -> list[dict[str, Any]]:
    signatures = []

    for component in components:
        component_atoms = set(component)
        component_edges = []
        for (atom1, atom2), label in graph.edge_labels.items():
            if atom1 not in component_atoms or atom2 not in component_atoms:
                continue
            endpoint_colors = sorted((colors[atom1], colors[atom2]))
            component_edges.append([endpoint_colors[0], label, endpoint_colors[1]])
        signatures.append(
            {
                "nodes": len(component),
                "edges": len(component_edges),
                "node_colors": _count_labels([colors[atom] for atom in component]),
                "edge_colors": _count_values(component_edges),
            }
        )

    return sorted(signatures, key=_sort_key)


def _route_cgr_bucket_fingerprint_from_prepared(
    prepared: _PreparedRouteCGR,
    *,
    schema: str = BUCKET_HASH_SCHEMA,
) -> dict[str, Any]:
    graph = prepared.graph
    return {
        "schema": schema,
        "algorithm": "stdlib-weisfeiler-lehman-1",
        "nodes": graph.number_of_nodes,
        "edges": graph.number_of_edges,
        "components": len(prepared.components),
        "color_history_hashes": list(prepared.history_hashes),
        "component_signatures": _component_signatures(
            graph, prepared.colors, list(prepared.components)
        ),
    }


def route_cgr_bucket_fingerprint(route_cgr: Any) -> dict[str, Any]:
    """Return the fast non-exact WL bucket payload for a RouteCGR.

    Equal bucket hashes are only a prefilter. They must be confirmed with
    ``route_cgr_hash`` when exact RouteCGR identity matters.
    """

    return _route_cgr_bucket_fingerprint_from_prepared(_prepare_route_cgr(route_cgr))


def route_cgr_bucket_hash(route_cgr: Any) -> str:
    """Return a fast non-exact hash used to bucket candidate-equal RouteCGRs."""

    return _stable_digest(route_cgr_bucket_fingerprint(route_cgr))


def _route_cgr_bucket_hash_from_prepared(
    prepared: _PreparedRouteCGR,
    *,
    schema: str = BUCKET_HASH_SCHEMA,
) -> str:
    return _stable_digest(
        _route_cgr_bucket_fingerprint_from_prepared(prepared, schema=schema)
    )


def _order_encoding_key(
    graph: RouteCGRGraph, order: tuple[int, ...]
) -> tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]]:
    positions = {atom_id: index for index, atom_id in enumerate(order)}
    edge_labels = []

    for (atom1, atom2), label in graph.edge_labels.items():
        if atom1 not in positions or atom2 not in positions:
            continue
        position1 = positions[atom1]
        position2 = positions[atom2]
        if position2 < position1:
            position1, position2 = position2, position1
        edge_labels.append((position1, position2, label))

    return (
        tuple(graph.node_labels[atom_id] for atom_id in order),
        tuple(sorted(edge_labels)),
    )


def _order_encoding_from_key(
    encoding_key: tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]],
) -> dict[str, Any]:
    node_labels, edge_labels = encoding_key
    return {
        "node_labels": list(node_labels),
        "edge_labels": [list(edge_label) for edge_label in edge_labels],
    }


def _component_canonical_encoding(
    graph: RouteCGRGraph,
    colors: Mapping[int, str],
    component: tuple[int, ...],
) -> dict[str, Any]:
    groups_by_color = defaultdict(list)
    for atom_id in component:
        groups_by_color[colors[atom_id]].append(atom_id)

    color_groups = [
        tuple(sorted(groups_by_color[color]))
        for color in sorted(groups_by_color, key=_sort_key)
    ]
    best_key = None

    for group_orders in product(*(permutations(group) for group in color_groups)):
        order = tuple(atom_id for group in group_orders for atom_id in group)
        encoding_key = _order_encoding_key(graph, order)
        if best_key is None or encoding_key < best_key:
            best_key = encoding_key

    return _order_encoding_from_key(best_key)


def _route_cgr_fingerprint_from_prepared(
    prepared: _PreparedRouteCGR,
    *,
    schema: str = HASH_SCHEMA,
) -> dict[str, Any]:
    graph = prepared.graph
    component_encodings = [
        _component_canonical_encoding(graph, prepared.colors, component)
        for component in prepared.components
    ]
    return {
        "schema": schema,
        "algorithm": "exact-wl-colored-canonical-adjacency-v1",
        "components": sorted(component_encodings, key=_sort_key),
    }


def route_cgr_fingerprint(route_cgr: Any) -> dict[str, Any]:
    """Return the exact canonical payload used by ``route_cgr_hash``.

    This exact layer first uses WL colors to split atoms into invariant
    buckets, then enumerates all atom orders inside each remaining color class
    to produce a canonical adjacency encoding. Equal ``route_cgr_hash`` values
    therefore mean exact equality under the labels documented by HASH_INCLUDES.

    Deliberately excluded Chython/CGR state is listed in HASH_EXCLUDES.
    """

    return _route_cgr_fingerprint_from_prepared(_prepare_route_cgr(route_cgr))


def route_cgr_hash(route_cgr: Any) -> str:
    """Return an exact RouteCGR hash that ignores atom-map numbering."""

    return _stable_digest(route_cgr_fingerprint(route_cgr))


def route_cgr_fingerprint_without_route_order(route_cgr: Any) -> dict[str, Any]:
    """Return exact RouteCGR fingerprint while ignoring route-order metadata."""

    return _route_cgr_fingerprint_from_prepared(
        _prepare_route_cgr(route_cgr, include_route_order=False),
        schema=ROUTE_ORDER_AGNOSTIC_HASH_SCHEMA,
    )


def route_cgr_hash_without_route_order(route_cgr: Any) -> str:
    """Return an exact RouteCGR hash with route-order metadata excluded."""

    return _stable_digest(route_cgr_fingerprint_without_route_order(route_cgr))


def _route_cgr_hash_from_prepared(prepared: _PreparedRouteCGR) -> str:
    return _stable_digest(_route_cgr_fingerprint_from_prepared(prepared))


def _route_order_agnostic_hash_from_prepared(prepared: _PreparedRouteCGR) -> str:
    return _stable_digest(
        _route_cgr_fingerprint_from_prepared(
            prepared,
            schema=ROUTE_ORDER_AGNOSTIC_HASH_SCHEMA,
        )
    )


def _numbered_fingerprint_from_graph(
    graph: RouteCGRGraph,
    *,
    schema: str = f"{HASH_SCHEMA}-numbered-cache-v1",
) -> dict[str, Any]:
    """Return an atom-numbered fingerprint used only for local hash caching.

    This is not a public identity because it depends on atom-map numbers.
    It is safe as a cache key: equal numbered fingerprints imply equal exact
    RouteCGR hashes, while unequal numbered fingerprints make no claim.
    """

    return {
        "schema": schema,
        "nodes": [
            [atom_id, graph.node_labels[atom_id]]
            for atom_id in sorted(graph.node_labels)
        ],
        "edges": [
            [atom1, atom2, label]
            for (atom1, atom2), label in sorted(graph.edge_labels.items())
        ],
    }


class _RouteCGRHashCache:
    """Per-call cache for mutable RouteCGR objects."""

    def __init__(self, *, include_route_order: bool = True) -> None:
        self.include_route_order = include_route_order
        self.hash_schema = (
            HASH_SCHEMA
            if include_route_order
            else ROUTE_ORDER_AGNOSTIC_HASH_SCHEMA
        )
        self.bucket_hash_schema = (
            BUCKET_HASH_SCHEMA
            if include_route_order
            else ROUTE_ORDER_AGNOSTIC_BUCKET_HASH_SCHEMA
        )
        self.numbered_hash_schema = f"{self.hash_schema}-numbered-cache-v1"
        self._prepared_by_object_id: dict[int, _PreparedRouteCGR] = {}
        self._numbered_hash_by_object_id: dict[int, str] = {}
        self._bucket_by_object_id: dict[int, str] = {}
        self._bucket_by_numbered_hash: dict[str, str] = {}
        self._exact_by_object_id: dict[int, str] = {}
        self._exact_by_numbered_hash: dict[str, str] = {}

    def _prepared(self, route_cgr: Any) -> _PreparedRouteCGR:
        object_id = id(route_cgr)
        if object_id not in self._prepared_by_object_id:
            self._prepared_by_object_id[object_id] = _prepare_route_cgr(
                route_cgr,
                include_route_order=self.include_route_order,
            )
        return self._prepared_by_object_id[object_id]

    def _numbered_hash(self, route_cgr: Any) -> str:
        object_id = id(route_cgr)
        if object_id not in self._numbered_hash_by_object_id:
            prepared = self._prepared(route_cgr)
            self._numbered_hash_by_object_id[object_id] = _stable_digest(
                _numbered_fingerprint_from_graph(
                    prepared.graph,
                    schema=self.numbered_hash_schema,
                )
            )
        return self._numbered_hash_by_object_id[object_id]

    def bucket(self, route_cgr: Any) -> _RouteCGRBucketResult:
        object_id = id(route_cgr)
        numbered_hash = self._numbered_hash(route_cgr)
        if object_id in self._bucket_by_object_id:
            return _RouteCGRBucketResult(
                bucket_hash=self._bucket_by_object_id[object_id],
                numbered_hash=numbered_hash,
            )

        if numbered_hash in self._bucket_by_numbered_hash:
            bucket_hash = self._bucket_by_numbered_hash[numbered_hash]
        else:
            prepared = self._prepared(route_cgr)
            bucket_hash = _route_cgr_bucket_hash_from_prepared(
                prepared,
                schema=self.bucket_hash_schema,
            )
            self._bucket_by_numbered_hash[numbered_hash] = bucket_hash

        self._bucket_by_object_id[object_id] = bucket_hash
        return _RouteCGRBucketResult(
            bucket_hash=bucket_hash,
            numbered_hash=numbered_hash,
        )

    def exact_hash(self, route_cgr: Any) -> str:
        object_id = id(route_cgr)
        if object_id in self._exact_by_object_id:
            return self._exact_by_object_id[object_id]

        numbered_hash = self._numbered_hash(route_cgr)
        if numbered_hash in self._exact_by_numbered_hash:
            exact_hash = self._exact_by_numbered_hash[numbered_hash]
        else:
            prepared = self._prepared(route_cgr)
            exact_hash = _stable_digest(
                _route_cgr_fingerprint_from_prepared(
                    prepared,
                    schema=self.hash_schema,
                )
            )
            self._exact_by_numbered_hash[numbered_hash] = exact_hash

        self._exact_by_object_id[object_id] = exact_hash
        return exact_hash

    def hashes(self, route_cgr: Any) -> _RouteCGRHashResult:
        return _RouteCGRHashResult(
            bucket_hash=self.bucket(route_cgr).bucket_hash,
            exact_hash=self.exact_hash(route_cgr),
        )


def route_cgrs_equal(left: Any, right: Any) -> bool:
    """Return True when two RouteCGRs are exactly equal under this hash contract."""

    left_prepared = _prepare_route_cgr(left)
    right_prepared = _prepare_route_cgr(right)
    if _route_cgr_bucket_hash_from_prepared(
        left_prepared
    ) != _route_cgr_bucket_hash_from_prepared(right_prepared):
        return False
    return (
        _route_cgr_hash_from_prepared(left_prepared)
        == _route_cgr_hash_from_prepared(right_prepared)
    )


def route_cgr_metadata(route_cgr: Any) -> dict[str, Any]:
    """Return human-readable metadata for auditing each generated hash."""

    transient_bonds = []
    bond_route_orders = []
    for atom1, atom2, bond in route_cgr.bonds():
        bond_info = {
            "atoms": sorted((atom1, atom2)),
            "order": bond.order,
            "p_order": bond.p_order,
            "route_orders": _route_orders(getattr(bond, "route_order", None)),
            "route_step_orders": _route_orders(
                getattr(bond, "route_step_order", None)
            ),
        }
        if bond.order is None and bond.p_order is None:
            transient_bonds.append(bond_info)
        if bond_info["route_orders"] or bond_info["route_step_orders"]:
            bond_route_orders.append(bond_info)

    atom_route_orders = [
        {
            "atom": atom_id,
            "route_orders": route_orders,
            "route_step_orders": _route_orders(
                getattr(atom, "route_step_order", None)
            ),
        }
        for atom_id, atom in route_cgr.atoms()
        if (route_orders := _route_orders(getattr(atom, "route_order", None)))
        or _route_orders(getattr(atom, "route_step_order", None))
    ]
    return {
        "transient_bonds": transient_bonds,
        "atom_route_orders": atom_route_orders,
        "bond_route_orders": bond_route_orders,
    }


def hash_route_cgrs(route_cgrs: Mapping[int, Any]) -> dict[str, Any]:
    """Hash one run and retain route IDs for duplicate RouteCGRs."""

    routes = []
    routes_by_hash = defaultdict(list)
    route_ids_by_bucket_hash = defaultdict(list)
    hash_cache = _RouteCGRHashCache()
    for route_id, route_cgr in sorted(route_cgrs.items()):
        hash_result = hash_cache.hashes(route_cgr)
        route_ids_by_bucket_hash[hash_result.bucket_hash].append(route_id)
        routes_by_hash[hash_result.exact_hash].append(route_id)
        routes.append(
            {
                "route_id": route_id,
                "bucket_hash": hash_result.bucket_hash,
                "hash": hash_result.exact_hash,
                **route_cgr_metadata(route_cgr),
            }
        )
    return {
        "hash_schema": HASH_SCHEMA,
        "bucket_hash_schema": BUCKET_HASH_SCHEMA,
        "hash_includes": HASH_INCLUDES,
        "hash_excludes": HASH_EXCLUDES,
        "route_count": len(routes),
        "unique_hash_count": len(routes_by_hash),
        "routes": routes,
        "route_ids_by_hash": dict(sorted(routes_by_hash.items())),
        "route_ids_by_bucket_hash": dict(sorted(route_ids_by_bucket_hash.items())),
    }


def route_order_variant_sets(route_cgrs: Mapping[int, Any]) -> list[list[list[int]]]:
    """Return route IDs that differ only by atom/bond route-order metadata.

    The function first groups by a cheap route-order-agnostic WL bucket, then
    exact-confirms only candidate buckets with more than one route. Full
    route-order-aware exact hashes are computed only for confirmed candidates.

    Each returned item is a partition of route IDs with the same RouteCGR after
    removing route-order metadata, split by their full route-order-aware identity.
    """

    agnostic_cache = _RouteCGRHashCache(include_route_order=False)
    full_cache = _RouteCGRHashCache()
    route_ids_by_agnostic_bucket = defaultdict(list)

    for route_id, route_cgr in sorted(route_cgrs.items()):
        bucket_hash = agnostic_cache.bucket(route_cgr).bucket_hash
        route_ids_by_agnostic_bucket[bucket_hash].append(route_id)

    variant_groups = []
    for bucket_route_ids in route_ids_by_agnostic_bucket.values():
        if len(bucket_route_ids) < 2:
            continue

        route_ids_by_agnostic_hash = defaultdict(list)
        for route_id in bucket_route_ids:
            agnostic_hash = agnostic_cache.exact_hash(route_cgrs[route_id])
            route_ids_by_agnostic_hash[agnostic_hash].append(route_id)

        for route_ids in route_ids_by_agnostic_hash.values():
            if len(route_ids) < 2:
                continue

            route_ids_by_full_hash = defaultdict(list)
            for route_id in route_ids:
                full_hash = full_cache.exact_hash(route_cgrs[route_id])
                route_ids_by_full_hash[full_hash].append(route_id)

            if len(route_ids_by_full_hash) <= 1:
                continue

            partition = [
                sorted(full_route_ids)
                for _, full_route_ids in sorted(route_ids_by_full_hash.items())
            ]
            partition.sort(key=lambda ids: (ids[0], ids))
            variant_groups.append(partition)

    return sorted(variant_groups, key=lambda group: (group[0][0], group))


def _route_ids_by_exact_hash(
    route_cgrs: Mapping[int, Any],
    hash_cache: _RouteCGRHashCache,
    route_ids: list[int] | None = None,
) -> dict[str, list[int]]:
    route_ids_by_hash = defaultdict(list)
    if route_ids is None:
        items = sorted(route_cgrs.items())
    else:
        items = [(route_id, route_cgrs[route_id]) for route_id in sorted(route_ids)]
    for route_id, route_cgr in items:
        route_ids_by_hash[hash_cache.exact_hash(route_cgr)].append(route_id)
    return dict(sorted(route_ids_by_hash.items()))


def _route_ids_by_bucket_hash(
    route_cgrs: Mapping[int, Any],
    hash_cache: _RouteCGRHashCache,
) -> dict[str, list[int]]:
    route_ids_by_bucket = defaultdict(list)
    for route_id, route_cgr in sorted(route_cgrs.items()):
        route_ids_by_bucket[hash_cache.bucket(route_cgr).bucket_hash].append(route_id)
    return dict(sorted(route_ids_by_bucket.items()))


def _prefixed_unique_route_ids(
    exact_hashes: Mapping[str, list[int]],
    bucket_hashes: Mapping[str, list[int]],
) -> dict[str, list[int]]:
    unique_route_ids = {
        f"exact:{route_hash}": route_ids
        for route_hash, route_ids in sorted(exact_hashes.items())
    }
    unique_route_ids.update(
        {
            f"bucket:{bucket_hash}": route_ids
            for bucket_hash, route_ids in sorted(bucket_hashes.items())
        }
    )
    return unique_route_ids


def compare_route_cgr_dicts(
    route_cgr_dict_1: Mapping[int, Any],
    route_cgr_dict_2: Mapping[int, Any],
) -> dict[str, Any]:
    """Compare two RouteCGR dictionaries through bucket-first exact matching.

    The input dictionaries may use different route-id namespaces. Results are
    grouped by fast WL bucket first. Exact hashes are computed only for buckets
    appearing in both inputs. Bucket-only routes are therefore returned as
    bucket-hash identities; routes in shared buckets are exact-hash confirmed.
    """

    hash_cache = _RouteCGRHashCache()
    route_ids_by_bucket_1 = _route_ids_by_bucket_hash(route_cgr_dict_1, hash_cache)
    route_ids_by_bucket_2 = _route_ids_by_bucket_hash(route_cgr_dict_2, hash_cache)
    buckets_1 = set(route_ids_by_bucket_1)
    buckets_2 = set(route_ids_by_bucket_2)
    overlap_buckets = sorted(buckets_1 & buckets_2)
    unique_buckets_1 = sorted(buckets_1 - buckets_2)
    unique_buckets_2 = sorted(buckets_2 - buckets_1)

    overlap_bucket_route_ids_1 = [
        route_id
        for bucket_hash in overlap_buckets
        for route_id in route_ids_by_bucket_1[bucket_hash]
    ]
    overlap_bucket_route_ids_2 = [
        route_id
        for bucket_hash in overlap_buckets
        for route_id in route_ids_by_bucket_2[bucket_hash]
    ]
    route_ids_by_hash_1 = _route_ids_by_exact_hash(
        route_cgr_dict_1, hash_cache, overlap_bucket_route_ids_1
    )
    route_ids_by_hash_2 = _route_ids_by_exact_hash(
        route_cgr_dict_2, hash_cache, overlap_bucket_route_ids_2
    )
    hashes_1 = set(route_ids_by_hash_1)
    hashes_2 = set(route_ids_by_hash_2)
    overlap_hashes = sorted(hashes_1 & hashes_2)
    unique_hashes_1 = sorted(hashes_1 - hashes_2)
    unique_hashes_2 = sorted(hashes_2 - hashes_1)
    route_ids_unique_1_by_exact_hash = {
        route_hash: route_ids_by_hash_1[route_hash]
        for route_hash in unique_hashes_1
    }
    route_ids_unique_2_by_exact_hash = {
        route_hash: route_ids_by_hash_2[route_hash]
        for route_hash in unique_hashes_2
    }
    route_ids_unique_1_by_bucket_hash = {
        bucket_hash: route_ids_by_bucket_1[bucket_hash]
        for bucket_hash in unique_buckets_1
    }
    route_ids_unique_2_by_bucket_hash = {
        bucket_hash: route_ids_by_bucket_2[bucket_hash]
        for bucket_hash in unique_buckets_2
    }

    return {
        "hash_schema": HASH_SCHEMA,
        "bucket_hash_schema": BUCKET_HASH_SCHEMA,
        "hash_includes": HASH_INCLUDES,
        "hash_excludes": HASH_EXCLUDES,
        "comparison": "bucket-first-exact-overlap-v1",
        "route_count_1": len(route_cgr_dict_1),
        "route_count_2": len(route_cgr_dict_2),
        "bucket_count_1": len(buckets_1),
        "bucket_count_2": len(buckets_2),
        "overlap_bucket_count": len(overlap_buckets),
        "bucket_only_count_1": len(unique_buckets_1),
        "bucket_only_count_2": len(unique_buckets_2),
        "overlap_exact_count_1": len(hashes_1),
        "overlap_exact_count_2": len(hashes_2),
        "overlap_exact_count": len(overlap_hashes),
        "route_ids_overlap": {
            route_hash: {
                "route_cgr_dict_1": route_ids_by_hash_1[route_hash],
                "route_cgr_dict_2": route_ids_by_hash_2[route_hash],
            }
            for route_hash in overlap_hashes
        },
        "route_ids_unique_1": _prefixed_unique_route_ids(
            route_ids_unique_1_by_exact_hash,
            route_ids_unique_1_by_bucket_hash,
        ),
        "route_ids_unique_2": _prefixed_unique_route_ids(
            route_ids_unique_2_by_exact_hash,
            route_ids_unique_2_by_bucket_hash,
        ),
        "route_ids_unique_1_by_exact_hash": route_ids_unique_1_by_exact_hash,
        "route_ids_unique_2_by_exact_hash": route_ids_unique_2_by_exact_hash,
        "route_ids_unique_1_by_bucket_hash": route_ids_unique_1_by_bucket_hash,
        "route_ids_unique_2_by_bucket_hash": route_ids_unique_2_by_bucket_hash,
    }
