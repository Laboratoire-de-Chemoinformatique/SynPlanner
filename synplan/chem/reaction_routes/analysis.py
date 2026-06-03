"""Analysis helpers for RouteCGR comparison and route-derived BB usage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def flatten_route_id_groups(route_id_groups: Mapping[Any, Iterable[int]]) -> list[int]:
    """Flatten a mapping of identity keys to route-id iterables."""

    return sorted(
        route_id
        for route_ids in route_id_groups.values()
        for route_id in route_ids
    )


def route_cgr_subset(route_cgrs: Mapping[int, Any], route_ids: Iterable[int]) -> dict:
    """Return a `{route_id: RouteCGR}` subset for the requested route IDs."""

    return {route_id: route_cgrs[route_id] for route_id in route_ids}


def route_cgr_overlap_rows(comparison_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one named row per exact RouteCGR overlap identity.

    `compare_route_cgr_dicts` stores overlap route IDs grouped by exact RouteCGR
    hash. This helper flattens that structure for notebook inspection while
    keeping both the representative route IDs and the full duplicate ID lists.
    """

    rows = []
    for exact_hash, route_ids in comparison_result.get("route_ids_overlap", {}).items():
        route_ids_1 = list(route_ids.get("route_cgr_dict_1", []))
        route_ids_2 = list(route_ids.get("route_cgr_dict_2", []))
        if not route_ids_1 or not route_ids_2:
            continue

        rows.append(
            {
                "exact_hash": exact_hash,
                "route_id_1": route_ids_1[0],
                "route_id_2": route_ids_2[0],
                "route_ids_1": route_ids_1,
                "route_ids_2": route_ids_2,
            }
        )

    return rows


def sb_cgr_identity_to_cluster_id(clusters: Mapping[Any, Mapping[str, Any]]) -> dict:
    """Map SB-CGR string identity to its cluster ID."""

    return {
        str(cluster["sb_cgr"]): cluster_id
        for cluster_id, cluster in clusters.items()
        if cluster.get("sb_cgr") is not None
    }


def compare_sb_cgr_clusters(
    clusters_1: Mapping[Any, Mapping[str, Any]],
    clusters_2: Mapping[Any, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two cluster dictionaries by representative SB-CGR identity."""

    sb_to_cluster_1 = sb_cgr_identity_to_cluster_id(clusters_1)
    sb_to_cluster_2 = sb_cgr_identity_to_cluster_id(clusters_2)
    identities_1 = set(sb_to_cluster_1)
    identities_2 = set(sb_to_cluster_2)
    unique_1 = sorted(identities_1 - identities_2)
    overlap = sorted(identities_1 & identities_2)
    unique_2 = sorted(identities_2 - identities_1)

    return {
        "unique_cluster_ids_1": [sb_to_cluster_1[key] for key in unique_1],
        "overlap_cluster_ids": [
            (sb_to_cluster_1[key], sb_to_cluster_2[key]) for key in overlap
        ],
        "unique_cluster_ids_2": [sb_to_cluster_2[key] for key in unique_2],
        "sb_keys_1": identities_1,
        "sb_keys_2": identities_2,
        "overlap_sb_keys": set(overlap),
    }


def _target_atoms_from_pseudo_products(pseudo_products: Any) -> set[int]:
    products = pseudo_products.split()
    if not products:
        return set()
    target_product = min(products, key=lambda mol: min(mol._atoms))
    return set(target_product._atoms)


def route_cgr_pseudo_reactants_by_role(route_cgr: Any) -> dict[str, list[Any]]:
    """Split decomposed RouteCGR pseudo-reactants into real and supporting parts.

    Real pseudo-reactants contain atoms that survive into the target product
    projection. Supporting pseudo-reactants have no target-product atom overlap
    and usually correspond to PG/FGI/support fragments.
    """

    pseudo_reactants, pseudo_products = route_cgr.decompose()
    target_atoms = _target_atoms_from_pseudo_products(pseudo_products)

    result = {"real_bb": [], "supporting": []}
    if not target_atoms:
        return result

    for mol in pseudo_reactants.split():
        kind = "real_bb" if set(mol._atoms) & target_atoms else "supporting"
        result[kind].append(mol)

    return result
