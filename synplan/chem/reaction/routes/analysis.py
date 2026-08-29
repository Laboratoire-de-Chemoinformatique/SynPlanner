"""Analysis helpers for RouteCGR comparison and route-derived BB usage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, TypeAlias

from chython import smiles as chython_smiles

from synplan.chem.reaction.routes.representation.components import (
    route_cgr_pseudo_reactants_by_role,
)
from synplan.chem.reaction.routes.representation.container import unwrap_cgr

RouteId: TypeAlias = Hashable

__all__ = [
    "collect_bb_usage_stats",
    "compare_sb_cgr_clusters",
    "flatten_route_id_groups",
    "route_cgr_overlap_rows",
    "route_cgr_pseudo_reactants_by_role",
    "route_cgr_subset",
    "route_ids_with_exact_bb",
    "sb_cgr_identity_to_cluster_id",
]


def _cluster_sb_cgr(cluster: Any) -> Any:
    if isinstance(cluster, Mapping):
        return cluster.get("sb_cgr")
    return getattr(cluster, "sb_cgr", None)


def flatten_route_id_groups(route_id_groups: Mapping[Any, Iterable[int]]) -> list[int]:
    """Flatten a mapping of identity keys to route-id iterables."""

    return sorted(
        route_id for route_ids in route_id_groups.values() for route_id in route_ids
    )


def route_cgr_subset(route_cgrs: Mapping[int, Any], route_ids: Iterable[int]) -> dict:
    """Return a `{route_id: RouteCGR}` subset for the requested route IDs."""

    return {route_id: route_cgrs[route_id] for route_id in route_ids}


def route_cgr_overlap_rows(
    comparison_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
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
        str(sb_cgr): cluster_id
        for cluster_id, cluster in clusters.items()
        if (sb_cgr := _cluster_sb_cgr(cluster)) is not None
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


def route_ids_with_exact_bb(
    bb_smi: str,
    all_route_cgrs: Mapping[int, Any],
    kind: str = "any",
) -> list[int]:
    """Return route IDs where an exact pseudo-reactant BB identity is present.

    Parameters
    ----------
    bb_smi
        Building-block SMILES. It is parsed by Chython and compared by
        Chython's canonical string representation.
    all_route_cgrs
        Mapping of route ID to RouteCGR.
    kind
        `"any"`, `"real"`, or `"supporting"`. Real BBs have at least one atom
        that survives into the target product projection. Supporting BBs do not.
    """

    if kind not in {"any", "real", "supporting"}:
        raise ValueError("kind must be one of: 'any', 'real', 'supporting'")

    bb_key = str(chython_smiles(bb_smi)) if isinstance(bb_smi, str) else str(bb_smi)
    hits = []

    for route_id, route_cgr_value in all_route_cgrs.items():
        route_cgr = unwrap_cgr(route_cgr_value)
        if route_cgr is None:
            continue

        roles = route_cgr_pseudo_reactants_by_role(route_cgr)
        if kind == "real":
            candidates = roles["real_bb"]
        elif kind == "supporting":
            candidates = roles["supporting"]
        else:
            candidates = roles["real_bb"] + roles["supporting"]

        for mol in candidates:
            if str(mol) != bb_key:
                continue

            hits.append(route_id)
            break

    return sorted(hits)


def collect_bb_usage_stats(all_route_cgrs: Mapping[int, Any]) -> dict[str, Any]:
    """Collect real/supporting pseudo-reactant usage from RouteCGRs.

    A real BB has at least one atom that survives into the target product
    projection of the decomposed RouteCGR. A supporting BB has no target-product
    atoms and is usually a protecting group, functional-group interconversion
    reagent, salt, or other route support fragment.
    """

    stats = {
        "real_bb": defaultdict(lambda: {"occurrences": 0, "route_ids": set()}),
        "supporting": defaultdict(lambda: {"occurrences": 0, "route_ids": set()}),
        "by_route": {},
    }

    for route_id, route_cgr_value in all_route_cgrs.items():
        route_cgr = unwrap_cgr(route_cgr_value)
        if route_cgr is None:
            continue

        roles = route_cgr_pseudo_reactants_by_role(route_cgr)
        if not roles["real_bb"] and not roles["supporting"]:
            continue

        route_real = []
        route_supporting = []
        for kind, mols in roles.items():
            for mol in mols:
                key = str(mol)
                stats[kind][key]["occurrences"] += 1
                stats[kind][key]["route_ids"].add(route_id)
                (route_real if kind == "real_bb" else route_supporting).append(key)

        stats["by_route"][route_id] = {
            "real_bb": sorted(route_real),
            "supporting": sorted(route_supporting),
        }

    for kind in ("real_bb", "supporting"):
        stats[kind] = {
            smi: {
                "occurrences": data["occurrences"],
                "route_count": len(data["route_ids"]),
                "route_ids": sorted(data["route_ids"]),
            }
            for smi, data in sorted(
                stats[kind].items(),
                key=lambda item: (-len(item[1]["route_ids"]), item[0]),
            )
        }

    return stats
