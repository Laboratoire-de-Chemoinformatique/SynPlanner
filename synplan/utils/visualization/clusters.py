"""Shared helpers for Strategic Bond Search cluster payloads."""

from __future__ import annotations

from typing import Iterable, Optional


def normalize_strat_bonds(
    strat_bonds: Optional[Iterable[Iterable[int]]],
) -> list[list[int]]:
    if not strat_bonds:
        return []

    normalized: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for bond in strat_bonds:
        if not bond or len(bond) < 2:
            continue
        try:
            a, b = bond
        except (TypeError, ValueError):
            continue
        try:
            a_int = int(a)
            b_int = int(b)
        except (TypeError, ValueError):
            continue

        pair = tuple(sorted((a_int, b_int)))
        if pair in seen:
            continue
        seen.add(pair)
        normalized.append([pair[0], pair[1]])

    normalized.sort()
    return normalized


def build_cluster_payload(clusters: dict[str, dict]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for cluster_id, data in clusters.items():
        if not isinstance(data, dict):
            continue

        route_ids: list[int] = []
        for route_id in data.get("route_ids") or []:
            try:
                route_ids.append(int(route_id))
            except (TypeError, ValueError):
                continue

        payload.append(
            {
                "id": str(cluster_id),
                "bonds": normalize_strat_bonds(data.get("strat_bonds")),
                "route_ids": sorted(set(route_ids)),
            }
        )
    return payload
