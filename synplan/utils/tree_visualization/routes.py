"""Route metadata helpers shared by tree visualization renderers."""

from __future__ import annotations

import itertools
from collections.abc import Iterable

from synplan.mcts.tree import Tree
from synplan.utils.tree_visualization.molecules import (
    node_primary_molecule,
    svg_from_smiles,
)


def route_nodes_by_route(
    tree: Tree, route_ids: Iterable[int] | None = None
) -> dict[str, list[int]]:
    if route_ids is None:
        route_ids_set = set(tree.winning_nodes)
    else:
        route_ids_set = {int(route_id) for route_id in route_ids}

    route_nodes: dict[str, list[int]] = {}
    for route_id in sorted(route_ids_set):
        if route_id not in tree.nodes:
            continue
        nodes: list[int] = []
        current = route_id
        seen: set[int] = set()
        while current and current not in seen:
            seen.add(current)
            nodes.append(current)
            current = tree.parents.get(current)
        route_nodes[str(route_id)] = nodes
    return route_nodes


def is_building_block(precursor, tree: Tree) -> bool:
    if precursor is None:
        return False
    try:
        return precursor.is_building_block(
            tree.building_blocks, getattr(tree.config, "min_mol_size", 0)
        )
    except Exception:
        try:
            return str(precursor) in tree.building_blocks
        except Exception:
            return False


def route_extras_by_route(
    tree: Tree, route_ids: Iterable[int]
) -> dict[str, dict[str, object]]:
    extras: dict[str, dict[str, object]] = {}
    for route_id in sorted(route_ids):
        if route_id not in tree.nodes:
            continue

        path_ids: list[int] = []
        current = route_id
        seen: set[int] = set()
        while current and current not in seen:
            seen.add(current)
            path_ids.append(current)
            current = tree.parents.get(current)
        path_ids = list(reversed(path_ids))
        if len(path_ids) < 2:
            continue

        smiles_to_node: dict[str, int] = {}
        base_smiles: set[str] = set()
        for node_id in path_ids:
            node = tree.nodes.get(node_id)
            molecule = node_primary_molecule(node)
            if molecule is None:
                continue
            try:
                smiles = str(molecule)
            except Exception:
                continue
            if smiles:
                base_smiles.add(smiles)
                if smiles not in smiles_to_node:
                    smiles_to_node[smiles] = node_id

        by_parent: dict[str, list[dict[str, str]]] = {}
        route_seen_smiles: set[str] = set()
        for before_id, after_id in itertools.pairwise(path_ids):
            before = tree.nodes.get(before_id)
            after = tree.nodes.get(after_id)
            if before is None or after is None:
                continue

            parent_precursor = getattr(before, "curr_precursor", None)
            if parent_precursor is None:
                continue
            try:
                parent_smiles = str(parent_precursor)
            except Exception:
                continue
            if not parent_smiles:
                continue

            extra_items: list[dict[str, str]] = []
            seen_smiles: set[str] = set()
            for precursor in getattr(after, "new_precursors", ()) or ():
                try:
                    child_smiles = str(precursor)
                except Exception:
                    continue
                if (
                    not child_smiles
                    or child_smiles in smiles_to_node
                    or child_smiles in base_smiles
                    or child_smiles in route_seen_smiles
                ):
                    continue
                if child_smiles in seen_smiles:
                    continue

                seen_smiles.add(child_smiles)
                route_seen_smiles.add(child_smiles)
                status = (
                    "starting material"
                    if is_building_block(precursor, tree)
                    else "intermediate"
                )
                extra_items.append(
                    {
                        "smiles": child_smiles,
                        "status": status,
                        "svg": svg_from_smiles(child_smiles),
                    }
                )
            if extra_items:
                by_parent[str(before_id)] = extra_items
        if by_parent:
            extras[str(route_id)] = {"by_parent": by_parent}
    return extras
