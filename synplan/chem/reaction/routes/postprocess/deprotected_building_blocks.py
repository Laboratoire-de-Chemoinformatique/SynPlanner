"""Expand deprotected stock leaves back to their purchasable structures."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from chython import smiles as read_smiles
from chython.containers import MoleculeContainer, ReactionContainer

from synplan.chem.building_blocks.catalog import BuildingBlockCatalog
from synplan.chem.building_blocks.deprotection import deprotect_molecule
from synplan.chem.building_blocks.provenance import validate_deprotection_provenance
from synplan.chem.reaction.routes.tree_ops import (
    iter_molecule_leaves,
    iter_route_nodes,
    max_route_atom_map,
    node_at,
    reindex_reaction_steps,
)


class RouteExpansionLimitError(ValueError):
    """Raised before expansion would exceed the configured route-variant cap."""

    def __init__(self, *, required: int, limit: int) -> None:
        self.required = required
        self.limit = limit
        super().__init__(
            f"deprotected BB expansion requires {required} variants; limit is {limit}"
        )


def _molecule(smiles: str, *, context: str) -> MoleculeContainer:
    try:
        molecule = read_smiles(smiles)
    except Exception as error:
        raise ValueError(f"Invalid SMILES for {context}: {smiles!r}") from error
    if not isinstance(molecule, MoleculeContainer):
        raise ValueError(f"Expected a molecule for {context}, got {smiles!r}")
    return molecule


def _validate_route(route: Mapping[str, Any]) -> None:
    if not isinstance(route, Mapping) or route.get("type") != "mol":
        raise ValueError("route must be a mapping containing a molecule root")
    for _path, node in iter_route_nodes(route):
        node_type = node.get("type")
        if node_type not in {"mol", "reaction"}:
            raise ValueError(f"Unsupported route node type: {node_type!r}")
        if not isinstance(node.get("smiles"), str):
            raise ValueError("Every route node must contain a string 'smiles'")


def _terminal_matches(
    route: Mapping[str, Any], catalog: BuildingBlockCatalog
) -> list[tuple[tuple[int, ...], list[Mapping[str, object]]]]:
    matches: list[tuple[tuple[int, ...], list[Mapping[str, object]]]] = []

    for path, node in iter_molecule_leaves(route):
        if node.get("in_stock") is not True:
            continue
        molecule = _molecule(node["smiles"], context=f"route leaf at {path}")
        records = (node.get("bb") or {}).get("records", [])
        alternatives = list(
            catalog.protected_alternative_records(
                str(molecule), provenance_records=records
            )
        )
        if alternatives:
            matches.append((path, alternatives))
    return matches


def _mapped_consumed_leaf(
    route: Mapping[str, Any], path: tuple[int, ...], leaf: Mapping[str, Any]
) -> MoleculeContainer:
    if not path:
        # A stock-only route has no consuming reaction and therefore no existing
        # atom-map contract to preserve.
        molecule = _molecule(leaf["smiles"], context="stock-only route root")
        molecule.remap({number: number for number in molecule._atoms})
        return molecule

    parent = node_at(route, path[:-1])
    if parent.get("type") != "reaction":
        raise ValueError("A terminal building block must be a child of a reaction")

    reaction = read_smiles(parent["smiles"])
    plain_leaf = _molecule(leaf["smiles"], context=f"route leaf at {path}")
    for reactant in reaction.reactants:
        mapping = next(iter(plain_leaf.get_mapping(reactant)), None)
        if mapping is not None and len(mapping) == len(plain_leaf._atoms) == len(
            reactant._atoms
        ):
            return reactant.copy()
    raise ValueError(
        f"Could not reconcile leaf {leaf['smiles']!r} with its consuming reaction"
    )


def _deprotection_reaction(
    record: Mapping[str, object],
    mapped_product: MoleculeContainer,
    next_atom_map: int,
) -> tuple[str, int, str]:
    protected_smiles = str(record["input_smiles"])
    mapped_deprotection = str(record.get("mapped_deprotection") or "")
    if mapped_deprotection:
        reaction = validate_deprotection_provenance(
            record,
            context=f"protected building block {protected_smiles!r}",
            required=True,
        )
        if reaction is None:
            raise ValueError("Exact deprotection provenance has no transformation")
        protected = reaction.reactants[0].copy()
        deprotected = reaction.products[0]
        replay_mode = "exact"
    else:
        protected = _molecule(protected_smiles, context="protected building block")
        policy = str(record.get("deprotection_policy") or "aggressive")
        if policy not in {"conservative", "aggressive"}:
            raise ValueError(f"Invalid recorded deprotection policy: {policy!r}")
        deprotected = deprotect_molecule(protected, policy=policy)
        replay_mode = "legacy_inference"

    mapping = next(iter(deprotected.get_mapping(mapped_product)), None)
    if (
        mapping is None
        or len(mapping) != len(deprotected._atoms)
        or len(mapping) != len(mapped_product._atoms)
    ):
        raise ValueError(
            f"Recorded deprotection for {protected_smiles!r} "
            "does not produce the route leaf"
        )

    remapping: dict[int, int] = dict(mapping)
    next_atom_map = max(next_atom_map, max(mapped_product._atoms, default=0) + 1)
    for atom_number in protected._atoms:
        if atom_number not in remapping:
            remapping[atom_number] = next_atom_map
            next_atom_map += 1
    protected.remap(remapping)
    reaction = ReactionContainer(
        reactants=[protected], products=[mapped_product.copy()]
    )
    return format(reaction, "m"), next_atom_map, replay_mode


def _insert_deprotection(
    route: dict[str, Any],
    path: tuple[int, ...],
    record: Mapping[str, object],
    next_atom_map: int,
) -> int:
    leaf = node_at(route, path)
    if not isinstance(leaf, dict):
        raise ValueError("route molecule nodes must be mutable dictionaries")
    protected_smiles = str(record["input_smiles"])
    mapped_product = _mapped_consumed_leaf(route, path, leaf)
    reaction_smiles, next_atom_map, replay_mode = _deprotection_reaction(
        record, mapped_product, next_atom_map
    )
    source_index = record.get("source_index")
    source_records = [
        dict(candidate)
        for candidate in (leaf.get("bb") or {}).get("records", [])
        if candidate.get("input_smiles") == protected_smiles
        and (source_index is None or candidate.get("source_index") == source_index)
    ]
    if not source_records:
        source_records = [dict(record)]
    provenance_meta: dict[str, object] = {"preprocessing_provenance": replay_mode}
    for key in (
        "source_index",
        "deprotection_policy",
        "protective_rules_sha256",
        "deprotection_events",
    ):
        if value := record.get(key):
            provenance_meta[key] = value
    leaf["in_stock"] = False
    leaf["children"] = [
        {
            "type": "reaction",
            "smiles": reaction_smiles,
            "children": [
                {
                    "type": "mol",
                    "smiles": protected_smiles,
                    "in_stock": True,
                    "bb": {"records": source_records},
                }
            ],
            "meta": {
                "reaction_class": "deprotection",
                "source": "building_blocks_identity",
                **provenance_meta,
            },
        }
    ]
    return next_atom_map


def expand_deprotected_building_blocks(
    route: Mapping[str, Any],
    catalog: BuildingBlockCatalog | str | Path,
    *,
    max_variants: int | None = None,
) -> list[dict[str, Any]]:
    """Return route variants with real protected building blocks as leaves.

    Eligible leaves are terminal, explicitly in-stock molecules whose canonical
    structure occurs on a ``deprotected`` row of the identity TSV.  Multiple
    matches are expanded as a Cartesian product.  The caller's route is never
    mutated.
    """
    _validate_route(route)
    if max_variants is not None and max_variants < 1:
        raise ValueError("max_variants must be at least 1")
    resolved_catalog = (
        catalog
        if isinstance(catalog, BuildingBlockCatalog)
        else BuildingBlockCatalog.from_files(catalog)
    )
    matches = _terminal_matches(route, resolved_catalog)
    if not matches:
        return [deepcopy(dict(route))]

    variant_count = math.prod(len(alternatives) for _, alternatives in matches)
    if max_variants is not None and variant_count > max_variants:
        raise RouteExpansionLimitError(required=variant_count, limit=max_variants)

    first_new_atom_map = max_route_atom_map(route) + 1
    variants: list[dict[str, Any]] = []
    for choices in product(*(alternatives for _, alternatives in matches)):
        variant = deepcopy(dict(route))
        next_atom_map = first_new_atom_map
        for (path, _), record in zip(matches, choices, strict=True):
            next_atom_map = _insert_deprotection(variant, path, record, next_atom_map)
        reindex_reaction_steps(variant)
        variants.append(variant)
    return variants


__all__ = ["RouteExpansionLimitError", "expand_deprotected_building_blocks"]
