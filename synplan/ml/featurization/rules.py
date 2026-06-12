"""Query-CGR graph tensorization for MHN rule embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import torch
from chython import smarts
from chython.containers import QueryCGRContainer, ReactionContainer
from torch_geometric.data import Data

from synplan.chem.reaction.rules.representation.config import (
    RULE_GRAPH_CHARGE_OFFSET,
    RULE_GRAPH_COUNT_LABELS,
    RULE_GRAPH_EDGE_FEATURE_DIM,
    RULE_GRAPH_HYBRIDIZATIONS,
    RULE_GRAPH_NODE_FEATURE_DIM,
    RULE_GRAPH_ORDER_LABELS,
    RULE_GRAPH_RING_SIZE_LABELS,
    RULE_GRAPH_SCHEMA_VERSION,
    RULE_GRAPH_SIDES,
)
from synplan.chem.reaction.rules.representation.morgan import (
    query_reaction_atom_labels,
)
from synplan.chem.reaction.rules.representation.query_cgr import (
    compress_labels,
    query_cgr_atom_label,
    query_cgr_bond_label,
)

# Local aliases; canonical definitions live in chem…representation.config.
_SIDES = RULE_GRAPH_SIDES
_HYBRIDIZATIONS = RULE_GRAPH_HYBRIDIZATIONS
_COUNT_LABELS = RULE_GRAPH_COUNT_LABELS
_RING_SIZE_LABELS = RULE_GRAPH_RING_SIZE_LABELS
_ORDER_LABELS = RULE_GRAPH_ORDER_LABELS
_CHARGE_OFFSET = RULE_GRAPH_CHARGE_OFFSET


def _numbers(value: Any) -> tuple[float, ...]:
    if value is None or isinstance(value, str):
        return ()
    if isinstance(value, bool):
        return (float(value),)
    if isinstance(value, Integral | Real):
        return (float(value),)
    if isinstance(value, Mapping):
        collected = []
        for item in value.values():
            collected.extend(_numbers(item))
        return tuple(collected)
    if isinstance(value, Sequence):
        collected = []
        for item in value:
            collected.extend(_numbers(item))
        return tuple(collected)
    return ()


def _first_number(value: Any, default: float = 0.0) -> float:
    values = _numbers(value)
    if values:
        return values[0]
    return default


def _charge_feature(value: Any, *, present: bool = True) -> float:
    if not present:
        return 0.0
    return max(0.0, _first_number(value) + _CHARGE_OFFSET)


def _membership(value: Any, allowed: Sequence[int]) -> list[float]:
    values = set(_numbers(value))
    return [float(float(item) in values) for item in allowed]


def _numeric_set_features(value: Any, allowed: Sequence[int]) -> list[float]:
    values = tuple(sorted(set(_numbers(value))))
    allowed_values = {float(item) for item in allowed}
    overflow = tuple(item for item in values if item not in allowed_values)
    return [
        float(len(values)),
        *_membership(values, allowed),
        float(len(overflow)),
        min(overflow) if overflow else 0.0,
        max(overflow) if overflow else 0.0,
    ]


def _side_maps(atom_labels: Any) -> dict[str, list[dict[str, Any]]]:
    side_maps: dict[str, list[dict[str, Any]]] = {side: [] for side in _SIDES}
    for side, labels in atom_labels or ():
        if side in side_maps:
            side_maps[side].append(dict(labels))
    return side_maps


def _side_values(entries: Sequence[dict[str, Any]], field: str) -> tuple[float, ...]:
    values = []
    for entry in entries:
        values.extend(_numbers(entry.get(field)))
    return tuple(values)


def _side_features(entries: Sequence[dict[str, Any]]) -> list[float]:
    present = bool(entries)
    atomic_numbers = _side_values(entries, "atomic_number")
    features = [
        float(present),
        atomic_numbers[0] if atomic_numbers else 0.0,
        float(present and not atomic_numbers),
        _charge_feature(_side_values(entries, "charge"), present=present),
        float(any(_side_values(entries, "is_radical"))),
    ]
    features.extend(
        _numeric_set_features(_side_values(entries, "neighbors"), _COUNT_LABELS)
    )
    features.extend(
        _numeric_set_features(_side_values(entries, "heteroatoms"), _COUNT_LABELS)
    )
    features.extend(
        _membership(_side_values(entries, "hybridization"), _HYBRIDIZATIONS)
    )
    features.extend(
        _numeric_set_features(
            _side_values(entries, "implicit_hydrogens"), _COUNT_LABELS
        )
    )
    features.extend(
        _numeric_set_features(_side_values(entries, "ring_sizes"), _RING_SIZE_LABELS)
    )
    return features


def _node_order_labels(
    query_cgr: QueryCGRContainer, atom_labels: Mapping[int, Any]
) -> dict[int, tuple]:
    return {
        atom: (query_cgr_atom_label(query_cgr, atom), atom_labels.get(atom))
        for atom in query_cgr._atoms
    }


def _canonical_atom_order(
    query_cgr: QueryCGRContainer, atom_labels: Mapping[int, Any]
) -> tuple[int, ...]:
    labels = _node_order_labels(query_cgr, atom_labels)
    colors = compress_labels(labels)
    atoms = tuple(query_cgr._atoms)

    for _ in range(len(atoms)):
        signatures = {}
        for atom in atoms:
            neighborhood = tuple(
                sorted(
                    (
                        (
                            query_cgr_bond_label(query_cgr, atom, neighbor),
                            colors[neighbor],
                        )
                        for neighbor in query_cgr._bonds[atom]
                    ),
                    key=repr,
                )
            )
            signatures[atom] = (colors[atom], neighborhood)
        refined = compress_labels(signatures)
        if refined == colors:
            break
        colors = refined

    return tuple(
        atom
        for color in sorted(set(colors.values()))
        for atom in sorted(
            (item for item in atoms if colors[item] == color),
            key=lambda item: (repr(labels[item]), len(query_cgr._bonds[item]), item),
        )
    )


def _node_features(
    query_cgr: QueryCGRContainer,
    atom: int,
    atom_labels: Mapping[int, Any],
) -> list[float]:
    label = query_cgr_atom_label(query_cgr, atom)
    features = [
        _first_number(label[0]),
        _first_number(label[2]),
        _charge_feature(label[3]),
        _charge_feature(label[4]),
        float(label[5]),
        float(label[6]),
        float(len(query_cgr._bonds[atom])),
    ]
    features.extend(_numeric_set_features(label[7], _COUNT_LABELS))
    features.extend(_numeric_set_features(label[8], _COUNT_LABELS))
    features.extend(_membership(label[9], _HYBRIDIZATIONS))
    features.extend(_membership(label[10], _HYBRIDIZATIONS))

    side_maps = _side_maps(atom_labels.get(atom))
    for side in _SIDES:
        features.extend(_side_features(side_maps[side]))

    if len(features) != RULE_GRAPH_NODE_FEATURE_DIM:
        raise ValueError(
            f"Expected {RULE_GRAPH_NODE_FEATURE_DIM} node features, got {len(features)}"
        )
    return features


def _order_number(order: Any) -> float:
    if order is None:
        return 0.0
    return _first_number(order)


def _order_features(order: Any) -> list[float]:
    return [float(order == label) for label in _ORDER_LABELS]


def _edge_features(
    query_cgr: QueryCGRContainer, atom_1: int, atom_2: int
) -> list[float]:
    order, product_order = query_cgr_bond_label(query_cgr, atom_1, atom_2)
    has_order = order is not None
    has_product_order = product_order is not None
    changed = order != product_order
    features = []
    features.extend(_order_features(order))
    features.extend(_order_features(product_order))
    features.extend(
        [
            float(changed),
            float(not has_order and has_product_order),
            float(has_order and not has_product_order),
            float(has_order and has_product_order and changed),
            _order_number(order),
            _order_number(product_order),
        ]
    )
    if len(features) != RULE_GRAPH_EDGE_FEATURE_DIM:
        raise ValueError(
            f"Expected {RULE_GRAPH_EDGE_FEATURE_DIM} edge features, got {len(features)}"
        )
    return features


def query_cgr_to_pyg(
    query_cgr: QueryCGRContainer,
    *,
    atom_labels: Mapping[int, Any] | None = None,
) -> Data:
    """Convert a Chython QueryCGRContainer into a PyG graph with rule labels.

    Chython stores composed QueryCGR topology in private ``_atoms`` and
    ``_bonds`` slots; SynPlanner centralizes that access here and labels atoms
    and bonds through its canonical QueryCGR helpers.
    """
    atom_labels = dict(atom_labels or {})
    order = _canonical_atom_order(query_cgr, atom_labels)
    positions = {atom: index for index, atom in enumerate(order)}

    x = torch.tensor(
        [_node_features(query_cgr, atom, atom_labels) for atom in order],
        dtype=torch.float,
    )
    edge_index = []
    edge_attr = []
    for atom in order:
        for neighbor in query_cgr._bonds[atom]:
            edge_index.append([positions[atom], positions[neighbor]])
            edge_attr.append(_edge_features(query_cgr, atom, neighbor))

    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty(
            (0, RULE_GRAPH_EDGE_FEATURE_DIM), dtype=torch.float
        )

    return Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)


def query_cgr_graph_from_rule_query(rule_query: ReactionContainer) -> Data:
    """Build a QueryCGR rule graph from a parsed SMARTS reaction rule."""
    return query_cgr_to_pyg(
        rule_query.compose(),
        atom_labels=query_reaction_atom_labels(rule_query),
    )


def query_cgr_graphs_from_smarts(
    rule_smarts: Sequence[str],
    *,
    schema_version: str = RULE_GRAPH_SCHEMA_VERSION,
) -> list[Data]:
    """Build ordered QueryCGR rule graphs from retrospective rule SMARTS."""
    if not schema_version:
        raise ValueError("rule_graph_schema_version must be non-empty")

    graphs = []
    for index, rule_smarts_text in enumerate(rule_smarts):
        try:
            graphs.append(query_cgr_graph_from_rule_query(smarts(rule_smarts_text)))
        except Exception as err:
            raise ValueError(
                f"Failed to featurize reaction rule graph at index {index}:\n"
                f"  SMARTS: {rule_smarts_text}\n"
                f"  error: {type(err).__name__}: {err}"
            ) from err
    return graphs


__all__ = [
    "RULE_GRAPH_EDGE_FEATURE_DIM",
    "RULE_GRAPH_NODE_FEATURE_DIM",
    "RULE_GRAPH_SCHEMA_VERSION",
    "query_cgr_graph_from_rule_query",
    "query_cgr_graphs_from_smarts",
    "query_cgr_to_pyg",
]
