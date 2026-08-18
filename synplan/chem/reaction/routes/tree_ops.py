"""Shared operations over JSON-compatible synthesis-route trees."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from chython import smiles as read_smiles


def node_children(node: Mapping[str, Any], *, path: str = "$") -> list[Any]:
    """Return node children, validating the common route-tree container shape."""
    children = node.get("children", [])
    if children is None:
        return []
    if not isinstance(children, list):
        raise ValueError(f"invalid route node at {path}: children must be a list")
    return children


def iter_route_nodes(
    route: Mapping[str, Any],
) -> Iterator[tuple[tuple[int, ...], Mapping[str, Any]]]:
    """Yield every route node in preorder together with its child-index path."""
    if not isinstance(route, Mapping):
        raise ValueError("route must be a mapping")

    def visit(node: Mapping[str, Any], path: tuple[int, ...]):
        yield path, node
        for index, child in enumerate(node_children(node, path=str(path))):
            if not isinstance(child, Mapping):
                raise ValueError(f"invalid route child at {(*path, index)}")
            yield from visit(child, (*path, index))

    yield from visit(route, ())


def iter_molecule_leaves(
    route: Mapping[str, Any],
) -> Iterator[tuple[tuple[int, ...], Mapping[str, Any]]]:
    """Yield terminal molecule nodes and their paths in route order."""
    for path, node in iter_route_nodes(route):
        if node.get("type") == "mol" and not node_children(node, path=str(path)):
            yield path, node


def node_at(route: Mapping[str, Any], path: tuple[int, ...]) -> Mapping[str, Any]:
    """Resolve one previously emitted child-index path."""
    node = route
    for index in path:
        child = node_children(node, path=str(path))[index]
        if not isinstance(child, Mapping):
            raise ValueError(f"invalid route child at {path}")
        node = child
    return node


def iter_reactions_postorder(
    route: Mapping[str, Any],
) -> Iterator[Mapping[str, Any]]:
    """Yield reaction nodes in synthesis/step order."""
    def visit(node: Mapping[str, Any]):
        for child in node_children(node):
            if not isinstance(child, Mapping):
                raise ValueError("route child must be a mapping")
            yield from visit(child)
        if node.get("type") == "reaction":
            yield node

    yield from visit(route)


def reindex_reaction_steps(route: Mapping[str, Any]) -> None:
    """Assign contiguous postorder step IDs to mutable reaction nodes."""
    for step_id, reaction in enumerate(iter_reactions_postorder(route)):
        if not isinstance(reaction, dict):
            raise ValueError("reaction nodes must be mutable dictionaries")
        reaction["step_id"] = step_id


def max_route_atom_map(route: Mapping[str, Any]) -> int:
    """Return the largest atom map present in any serialized route reaction."""
    maximum = 0
    for _path, node in iter_route_nodes(route):
        if node.get("type") != "reaction":
            continue
        try:
            reaction = read_smiles(node["smiles"])
            maximum = max(
                maximum,
                max(
                    (
                        atom_number
                        for side in (
                            reaction.reactants,
                            reaction.reagents,
                            reaction.products,
                        )
                        for molecule in side
                        for atom_number in molecule._atoms
                    ),
                    default=0,
                ),
            )
        except Exception as error:
            reaction_smiles = node.get("smiles")
            raise ValueError(
                f"invalid reaction SMILES in route: {reaction_smiles!r}"
            ) from error
    return maximum


__all__ = [
    "iter_molecule_leaves",
    "iter_reactions_postorder",
    "iter_route_nodes",
    "max_route_atom_map",
    "node_at",
    "node_children",
    "reindex_reaction_steps",
]
