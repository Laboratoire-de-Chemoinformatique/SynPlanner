"""Ordering the steps of a route. Walking a search tree is the tree's own job."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def steps_by_product(steps: Sequence[Any]) -> dict[int, int]:
    """``{id(product): step index}`` -- what makes a route a graph rather than a list.

    Keyed by identity, not by spelling: in a symmetric disconnection two steps
    make the same SMILES and stay two steps.
    """

    return {id(step.product): index for index, step in enumerate(steps)}


def root_step(steps: Sequence[Any]) -> int:
    """Index of the step whose product no other step consumes.

    :param steps: :class:`~synplan.chem.reaction.routes.route.Step` records.
    :raises ValueError: unless exactly one step qualifies; a set of steps that
        does not end in one molecule is not a route.
    """

    consumed = {id(mol) for step in steps for mol in step.reaction.reactants}
    roots = [i for i, step in enumerate(steps) if id(step.product) not in consumed]
    if len(roots) != 1:
        raise ValueError(f"a route has one final step, found {len(roots)}")
    return roots[0]


def linearise(steps: Sequence[Any]) -> tuple[int, ...]:
    """Order the steps so that each branch of a convergent route is contiguous.

    The search enumerates precursors breadth-first, which interleaves the branches;
    a depth-first post-order walk from the target puts every linear stretch back
    together. The result is still topological: a step comes after the steps feeding
    it. Which step feeds which is read from object identity -- in a symmetric
    disconnection two steps share a product SMILES and are still two steps.

    :param steps: :class:`~synplan.chem.reaction.routes.route.Step` records, in
        any topological order.
    :return: Indices into ``steps``, deepest step first and the target's
        disconnection last.
    :raises ValueError: if a step is not reachable from the final one.
    """

    by_product = steps_by_product(steps)
    order: list[int] = []
    seen: set[int] = set()

    def visit(index: int) -> None:
        seen.add(index)
        for mol in steps[index].reaction.reactants:
            feeder = by_product.get(id(mol))
            if feeder is not None and feeder not in seen:
                visit(feeder)
        order.append(index)

    visit(root_step(steps))
    if len(order) != len(steps):
        raise ValueError(
            f"{len(steps) - len(order)} of {len(steps)} steps do not reach the target"
        )
    return tuple(order)


__all__ = [
    "linearise",
    "root_step",
    "steps_by_product",
]
