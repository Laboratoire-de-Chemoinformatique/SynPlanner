"""One synthesis route, detached from the tree that found it.

Structure is carried by object identity: the product of one step *is* the
matching reactant of the next. Nothing here re-derives which step feeds which
from SMILES.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from synplan.chem.reaction.routes.io.json import read_route_tree, route_tree
from synplan.chem.reaction.routes.io.metadata import reaction_metadata
from synplan.chem.reaction.routes.representation.route_cgr import build_route_cgr
from synplan.chem.reaction.routes.traversal import (
    linearise,
    root_step,
    steps_by_product,
)
from synplan.chem.utils import mapped_smiles, molecule_key
from synplan.utils.routedraw import ARROW_DEFS, ROUTE_CSS, Layouts, draw_route

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from chython.containers import MoleculeContainer, ReactionContainer

    from synplan.chem.reaction.routes.contracts import RouteCGRBuildResult, RouteNode
    from synplan.chem.reaction.routes.representation.container import RouteCGRContainer
    from synplan.mcts.tree import Tree

__all__ = [
    "Conditions",
    "MoleculePosition",
    "Route",
    "RouteProvenance",
    "Step",
    "StepOrigin",
]


class MoleculePosition(str, Enum):
    """Where a molecule sits in a route. Not whether it can be bought.

    Named for the molecule, because the prefix says what the value is about and
    because a bare ``Position`` would fight the step number, also a position.
    """

    TARGET = "target"
    INTERMEDIATE = "intermediate"
    LEAF = "leaf"

    __str__ = str.__str__


@dataclass(frozen=True)
class StepOrigin:
    """Where a step came from in the search.

    Named for the step, so an instance in hand says what it describes without
    any memory of where it was read from.
    """

    rule_key: str | None = None
    rule_source: str | None = None
    rule_id: int | None = None
    tree_node_id: int | None = None


@dataclass(frozen=True)
class Conditions:
    """Reaction conditions. ``source`` is mandatory so a prediction never reads
    as a literature value."""

    source: Literal["predicted", "literature", "manual"]


@dataclass(frozen=True)
class RouteProvenance:
    """The run that produced the route, not the route itself.

    Named for the route: bare ``Provenance`` did not say whether it described a
    step or the whole route, and it describes the run behind the whole route.

    :param uncanonical: How many molecules a file the route was read from spelled
        in a way chython cannot canonicalise. They are kept as written, so the
        route is whole; the count is what says the file was.
    """

    search_score: float | None = None
    tree_node_id: int | None = None
    uncanonical: int = 0


@dataclass(frozen=True)
class Step:
    """One disconnection.

    :param reaction: The reaction as it is run in the lab.
    :param product: The molecule this step disconnects. One of
        ``reaction.products``, by identity -- never a copy, never a guess at
        which product the route cares about.
    """

    reaction: ReactionContainer
    product: MoleculeContainer
    origin: StepOrigin | None = None
    conditions: Conditions | None = None

    def __post_init__(self) -> None:
        if not any(self.product is mol for mol in self.reaction.products):
            raise ValueError(f"{self.product} is not a product of {self.reaction}")

    def __repr__(self) -> str:
        return f"<Step {self.reaction}>"


def _composition_error(result: RouteCGRBuildResult) -> ValueError:
    """A ValueError carrying the failed composition's ``RouteDiagnostic``."""

    error = ValueError(f"RouteCGR composition failed: {result.diagnostic.message}")
    error.diagnostic = result.diagnostic
    return error


def _leaves(steps: Sequence[Step]) -> tuple[MoleculeContainer, ...]:
    """Every reactant no step produces, in step order."""

    made = steps_by_product(steps)
    out, seen = [], set()
    for step in steps:
        for mol in step.reaction.reactants:
            if id(mol) not in made and id(mol) not in seen:
                seen.add(id(mol))
                out.append(mol)
    return tuple(out)


def _origin(metadata: dict[str, Any]) -> StepOrigin:
    return StepOrigin(
        rule_key=metadata.get("rule_key"),
        rule_source=metadata.get("rule_source"),
        rule_id=metadata.get("rule_id"),
        tree_node_id=metadata.get("tree_node_id"),
    )


@dataclass(frozen=True)
class Route:
    """A single retrosynthetic route: its steps, its verdict, and nothing else.

    Holds plain data only, so it outlives the :class:`~synplan.mcts.tree.Tree`
    that produced it and survives pickling. Detached it never re-decides which
    leaves are purchasable: whoever built it already knew.

    Construction re-linearises the steps (:func:`linearise`), so position is the
    single ordering: ``steps[i]`` is the disconnection carrying the disc numbered
    ``i + 1`` in the drawing, and ``steps[-1]`` cuts the target.

    :param steps: The route's steps, in any topological order.
    :param unresolved: :func:`molecule_key` of the leaves that are not
        purchasable -- the whole stored verdict. Empty on a solved route.
    :param provenance: Where the route came from -- the search run behind it, or
        what reading it out of a file turned up.
    """

    steps: tuple[Step, ...]
    unresolved: frozenset[str] = frozenset()
    provenance: RouteProvenance | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a route needs at least one step")
        order = linearise(self.steps)
        object.__setattr__(self, "steps", tuple(self.steps[i] for i in order))
        object.__setattr__(self, "unresolved", frozenset(self.unresolved))

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_tree(cls, tree: Tree, node_id: int) -> Route:
        """Pull the route ending at ``node_id`` out of ``tree``.

        The node's ``precursors_to_expand`` is the search's own list of leaves it
        could not buy, so the verdict is read, not recomputed.

        :raises KeyError: if ``node_id`` is not in the tree.
        """

        if node_id not in tree.nodes:
            raise KeyError(node_id)
        metadata = tree.step_metadata(node_id)
        steps = tuple(
            # one product per step by construction: the precursor being expanded
            Step(reaction, reaction.products[0], _origin(metadata.get(index, {})))
            for index, reaction in enumerate(tree.synthesis_route(node_id))
        )
        return cls(
            steps=steps,
            unresolved=frozenset(
                molecule_key(precursor.molecule)
                for precursor in tree.nodes[node_id].precursors_to_expand
            ),
            provenance=RouteProvenance(tree.route_score(node_id), node_id),
        )

    @classmethod
    def from_json(cls, route_json: RouteNode) -> Route:
        """Rebuild a route from one v1 JSON route tree.

        Each leaf carries the verdict of the molecule node it was read from, so
        the exported verdict is reproduced rather than re-decided against a stock
        the file does not carry. v1 JSON has no envelope, so the route comes back
        with no search behind it: its :attr:`provenance` holds only what the read
        itself found out.
        """

        parsed = read_route_tree(route_json)
        steps = tuple(
            Step(reaction, product, _origin(metadata))
            for reaction, product, metadata in parsed.steps
        )
        return cls(
            steps=steps,
            unresolved=frozenset(molecule_key(mol) for mol in parsed.unresolved),
            provenance=RouteProvenance(uncanonical=parsed.uncanonical),
        )

    # ------------------------------------------------------------------
    # what the route is
    # ------------------------------------------------------------------

    @property
    def target(self) -> MoleculeContainer:
        """The molecule the route makes: the product no step consumes."""

        return self.steps[root_step(self.steps)].product

    def leaves(self) -> tuple[MoleculeContainer, ...]:
        """The starting materials -- every reactant no step produces."""

        return _leaves(self.steps)

    def _place(
        self, key: str, spelling: Callable[[MoleculeContainer], str]
    ) -> MoleculePosition | None:
        """``key`` against every molecule of the route, written by ``spelling``."""

        if key == spelling(self.target):
            return MoleculePosition.TARGET
        if any(key == spelling(leaf) for leaf in self.leaves()):
            return MoleculePosition.LEAF
        if any(key == spelling(step.product) for step in self.steps):
            return MoleculePosition.INTERMEDIATE
        return None

    def position(self, mol: MoleculeContainer) -> MoleculePosition:
        """Where ``mol`` sits in this route. Whether it can be bought is
        :attr:`unresolved`.

        A route's own molecules are canonical, so the lookup is by spelling and
        costs nothing. Only a molecule spelled some other way is normalised
        (:func:`molecule_key`) and looked up a second time.

        :raises KeyError: if the molecule is not in this route.
        """

        found = self._place(str(mol), str)
        if found is not None:
            return found
        key = molecule_key(mol)
        found = self._place(key, molecule_key)
        if found is None:
            raise KeyError(key)
        return found

    @property
    def solved(self) -> bool:
        """True when no leaf is unresolved."""

        return not self.unresolved

    @property
    def reactions_dict(self) -> dict[int, ReactionContainer]:
        """``{step index: reaction}`` -- the shape every dict-driven route API takes.

        ``{0: route.reactions_dict}`` is a ``routes_dict``, so this is the one
        adapter to ``build_route_cgr``, ``cluster_routes``, ``write_routes_csv``
        and friends. A projection: it drops which product each step disconnects.
        """

        return {index: step.reaction for index, step in enumerate(self.steps)}

    def __iter__(self) -> Iterator[Step]:
        """Steps, the deepest one first."""

        return iter(self.steps)

    def __len__(self) -> int:
        """Number of steps."""

        return len(self.steps)

    def __repr__(self) -> str:
        unresolved = len(self.unresolved)
        state = "solved" if not unresolved else f"unsolved ({unresolved} unresolved)"
        provenance = self.provenance or RouteProvenance()
        score = (
            "no score"
            if provenance.search_score is None
            else f"score {provenance.search_score:.3f}"
        )
        node_id = "-" if provenance.tree_node_id is None else provenance.tree_node_id
        return f"<Route {node_id}: {len(self)} steps, {state}, {score}>"

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def svg(
        self, align: bool = True, standalone: bool = True, layouts: Layouts = None
    ) -> str:
        """Draw the route as an SVG.

        Unresolved leaves take routedraw's ``oos`` role, which is the red one.

        :param align: Give every precursor its product's orientation.
        :param standalone: Inline ``ROUTE_CSS`` and ``ARROW_DEFS``. Turn it off
            for a page that carries one copy of both itself.
        :param layouts: A dict shared with the other routes of the same page, so a
            molecule two routes have in common is drawn the same way in both.
        """

        unresolved: tuple[MoleculeContainer, ...] = ()
        if self.unresolved:  # nothing is in an empty set: do not key the leaves
            unresolved = tuple(
                leaf for leaf in self.leaves() if molecule_key(leaf) in self.unresolved
            )
        svg = draw_route(self.steps, unresolved, align=align, layouts=layouts)
        if not standalone:
            return svg
        head = svg.index(">") + 1
        return f"{svg[:head]}<defs>{ARROW_DEFS}</defs><style>{ROUTE_CSS}</style>{svg[head:]}"

    def _repr_svg_(self) -> str:
        """Render in a notebook."""

        return self.svg()

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def to_json(self, *, reconcile_atom_mapping: bool = False) -> RouteNode:
        """Serialise to one v1 JSON route tree.

        The file's nesting is the route's own structure, so a multi-product step
        writes the product the route actually disconnects. Round trip keeps the
        steps, the per-step :class:`StepOrigin` and the leaf ``in_stock`` flags, and
        therefore :attr:`solved`. It drops :attr:`provenance` (v1 JSON has no
        envelope) and 2D coordinates. Nothing in the route is modified.

        :param reconcile_atom_mapping: Renumber atoms consistently across steps
            before writing. Off by default because it costs a full RouteCGR
            composition; turn it on when the JSON will later be re-composed into
            a RouteCGR, or the per-step-local numbering fuses distinct atoms.
        """

        stock = frozenset(molecule_key(leaf) for leaf in self.leaves())
        stock -= self.unresolved
        target = self.target

        if reconcile_atom_mapping:
            result = build_route_cgr(
                {0: self.reactions_dict}, 0, include_reactions=True
            )
            if not result.ok:
                raise _composition_error(result)
            reactions = dict(result.reactions_dict)
            # Composition rewrites every step: it renumbers them onto one atom
            # numbering and hands back the by-products the search's steps left
            # out. Neither the objects nor the numbering survive, so the link
            # falls back to spelling, the way a route read out of a file is
            # linked -- an earlier step that makes what this one consumes.
            products = {}
            for index, step in enumerate(self.steps):
                composed = next(
                    (p for p in reactions[index].products if p == step.product), None
                )
                if composed is None:
                    raise ValueError(
                        f"step {index} no longer makes {step.product} after composition"
                    )
                products[index] = composed
            made: dict[str, list[int]] = {}
            for index, product in products.items():
                made.setdefault(str(product), []).append(index)
            target = products[root_step(self.steps)]

            def source_of(mol: MoleculeContainer, consumer: int | None):
                """Which earlier step makes ``mol`` -- matched on spelling."""
                earlier = [
                    index
                    for index in made.get(str(mol), ())
                    if consumer is None or index < consumer
                ]
                if not earlier:
                    return None
                index = max(earlier)
                return index, reactions[index], products[index]
        else:
            reactions = self.reactions_dict
            made = steps_by_product(self.steps)

            def source_of(mol: MoleculeContainer, consumer: int | None):
                """Which step makes ``mol`` -- read off identity, never re-derived."""
                index = made.get(id(mol))
                return (
                    None if index is None else (index, self.steps[index].reaction, mol)
                )

        def step_fields(index: int) -> tuple[str, dict[str, Any]]:
            step = self.steps[index]
            extra: dict[str, Any] = {}
            metadata = reaction_metadata(step.reaction)
            if metadata:
                extra["meta"] = metadata
            if step.origin is not None:
                extra["tree_node_id"] = step.origin.tree_node_id
                extra["rule_id"] = step.origin.rule_id
                extra["rule_source"] = step.origin.rule_source
                extra["rule_key"] = step.origin.rule_key
            extra["step_id"] = index
            return mapped_smiles(reactions[index]), extra

        return route_tree(
            target, source_of, lambda mol, key, leaf: key in stock, step_fields
        )

    # ------------------------------------------------------------------
    # representation
    # ------------------------------------------------------------------

    def route_cgr(self, preserve_transient_bonds: bool = True) -> RouteCGRContainer:
        """Compose the route's condensed graph.

        The container draws itself (``.depict()``), hashes (``route_cgr_hash``),
        reduces to a strategic-bond CGR (``compose_sb_cgr``) and pickles safely,
        so hold the result if you need it twice -- this does not cache.

        :raises ValueError: carrying the ``RouteDiagnostic`` when composition
            fails. Callers wanting the diagnostic without an exception call
            ``build_route_cgr`` themselves.
        """

        result = build_route_cgr({0: self.reactions_dict}, 0, preserve_transient_bonds)
        if not result.ok:
            raise _composition_error(result)
        return result.cgr
