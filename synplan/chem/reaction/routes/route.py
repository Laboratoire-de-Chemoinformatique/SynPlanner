"""One synthesis route, detached from the tree that found it.

Facade only: every method delegates to the route machinery that already exists
(``Tree.synthesis_route``, ``routedraw.draw_route``, ``build_route_cgr``,
``make_json``). Nothing here walks a tree.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from synplan.chem.reaction.routes.io.json import (
    _route_step_metadata_from_tree,
    make_json,
    read_route_tree,
)
from synplan.chem.reaction.routes.representation.route_cgr import build_route_cgr
from synplan.chem.reaction.routes.traversal import linearise
from synplan.utils.routedraw import ARROW_DEFS, ROUTE_CSS, draw_route

if TYPE_CHECKING:
    from chython.containers import MoleculeContainer, ReactionContainer

    from synplan.chem.reaction.routes.contracts import RouteCGRBuildResult, RouteNode
    from synplan.chem.reaction.routes.representation.container import RouteCGRContainer
    from synplan.mcts.tree import Tree

__all__ = ["Route", "Step"]


def _composition_error(result: RouteCGRBuildResult) -> ValueError:
    """A ValueError carrying the failed composition's ``RouteDiagnostic``."""

    error = ValueError(f"RouteCGR composition failed: {result.diagnostic.message}")
    error.diagnostic = result.diagnostic
    return error


def _leaves(steps: tuple[ReactionContainer, ...]) -> dict[str, MoleculeContainer]:
    """``{smiles: molecule}`` for every reactant that is no step's product."""

    products = {str(mol) for step in steps for mol in step.products}
    leaves: dict[str, MoleculeContainer] = {}
    for step in steps:
        for mol in step.reactants:
            key = str(mol)
            if key not in products and key not in leaves:
                leaves[key] = mol
    return leaves


@dataclass(frozen=True)
class Step:
    """One step of a route, seen the way a chemist reads it.

    :param route: The route this step belongs to.
    :param step_id: Index in ``route.steps``, and the key every export writes.
    """

    route: Route
    step_id: int

    @property
    def number(self) -> int:
        """Display number, ``step_id + 1``: 1 is the first reaction run in the lab,
        ``len(route)`` the cut from the target. The disc in the drawing carries it."""

        return self.step_id + 1

    @property
    def reaction(self) -> ReactionContainer:
        """The reaction itself."""

        return self.route.steps[self.step_id]

    @property
    def conditions(self) -> Any | None:
        """Reaction conditions, a property of the reaction, so they travel with it
        through CSV, JSON and the CGR. ``None`` until something writes them."""

        return self.reaction.meta.get("conditions")

    @property
    def provenance(self) -> dict[str, Any]:
        """What produced this step: ``rule_key``, ``rule_source``, ``rule_id``,
        ``tree_node_id``. Empty when unknown."""

        return self.route.step_meta[self.step_id]

    def __repr__(self) -> str:
        return f"<Step {self.number}: {self.reaction}>"


@dataclass(frozen=True)
class Route:
    """A single retrosynthetic route: its steps, its verdict, and nothing else.

    Holds plain data only, so it outlives the :class:`~synplan.mcts.tree.Tree`
    that produced it and survives pickling. Detached it cannot rescore
    (``Tree.route_score`` sums per-node ``total_value``, which lives only in the
    tree, so :attr:`score` is carried, not recomputable), and it never re-decides
    which leaves are purchasable: whoever built it already knew.

    Construction re-linearises the steps (:func:`linearise`), so position is the
    single ordering: ``steps[i]`` is step number ``i + 1``, ``step_id`` ``i``, and
    the disc numbered ``i + 1`` in the drawing. Iterating yields :class:`Step`
    views in that order; :meth:`step` takes the number.

    :param steps: The route's reactions. Given in any topological order and
        stored depth-first from the target, deepest step first, so ``steps[-1]``
        is the disconnection of the target and each branch of a convergent route
        is a contiguous block.
    :param unresolved: SMILES of the leaves that are not purchasable -- the whole
        stored verdict. Empty on a solved route.
    :param step_meta: Per-step rule metadata; re-ordered with ``steps`` and padded
        to the same length. Keys are those ``io.json`` writes onto a reaction node
        bar ``step_id``, which is the index: ``tree_node_id``, ``rule_id``,
        ``rule_source``, ``rule_key``.
    :param score: ``Tree.route_score`` at extraction time; ``None`` when the
        route did not come from a tree.
    :param route_id: The terminal node id in the source tree, or the JSON key.
    """

    steps: tuple[ReactionContainer, ...]
    unresolved: frozenset[str] = frozenset()
    step_meta: tuple[dict[str, Any], ...] = ()
    score: float | None = None
    route_id: int | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a route needs at least one step")
        order = linearise(self.steps)
        meta = (*self.step_meta, *({},) * (len(order) - len(self.step_meta)))
        object.__setattr__(self, "steps", tuple(self.steps[i] for i in order))
        object.__setattr__(
            self,
            "step_meta",
            tuple({k: v for k, v in meta[i].items() if k != "step_id"} for i in order),
        )

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
        steps = tuple(tree.synthesis_route(node_id))
        metadata = _route_step_metadata_from_tree(tree, node_id)
        return cls(
            steps=steps,
            unresolved=frozenset(
                str(precursor) for precursor in tree.nodes[node_id].precursors_to_expand
            ),
            step_meta=tuple(metadata.get(step, {}) for step in range(len(steps))),
            score=tree.route_score(node_id),
            route_id=node_id,
        )

    @classmethod
    def from_json(cls, route_json: RouteNode, route_id: int | None = None) -> Route:
        """Rebuild a route from one v1 JSON route tree.

        The leaves the file flags ``in_stock`` are the purchasable ones; the rest
        are :attr:`unresolved`, so the exported verdict is reproduced rather than
        re-decided against a stock the file does not carry.
        """

        steps, metadata, in_stock = read_route_tree(route_json)
        order = sorted(steps)
        ordered = tuple(steps[step] for step in order)
        return cls(
            steps=ordered,
            unresolved=frozenset(_leaves(ordered)) - in_stock,
            step_meta=tuple(metadata[step] for step in order),
            route_id=route_id,
        )

    # ------------------------------------------------------------------
    # what the route is
    # ------------------------------------------------------------------

    @property
    def target(self) -> MoleculeContainer:
        """The target molecule -- ``steps[-1].products[0]``."""

        return self.steps[-1].products[0]

    def role(self, mol: MoleculeContainer | str) -> str:
        """What ``mol`` is in this route.

        ``target``, ``intermediate`` (a product of some step that is not the
        target), ``building_block`` or ``unresolved`` (a leaf, by the stored
        verdict).

        :raises KeyError: if the molecule is not in this route.
        """

        key = str(mol)
        if key == str(self.target):
            return "target"
        if key in _leaves(self.steps):
            return "unresolved" if key in self.unresolved else "building_block"
        if any(key == str(p) for step in self.steps for p in step.products):
            return "intermediate"
        raise KeyError(key)

    @property
    def dead_ends(self) -> tuple[MoleculeContainer, ...]:
        """The :attr:`unresolved` leaves, in route order. Drawn red."""

        return tuple(
            mol for key, mol in _leaves(self.steps).items() if key in self.unresolved
        )

    @property
    def solved(self) -> bool:
        """True when no leaf is unresolved."""

        return not self.unresolved

    @property
    def reactions_dict(self) -> dict[int, ReactionContainer]:
        """``{step_id: reaction}`` -- the shape every dict-driven route API takes.

        ``{0: route.reactions_dict}`` is a ``routes_dict``, so this is the one
        adapter to ``build_route_cgr``, ``make_json``, ``cluster_routes``,
        ``write_routes_csv`` and friends.
        """

        return dict(enumerate(self.steps))

    def step(self, number: int) -> Step:
        """The step with display ``number`` -- 1 is the deepest step, not an index.

        :raises IndexError: if ``number`` is outside ``1..len(route)``.
        """

        if not 1 <= number <= len(self.steps):
            raise IndexError(f"step number {number} outside 1..{len(self.steps)}")
        return Step(self, number - 1)

    def __iter__(self) -> Iterator[Step]:
        """Steps as views, number 1 first."""

        return (Step(self, step_id) for step_id in range(len(self.steps)))

    def __len__(self) -> int:
        """Number of steps."""

        return len(self.steps)

    def __repr__(self) -> str:
        dead_ends = len(self.unresolved)
        state = (
            "solved"
            if not dead_ends
            else f"unsolved ({dead_ends} dead end{'s' if dead_ends > 1 else ''})"
        )
        score = "no score" if self.score is None else f"score {self.score:.3f}"
        route_id = "-" if self.route_id is None else self.route_id
        return f"<Route {route_id}: {len(self)} steps, {state}, {score}>"

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def svg(self, align: bool = True) -> str:
        """Draw the route as a standalone SVG.

        Unresolved leaves take routedraw's ``oos`` role, which is the red one.

        :param align: Give every precursor its product's orientation.
        :return: A complete SVG document, ``ROUTE_CSS`` and ``ARROW_DEFS`` inlined.
        """

        svg = draw_route(self.steps, self.unresolved, align=align)
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

        ``step_id`` is the index, so the file lists the steps in the order this
        route does and step 0 is still performed first. Round trip keeps the steps,
        the per-step rule metadata and the leaf ``in_stock`` flags, and therefore
        :attr:`solved`. It drops :attr:`score` and :attr:`route_id` (v1 JSON has no
        envelope) and 2D coordinates.

        :param reconcile_atom_mapping: Renumber atoms consistently across steps
            before writing. Off by default because it costs a full RouteCGR
            composition; turn it on when the JSON will later be re-composed into
            a RouteCGR, or the per-step-local numbering fuses distinct atoms.
        """

        steps = self.reactions_dict
        if reconcile_atom_mapping:
            result = build_route_cgr({0: steps}, 0, include_reactions=True)
            if not result.ok:
                raise _composition_error(result)
            steps = dict(result.reactions_dict)
        metadata = {step: {**self.step_meta[step], "step_id": step} for step in steps}
        return make_json(
            {0: steps},
            keep_ids=False,
            route_metadata={0: metadata},
            strict=True,
            building_blocks=frozenset(_leaves(self.steps)) - self.unresolved,
            min_mol_size=0,
        )[0]

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
