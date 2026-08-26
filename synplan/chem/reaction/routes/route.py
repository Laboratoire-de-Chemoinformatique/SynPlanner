"""One synthesis route, detached from the tree that found it.

Facade only: every method delegates to the route machinery that already exists
(``Tree.synthesis_route``, ``routedraw.draw_route``, ``build_route_cgr``,
``make_json``). Nothing here walks a tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from synplan.chem.reaction.routes.io.json import (
    _route_step_metadata_from_tree,
    make_json,
    read_route_tree,
)
from synplan.chem.reaction.routes.representation.route_cgr import build_route_cgr
from synplan.utils.routedraw import ARROW_DEFS, ROUTE_CSS, draw_route

if TYPE_CHECKING:
    from chython.containers import MoleculeContainer, ReactionContainer

    from synplan.chem.reaction.routes.contracts import RouteCGRBuildResult, RouteNode
    from synplan.chem.reaction.routes.representation.container import RouteCGRContainer
    from synplan.mcts.tree import Tree

__all__ = ["Route"]


def _composition_error(result: RouteCGRBuildResult) -> ValueError:
    """A ValueError carrying the failed composition's ``RouteDiagnostic``."""

    error = ValueError(f"RouteCGR composition failed: {result.diagnostic.message}")
    error.diagnostic = result.diagnostic
    return error


@dataclass(frozen=True)
class Route:
    """A single retrosynthetic route: its steps, its stock, and nothing else.

    Holds plain data only, so it outlives the :class:`~synplan.mcts.tree.Tree`
    that produced it and survives pickling. Detached it cannot rescore
    (``Tree.route_score`` sums per-node ``total_value``, which lives only in the
    tree, so :attr:`score` is carried, not recomputable) and knows nothing about
    the rest of the search.

    :param steps: ``Tree.synthesis_route`` order, deepest step first, so
        ``steps[-1]`` is the disconnection of the target. Index in this tuple is
        the ``step_id`` every dict-driven route API expects.
    :param building_blocks: Stock SMILES the route was judged against.
    :param min_mol_size: Molecules this size or smaller count as stock regardless
        of ``building_blocks`` (``Precursor.is_building_block`` semantics).
    :param step_meta: Per-step rule metadata aligned with ``steps``; keys are
        those ``io.json`` writes onto a reaction node (``step_id``,
        ``tree_node_id``, ``rule_id``, ``rule_source``, ``rule_key``). Empty
        dicts when unknown.
    :param score: ``Tree.route_score`` at extraction time; ``None`` when the
        route did not come from a tree.
    :param route_id: The terminal node id in the source tree, or the JSON key.
    """

    steps: tuple[ReactionContainer, ...]
    building_blocks: frozenset[str] = frozenset()
    min_mol_size: int = 6
    step_meta: tuple[dict[str, Any], ...] = ()
    score: float | None = None
    route_id: int | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a route needs at least one step")

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_tree(cls, tree: Tree, node_id: int) -> Route:
        """Pull the route ending at ``node_id`` out of ``tree``.

        Works for any node, solved or not.

        :raises KeyError: if ``node_id`` is not in the tree.
        """

        if node_id not in tree.nodes:
            raise KeyError(node_id)
        steps = tuple(tree.synthesis_route(node_id))
        metadata = _route_step_metadata_from_tree(tree, node_id)
        return cls(
            steps=steps,
            building_blocks=frozenset(tree.building_blocks),
            min_mol_size=tree.config.min_mol_size,
            step_meta=tuple(metadata.get(step, {}) for step in range(len(steps))),
            score=tree.route_score(node_id),
            route_id=node_id,
        )

    @classmethod
    def from_json(cls, route_json: RouteNode, route_id: int | None = None) -> Route:
        """Rebuild a route from one v1 JSON route tree.

        ``building_blocks`` is reconstructed as the leaf SMILES flagged
        ``in_stock``, with ``min_mol_size=0``, so :attr:`solved` and
        :attr:`dead_ends` reproduce the exported flags rather than re-deciding
        them against a stock the file does not carry.
        """

        steps, metadata, in_stock = read_route_tree(route_json)
        order = sorted(steps)
        return cls(
            steps=tuple(steps[step] for step in order),
            building_blocks=in_stock,
            min_mol_size=0,
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

    @property
    def dead_ends(self) -> tuple[MoleculeContainer, ...]:
        """Terminal precursors that are not building blocks.

        A leaf is a reactant that is no step's product; it is a dead end when it
        fails the test ``Precursor.is_building_block`` applies. Empty on a solved
        route by definition. These are the molecules drawn red.
        """

        products = {str(mol) for step in self.steps for mol in step.products}
        seen, found = set(), []
        for step in self.steps:
            for mol in step.reactants:
                key = str(mol)
                if key in products or key in seen:
                    continue
                seen.add(key)
                if len(mol) > self.min_mol_size and key not in self.building_blocks:
                    found.append(mol)
        return tuple(found)

    @property
    def solved(self) -> bool:
        """True when every leaf is a building block. Derived, never stored."""

        return not self.dead_ends

    @property
    def reactions_dict(self) -> dict[int, ReactionContainer]:
        """``{step_id: reaction}`` -- the shape every dict-driven route API takes.

        ``{0: route.reactions_dict}`` is a ``routes_dict``, so this is the one
        adapter to ``build_route_cgr``, ``make_json``, ``cluster_routes``,
        ``write_routes_csv`` and friends.
        """

        return dict(enumerate(self.steps))

    def __len__(self) -> int:
        """Number of steps."""

        return len(self.steps)

    def __repr__(self) -> str:
        dead_ends = len(self.dead_ends)
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

        Dead-end leaves take routedraw's ``oos`` role, which is the red one.

        :param align: Give every precursor its product's orientation.
        :return: A complete SVG document, ``ROUTE_CSS`` and ``ARROW_DEFS`` inlined.
        """

        svg, _ = draw_route(
            self.steps, self.building_blocks, self.min_mol_size, align=align
        )
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

        Round trip keeps the steps, the per-step rule metadata and the leaf
        ``in_stock`` flags, and therefore :attr:`solved`. It drops :attr:`score`
        and :attr:`route_id` (v1 JSON has no envelope), the real stock set (only
        the leaves' verdicts survive) and 2D coordinates.

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
        metadata = {
            step: {
                **(self.step_meta[step] if step < len(self.step_meta) else {}),
                "step_id": step,
            }
            for step in steps
        }
        return make_json(
            {0: steps},
            keep_ids=False,
            route_metadata={0: metadata},
            strict=True,
            building_blocks=self.building_blocks,
            min_mol_size=self.min_mol_size,
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
