"""Module containing a class Tree that used for tree search of retrosynthetic routes."""

import itertools
import logging
from dataclasses import dataclass, field, fields
from time import time

from chython.containers import MoleculeContainer
from chython.reactor import Reactor
from tqdm.auto import tqdm

from synplan.chem.precursor import Precursor
from synplan.chem.reaction import Reaction, apply_reaction_rule
from synplan.chem.reaction_rules import POLICY_SOURCE_NAME
from synplan.mcts.evaluation import EvaluationStrategy
from synplan.mcts.expansion import PolicyNetworkFunction, _rule_query_pattern
from synplan.mcts.node import Node
from synplan.route_quality.scorer import RouteScorer
from synplan.utils.config import TreeConfig

from .algorithm import (
    UCT,
    Beam,
    BestFirst,
    BreadthFirst,
    LazyNestedMonteCarlo,
    NestedMonteCarlo,
)

ALGORITHMS = {
    "breadth_first": BreadthFirst,
    "best_first": BestFirst,
    "beam": Beam,
    "uct": UCT,
    "nmcs": NestedMonteCarlo,
    "lazy_nmcs": LazyNestedMonteCarlo,
}

logger = logging.getLogger(__name__)


@dataclass
class PerSourceCounters:
    """Per-priority-set counters within :class:`TreeStats`."""

    tried: int = 0
    succeeded: int = 0


@dataclass
class TreeStats:
    """Typed counters for an MCTS search run.

    ``priority_rules_tried`` / ``priority_rules_succeeded`` are aggregates
    across every priority set; the per-set breakdown lives in
    ``per_priority_source`` keyed by the source name supplied as the dict
    key in ``Tree(priority_rules={...})``.
    """

    expansion_calls: int = 0
    expansion_successes: int = 0
    total_rules_tried: int = 0
    total_rules_succeeded: int = 0
    policy_rules_tried: int = 0
    policy_rules_succeeded: int = 0
    priority_rules_tried: int = 0
    priority_rules_succeeded: int = 0
    per_priority_source: dict[str, PerSourceCounters] = field(default_factory=dict)
    dead_end_nodes: int = 0
    first_solution_iteration: int | None = None
    first_solution_time: float | None = None
    routes_found_at: list[tuple[int, float]] = field(default_factory=list)

    def __getitem__(self, key: str):
        if key in {f.name for f in fields(self)}:
            raise TypeError(
                f"TreeStats is a dataclass since 1.5.0. "
                f"Use tree.stats.{key} instead of tree.stats[{key!r}]."
            )
        raise KeyError(key)


_REMOVED_NODE_ATTRS: dict[str, tuple[str, str]] = {
    "nodes_visit": ("tree.nodes[node_id].visit", "1.5.0"),
    "nodes_depth": ("tree.nodes[node_id].depth", "1.5.0"),
    "nodes_prob": ("tree.nodes[node_id].prob", "1.5.0"),
    "nodes_init_value": ("tree.nodes[node_id].init_value", "1.5.0"),
    "nodes_total_value": ("tree.nodes[node_id].total_value", "1.5.0"),
    "nodes_rules": ("tree.nodes[node_id].rule_id", "1.5.0"),
}


class Tree:
    """Tree class with attributes and methods for Monte-Carlo tree search."""

    def __init__(
        self,
        target: MoleculeContainer,
        config: TreeConfig,
        reaction_rules: list[Reactor],
        building_blocks: set[str],
        expansion_function: PolicyNetworkFunction,
        evaluation_function: EvaluationStrategy = None,
        route_scorer: RouteScorer | None = None,
        priority_rules: dict[str, list[Reactor]] | None = None,
    ):
        """Initializes a tree object with optional parameters for tree search for target
        molecule.

        :param target: A target molecule for retrosynthetic routes search.
        :param config: A tree configuration.
        :param reaction_rules: A loaded reaction rules.
        :param building_blocks: A loaded building blocks.
        :param expansion_function: A loaded policy function.
        :param evaluation_function: An evaluation strategy. If None, a random
            evaluation strategy is used as default.
        :param route_scorer: Optional post-search route scorer for
            re-ranking winning routes.  When set, :meth:`route_score`
            delegates to ``route_scorer.rescore(original, route)``.
        :param priority_rules: Optional **mapping** of curated rule sets,
            ``{set_name: [Reactor, ...]}``. Each set contributes rules tagged
            with its mapping key as ``rule_source``; e.g.
            ``priority_rules={"ugi": ugi_rules, "boc_deprotection": boc_rules}``
            yields a search where each Ugi child carries ``rule_source="ugi"``
            and each Boc child carries ``rule_source="boc_deprotection"``.
            Per-set tried/succeeded counters are kept in
            ``tree.stats.per_priority_source[<set_name>]``; the
            ``priority_rules_tried`` / ``priority_rules_succeeded`` aggregates
            sum across all sets. Reserved set name: ``"policy"`` (raises
            ``ValueError``). When ``config.use_priority`` is set, every
            priority set is tried *before* the policy on every node — rules
            match by simple substructure isomorphism (``pattern < molecule``)
            and enter with ``prob=1.0``; combined with the per-product
            fragment-count multiplier in :meth:`_add_child_if_new`
            (``scaled_prob = prob × n_qualifying_fragments``), a priority
            disconnect producing N qualifying fragments enters UCB with prior
            N. Multi-fragment priority disconnects (e.g. 4-component Ugi)
            therefore dominate sibling selection by design.
        """

        # tree config parameters
        self.config = config

        # building blocks and reaction reaction_rules
        self.reaction_rules = tuple(reaction_rules)
        self.building_blocks = frozenset(building_blocks)
        self.priority_rules: dict[str, tuple[Reactor, ...]] = {
            name: tuple(rules) for name, rules in (priority_rules or {}).items()
        }
        if self.config.use_priority and not self.priority_rules:
            raise ValueError(
                "config.use_priority=True but no priority_rules were supplied. "
                "Pass priority_rules={'<name>': [...rules]} or set use_priority=False."
            )

        # policy and evaluation services
        assert expansion_function is not None, "Expansion function is required"
        self.expansion_function = expansion_function

        assert evaluation_function is not None, "Evaluation function is required"
        self.evaluator = evaluation_function

        # post-search route re-ranking
        self._route_scorer = route_scorer
        self._rescore_cache: dict[int, float] = {}

        # tree initialization
        target_node = self._init_target_node(target)
        self.nodes: dict[int, Node] = {1: target_node}
        self.parents: dict[int, int] = {1: 0}
        self.redundant_children: dict[int, set[int]] = {1: set()}
        self.children: dict[int, set[int]] = {1: set()}
        self.winning_nodes: list[int] = []
        self.visited_nodes: set[int] = set()
        self.expanded_nodes: set[int] = set()

        # default search parameters
        self.init_node_value: float = self.config.init_node_value

        # tree building limits
        self.curr_iteration: int = 0
        self.curr_tree_size: int = 2
        self.start_time: float = 0
        self.curr_time: float = 0

        # utils
        self._tqdm = True  # needed to disable tqdm with multiprocessing module

        # other tree search algorithms
        self.stop_at_first = False
        self.found_a_route = False
        self.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks = {}

        # search statistics
        self.stats: TreeStats = TreeStats()

        # choose search algorithm (normalize key)
        algo_key = str(config.algorithm).lower()
        if algo_key not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{config.algorithm}'. Allowed: {list(ALGORITHMS.keys())}"
            )
        self.algorithm = ALGORITHMS[algo_key](self)

        priority_rules_total = sum(len(rules) for rules in self.priority_rules.values())
        logger.debug(
            f"Tree init: target={str(target)[:50]}, "
            f"building_blocks={len(self.building_blocks)}, "
            f"reaction_rules={len(self.reaction_rules)}, "
            f"priority_rules={priority_rules_total} across "
            f"{len(self.priority_rules)} sets {list(self.priority_rules)}, "
            f"use_priority={self.config.use_priority}, "
            f"algorithm={algo_key}, "
            f"max_iterations={config.max_iterations}, "
            f"max_tree_size={config.max_tree_size}, "
            f"max_time={config.max_time}, "
            f"max_depth={config.max_depth}, "
            f"min_mol_size={config.min_mol_size}, "
            f"search_strategy={config.search_strategy}, "
            f"normalize_scores={config.normalize_scores}, "
        )

    def __len__(self) -> int:
        """Returns the current size (the number of nodes) in the tree."""

        return self.curr_tree_size - 1

    def __iter__(self) -> "Tree":
        """The function is defining an iterator for a Tree object.

        Also needed for the bar progress display.
        """

        self.start_time = time()
        if self._tqdm:
            self._tqdm = tqdm(
                total=self.config.max_iterations, disable=self.config.silent
            )
        return self

    def __repr__(self) -> str:
        """Returns a string representation of the tree (target SMILES, tree size, and
        the number of found routes)."""
        return self.report()

    def __next__(self) -> tuple[bool, list[int]]:
        """The __next__ method is used to do one iteration of the tree building.

        :return: Returns True if the route was found and the node id of the last node in
            the route. Otherwise, returns False and the id of the last visited node.
        """

        if self.curr_iteration >= self.config.max_iterations:
            raise StopIteration("Iterations limit exceeded.")
        if self.curr_tree_size >= self.config.max_tree_size:
            raise StopIteration("Max tree size exceeded or all possible routes found.")
        if self.curr_time >= self.config.max_time:
            raise StopIteration("Time limit exceeded.")
        if self.stop_at_first and self.found_a_route:
            raise StopIteration("Already found a route.")

        # start new iteration
        self.curr_iteration += 1
        self.curr_time = time() - self.start_time

        if self._tqdm:
            self._tqdm.update()

        is_solved, last_node_id = self.algorithm.step()

        if is_solved:
            self.stats.routes_found_at.append(
                (self.curr_iteration, round(self.curr_time, 4))
            )
            if self.stats.first_solution_iteration is None:
                self.stats.first_solution_iteration = self.curr_iteration
                self.stats.first_solution_time = round(self.curr_time, 4)

        return is_solved, last_node_id

    def _init_target_node(self, target: MoleculeContainer):
        assert isinstance(target, MoleculeContainer), (
            "Target should be given as MoleculeContainer"
        )
        assert len(target) > 3, "Target molecule has less than 3 atoms"

        target_molecule = Precursor(target)
        target_molecule.prev_precursors.append(Precursor(target))
        target_node = Node(
            precursors_to_expand=(target_molecule,), new_precursors=(target_molecule,)
        )

        return target_node

    def _expand_node(self, node_id: int) -> None:
        """Expands the node by generating new precursors using the policy function.

        :param node_id: The id of the node to be expanded.
        :return: None.
        """
        total_expanded = 0
        curr_node = self.nodes[node_id]
        prev_precursor = curr_node.curr_precursor.prev_precursors

        # Track raw product molecules to avoid repeating equivalent expansions
        tmp_products = set()
        expanded = False
        for prob, rule, rule_id, rule_source, policy_rank in self._iter_rules(
            curr_node
        ):
            self._increment_rule_stats(rule_source, tried=1)
            rule_produced = False
            # Multi-application fires for any priority source (every source
            # other than "policy"); the configured boolean is a global switch.
            enable_multirule = bool(
                rule_source != POLICY_SOURCE_NAME
                and self.config.priority_rule_multiapplication
            )
            for products in apply_reaction_rule(
                curr_node.curr_precursor.molecule,
                rule,
                multirule=enable_multirule,
                rm_dup=enable_multirule,
            ):
                if self._add_child_if_new(
                    node_id=node_id,
                    curr_node=curr_node,
                    prev_precursor=prev_precursor,
                    products=products,
                    prob=prob,
                    rule_id=rule_id,
                    policy_rank=policy_rank,
                    tmp_products=tmp_products,
                    rule_source=rule_source,
                ):
                    rule_produced = True
                    total_expanded += 1
                    expanded = True

            if rule_produced:
                self._increment_rule_stats(rule_source, succeeded=1)

        # update statistics
        self.stats.expansion_calls += 1
        if expanded:
            self.stats.expansion_successes += 1
        elif node_id != 1:
            self.stats.dead_end_nodes += 1

        if not expanded and node_id == 1:
            raise StopIteration("\nThe target molecule was not expanded.")

    def _iter_rules(self, curr_node: Node):
        """Yield reaction rules from each enabled source for the current node.

        When ``config.use_priority`` is set and ``priority_rules`` is non-empty,
        every priority set is iterated **first** with ``prob=1.0`` and a
        ``rule_source`` equal to its mapping key (e.g. ``"ugi"``). Combined
        with the fragment-count multiplier applied in :meth:`_add_child_if_new`,
        priority disconnects intentionally dominate sibling selection — see the
        ``priority_rules`` parameter on :meth:`__init__` for details. Sets are
        iterated in insertion order; rules within a set in their list order.

        Policy rules follow, paginated by ``policy_top_rules`` and yielded with
        the policy probability and a 1-indexed ``policy_rank``. Priority and
        policy rules are *additive*, not gated: both run on every node.
        """

        if self.config.use_priority and self.priority_rules:
            molecule = curr_node.curr_precursor.molecule
            for source_name, rules in self.priority_rules.items():
                for rule_id, rule in enumerate(rules):
                    pattern = _rule_query_pattern(rule)
                    if pattern is None:
                        continue
                    try:
                        if pattern < molecule:
                            yield 1.0, rule, rule_id, source_name, None
                    except TypeError:
                        continue

        policy_top_rules = self._get_policy_top_rules_limit()
        for policy_rank, (prob, rule, rule_id) in enumerate(
            self.expansion_function.predict_reaction_rules(
                curr_node.curr_precursor, self.reaction_rules
            ),
            start=1,
        ):
            if policy_top_rules is not None and policy_rank > policy_top_rules:
                break
            yield prob, rule, rule_id, POLICY_SOURCE_NAME, policy_rank

    def _get_policy_top_rules_limit(self) -> int | None:
        """Return the configured policy Top-N limit when exposed by the expansion fn."""

        config = getattr(self.expansion_function, "config", None)
        top_rules = getattr(config, "top_rules", None)
        if top_rules is None:
            top_rules = getattr(self.expansion_function, "top_rules", None)
        if top_rules is None:
            return None
        return int(top_rules)

    def _add_child_if_new(
        self,
        node_id: int,
        curr_node: Node,
        prev_precursor,
        products,
        prob: float,
        rule_id: int,
        policy_rank: int | None,
        tmp_products: set,
        rule_source: str,
    ) -> bool:
        """Add a child node if the generated products form a new valid state."""

        if not products or not (set(products) - tmp_products):
            return False
        tmp_products.update(products)

        rule_key = self._make_rule_key(rule_source, rule_id)
        for molecule in products:
            molecule.meta["reactor_id"] = rule_id
            molecule.meta["rule_source"] = rule_source
            molecule.meta["rule_key"] = rule_key
            molecule.meta["policy_rank"] = policy_rank

        # ``apply_reaction_rule`` already validated + canonicalized each
        # product in a single kekule pass; skip the redundant copy here.
        new_precursor = tuple(Precursor(mol, canonicalize=False) for mol in products)
        # Multiply prob by the number of qualifying fragments so that
        # disconnections producing more usable precursors are preferred. Note:
        # priority rules enter here with prob=1.0, so a priority disconnect
        # producing N qualifying fragments is recorded with prior N — this is
        # the intentional mechanism that lets curated priority rules outrank
        # policy rules during UCB sibling selection.
        scaled_prob = prob * len(
            [mol for mol in products if len(mol) > self.config.min_mol_size]
        )

        if not set(prev_precursor).isdisjoint(new_precursor):
            return False

        precursors_to_expand = (
            *curr_node.next_precursor,
            *(
                x
                for x in new_precursor
                if not x.is_building_block(
                    self.building_blocks, self.config.min_mol_size
                )
            ),
        )

        if self.config.enable_pruning:
            if (
                precursors_to_expand != ()
                and precursors_to_expand
                in self.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks
            ):
                existing_id = self.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks[
                    precursors_to_expand
                ]
                self.redundant_children[node_id].add(existing_id)
                return True

            self.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks[
                precursors_to_expand
            ] = self.curr_tree_size

        child_node = Node(
            precursors_to_expand=precursors_to_expand,
            new_precursors=new_precursor,
        )

        for np in new_precursor:
            np.prev_precursors = [np, *prev_precursor]

        self._add_node(
            node_id=node_id,
            new_node=child_node,
            policy_prob=scaled_prob,
            rule_id=rule_id,
            policy_rank=policy_rank,
            rule_source=rule_source,
        )
        return True

    def _increment_rule_stats(
        self, rule_source: str, tried: int = 0, succeeded: int = 0
    ) -> None:
        """Track per-source and aggregate rule usage statistics.

        Policy rules increment the dedicated policy counters. Anything else is
        treated as a priority source: the priority aggregates plus the
        per-source breakdown in ``stats.per_priority_source[rule_source]``.

        .. note::
           ``succeeded`` counts are **order-dependent** when two rule sources
           can produce the same product set on the same node. ``_iter_rules``
           yields priority sets first, then policy; ``_add_child_if_new``
           shares a single ``tmp_products`` deduplication set across the loop.
           If a priority rule "claims" a product set, an equivalent policy
           rule fired afterwards is silently skipped and the policy counter
           for that node is *not* incremented. The total tree shape is
           correct (we don't want duplicate children for the same product
           set) but per-source success rates can underrepresent the policy
           when priority rules overlap.
        """

        self.stats.total_rules_tried += tried
        self.stats.total_rules_succeeded += succeeded

        if rule_source == POLICY_SOURCE_NAME:
            self.stats.policy_rules_tried += tried
            self.stats.policy_rules_succeeded += succeeded
            return

        self.stats.priority_rules_tried += tried
        self.stats.priority_rules_succeeded += succeeded
        per_source = self.stats.per_priority_source.setdefault(
            rule_source, PerSourceCounters()
        )
        per_source.tried += tried
        per_source.succeeded += succeeded

    @staticmethod
    def _make_rule_key(rule_source: str | None, rule_id: int | None) -> str | None:
        """Build a collision-safe rule identifier."""

        if rule_source is None or rule_id is None:
            return None
        return f"{rule_source}:{rule_id}"

    def _add_node(
        self,
        node_id: int,
        new_node: Node,
        policy_prob: float | None = None,
        rule_id: int | None = None,
        policy_rank: int | None = None,
        rule_source: str | None = None,
    ) -> None:
        """Adds a new node to the tree with probability of reaction rules predicted by
        policy function and applied to the parent node of the new node.

        :param node_id: The id of the parent node.
        :param new_node: The new node to be added.
        :param policy_prob: The probability of reaction rules predicted by policy
            function for thr parent node.
        :return: None.
        """

        new_node_id = self.curr_tree_size

        # Per-node search state and provenance live on the Node itself.
        new_node.visit = 0
        new_node.prob = policy_prob if policy_prob is not None else 0.0
        new_node.rule_id = rule_id
        new_node.rule_source = rule_source
        new_node.rule_key = self._make_rule_key(rule_source, rule_id)
        new_node.policy_rank = policy_rank
        new_node.depth = self.nodes[node_id].depth + 1

        self.nodes[new_node_id] = new_node
        self.parents[new_node_id] = node_id
        self.redundant_children[new_node_id] = set()
        self.children[node_id].add(new_node_id)
        self.children[new_node_id] = set()
        self.curr_tree_size += 1

        if self.config.search_strategy == "evaluation_first":
            node_value = self._get_node_value(new_node_id)
        elif self.config.search_strategy == "expansion_first":
            node_value = self.init_node_value

        new_node.init_value = node_value
        new_node.total_value = node_value

    def _get_node_value(self, node_id: int) -> float:
        """Calculates the value for the given node (for example with rollout or value
        network).

        :param node_id: The id of the node to be evaluated.
        :return: The estimated value of the node.
        """

        node = self.nodes[node_id]
        node_value = self.evaluator.evaluate_node(
            node=node,
            node_id=node_id,
            nodes=self.nodes,
        )
        return node_value

    def _log_final_stats(self, reason: str = "completed") -> None:
        """Logs final tree statistics after search completes.

        :param reason: Reason for stopping (e.g., "iterations limit", "time limit").
        """
        max_depth = max(n.depth for n in self.nodes.values()) if self.nodes else 0
        logger.debug(
            f"Tree finished ({reason}): "
            f"iterations={self.curr_iteration}, "
            f"tree_size={self.curr_tree_size}, "
            f"nodes={len(self.nodes)}, "
            f"visited_nodes={len(self.visited_nodes)}, "
            f"expanded_nodes={len(self.expanded_nodes)}, "
            f"winning_nodes={len(self.winning_nodes)}, "
            f"max_depth={max_depth}, "
            f"time={self.curr_time:.2f}s, "
            f"children={sum(len(v) for v in self.children.values())}, "
            f"redundant_children={sum(len(v) for v in self.redundant_children.values())}"
        )

    def _update_visits(self, node_id: int) -> None:
        """Updates the number of visits from the current node to the root node.

        :param node_id: The id of the current node.
        :return: None.
        """

        while node_id:
            self.nodes[node_id].visit += 1
            node_id = self.parents[node_id]

    def report(self) -> str:
        """Returns the string representation of the tree."""

        return (
            f"Tree for: {self.nodes[1].precursors_to_expand[0]!s}\n"
            f"Time: {round(self.curr_time, 1)} seconds\n"
            f"Number of nodes: {len(self)}\n"
            f"Number of iterations: {self.curr_iteration}\n"
            f"Number of visited nodes: {len(self.visited_nodes)}\n"
            f"Number of found routes: {len(self.winning_nodes)}"
        )

    def route_score(self, node_id: int) -> float:
        """Calculates the score of a given route from the current node to the root node.
        The score depends on cumulated node values and the route length.

        When a ``route_scorer`` is set, the raw score is passed through
        ``route_scorer.rescore(original, route)`` for post-search
        re-ranking (e.g. protection-group penalty).

        :param node_id: The id of the current given node.
        :return: The route score.
        """

        cumulated_nodes_value, route_length = 0, 0
        nid = node_id
        while nid:
            route_length += 1
            cumulated_nodes_value += self.nodes[nid].total_value
            nid = self.parents[nid]

        original = cumulated_nodes_value / (route_length**2)

        if self._route_scorer is None:
            return original

        if node_id not in self._rescore_cache:
            route = self.synthesis_route(node_id)
            self._rescore_cache[node_id] = self._route_scorer.rescore(original, route)
        return self._rescore_cache[node_id]

    def route_to_node(self, node_id: int) -> list[Node,]:
        """Returns the route (list of id of nodes) to from the node current node to the
        root node.

        :param node_id: The id of the current node.
        :return: The list of nodes.
        """

        nodes = []
        while node_id:
            nodes.append(node_id)
            node_id = self.parents[node_id]
        return [self.nodes[node_id] for node_id in reversed(nodes)]

    def synthesis_route(self, node_id: int) -> tuple[Reaction,]:
        """Given a node_id, return a tuple of reactions that represent the
        retrosynthetic route from the current node.

        :param node_id: The id of the current node.
        :return: The tuple of extracted reactions representing the synthesis route.
        """

        nodes = self.route_to_node(node_id)

        reaction_sequence = [
            Reaction(
                [x.molecule for x in after.new_precursors],
                [before.curr_precursor.molecule],
            )
            for before, after in itertools.pairwise(nodes)
        ]

        for r in reaction_sequence:
            r.clean2d()
        return tuple(reversed(reaction_sequence))

    def newickify(self, visits_threshold: int = 0, root_node_id: int = 1):
        """
        Adopted from https://stackoverflow.com/questions/50003007/how-to-convert-python-dictionary-to-newick-form-format.

        :param visits_threshold: The minimum number of visits for the given node.
        :param root_node_id: The id of the root node.

        :return: The newick string and meta dict.
        """
        visited_nodes = set()

        def newick_render_node(current_node_id: int) -> str:
            """Recursively generates a Newick string representation of the tree.

            :param current_node_id: The id of the current node.
            :return: A string representation of a node in a Newick format.
            """
            assert current_node_id not in visited_nodes, (
                "Error: The tree may not be circular!"
            )
            node_visit = self.nodes[current_node_id].visit

            visited_nodes.add(current_node_id)
            if self.children[current_node_id]:
                # Nodes
                children = [
                    child
                    for child in list(self.children[current_node_id])
                    if self.nodes[child].visit >= visits_threshold
                ]
                children_strings = [newick_render_node(child) for child in children]
                children_strings = ",".join(children_strings)
                if children_strings:
                    return f"({children_strings}){current_node_id}:{node_visit}"
                # leafs within threshold
                return f"{current_node_id}:{node_visit}"

            return f"{current_node_id}:{node_visit}"

        newick_string = newick_render_node(root_node_id) + ";"

        meta = {}
        for node_id in iter(visited_nodes):
            node = self.nodes[node_id]
            node_value = round(node.total_value, 3)

            node_synthesisability = round(node.init_value)

            visit_in_node = node.visit
            meta[node_id] = (node_value, node_synthesisability, visit_in_node)

        return newick_string, meta

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def winning_rule_ranks(self) -> list[dict]:
        """For each winning route, return the rule rank and probability at each step.

        When available, the rank is the exact 1-indexed position of the chosen
        policy rule in the model's Top-N prediction list. If that value is not
        stored for a node, the rank falls back to a sibling-probability
        approximation.

        :return: List of dicts, one per winning route, each containing
            ``winning_node_id`` and ``steps`` (list of per-step dicts with
            node_id, rule_id, prob, rank).
        """
        results = []
        for win_id in self.winning_nodes:
            steps = []
            nid = win_id
            while nid and nid != 1:
                parent_id = self.parents[nid]
                node = self.nodes[nid]
                rule_id = node.rule_id
                prob = node.prob

                rank = node.policy_rank
                if rank is None:
                    siblings = self.children.get(parent_id, set())
                    rank = 1 + sum(
                        1
                        for sib in siblings
                        if sib != nid and self.nodes[sib].prob > prob
                    )

                steps.append(
                    {
                        "node_id": nid,
                        "rule_id": rule_id,
                        "rule_source": node.rule_source,
                        "rule_key": node.rule_key,
                        "policy_rank": node.policy_rank,
                        "prob": round(prob, 6),
                        "rank": rank,
                    }
                )
                nid = parent_id

            results.append(
                {
                    "winning_node_id": win_id,
                    "steps": list(reversed(steps)),
                }
            )
        return results

    def rule_applicability_rate(self) -> float:
        """Fraction of tried rules that produced valid products.

        :return: Float in [0, 1], or 0.0 if no rules were tried.
        """
        tried = self.stats.total_rules_tried
        if tried == 0:
            return 0.0
        return self.stats.total_rules_succeeded / tried

    def branching_profile(self) -> dict[int, dict]:
        """Compute mean branching factor per depth level (expanded nodes only).

        :return: Dict mapping depth -> {"mean_children": float, "nodes": int}.
        """
        from collections import defaultdict

        depth_children: dict[int, list[int]] = defaultdict(list)
        for nid in self.expanded_nodes:
            depth = self.nodes[nid].depth
            n_children = len(self.children.get(nid, set()))
            depth_children[depth].append(n_children)

        return {
            depth: {
                "mean_children": round(sum(counts) / len(counts), 2),
                "nodes": len(counts),
            }
            for depth, counts in sorted(depth_children.items())
        }

    def route_details(self, node_id: int) -> dict:
        """Get full details about a route ending at the given node.

        :param node_id: The id of the terminal (winning) node.
        :return: Dict with route_score, route_length, and per-step details.
        """
        steps = []
        nid = node_id
        while nid and nid != 1:
            parent_id = self.parents[nid]
            node = self.nodes[nid]
            steps.append(
                {
                    "node_id": nid,
                    "depth": node.depth,
                    "rule_id": node.rule_id,
                    "rule_source": node.rule_source,
                    "rule_key": node.rule_key,
                    "policy_rank": node.policy_rank,
                    "prob": round(node.prob, 6),
                    "init_value": round(node.init_value, 4),
                    "total_value": round(node.total_value, 4),
                    "visits": node.visit,
                    "is_solved": node.is_solved(),
                    "n_precursors": len(node.new_precursors),
                }
            )
            nid = parent_id

        return {
            "node_id": node_id,
            "route_score": round(self.route_score(node_id), 6),
            "route_length": len(steps),
            "steps": list(reversed(steps)),
        }

    def to_stats_dict(self) -> dict:
        """Return a flat dict with all tree statistics for CSV/JSON export.

        Combines basic tree metrics with policy analytics and search dynamics.
        """
        # Branching stats
        all_children_counts = [
            len(self.children.get(nid, set())) for nid in self.expanded_nodes
        ]
        max_bf = max(all_children_counts) if all_children_counts else 0
        mean_bf = (
            round(sum(all_children_counts) / len(all_children_counts), 2)
            if all_children_counts
            else 0.0
        )

        # Best route info
        best_score = None
        mean_winning_rank = None
        if self.winning_nodes:
            best_score = round(
                max(self.route_score(nid) for nid in self.winning_nodes), 6
            )
            ranks_info = self.winning_rule_ranks()
            all_ranks = [
                step["rank"] for route in ranks_info for step in route["steps"]
            ]
            if all_ranks:
                mean_winning_rank = round(sum(all_ranks) / len(all_ranks), 2)

        # A "priority" step is any step whose rule_source is not the policy
        # bucket. With multi-source priority sets, each set's name lives in
        # rule_source; this check folds them together for the aggregate.
        n_routes_with_priority = 0
        for route_id in self.winning_nodes:
            nid = route_id
            while nid and nid != 1:
                src = self.nodes[nid].rule_source
                if src is not None and src != POLICY_SOURCE_NAME:
                    n_routes_with_priority += 1
                    break
                nid = self.parents[nid]
        fraction_routes_with_priority = (
            n_routes_with_priority / len(self.winning_nodes)
            if self.winning_nodes
            else 0.0
        )

        per_priority_source = {
            name: {"tried": counters.tried, "succeeded": counters.succeeded}
            for name, counters in self.stats.per_priority_source.items()
        }

        return {
            # Existing basics
            "num_routes": len(self.winning_nodes),
            "num_nodes": len(self),
            "num_iter": self.curr_iteration,
            "tree_depth": (
                max(n.depth for n in self.nodes.values()) if self.nodes else 0
            ),
            "search_time": round(self.curr_time, 1),
            "solved": len(self.winning_nodes) > 0,
            # Policy performance
            "expansion_calls": self.stats.expansion_calls,
            "expansion_successes": self.stats.expansion_successes,
            "total_rules_tried": self.stats.total_rules_tried,
            "total_rules_succeeded": self.stats.total_rules_succeeded,
            "policy_rules_tried": self.stats.policy_rules_tried,
            "policy_rules_succeeded": self.stats.policy_rules_succeeded,
            "rule_applicability_rate": round(self.rule_applicability_rate(), 4),
            "dead_end_nodes": self.stats.dead_end_nodes,
            # Priority rules usage
            "priority_rules_tried": self.stats.priority_rules_tried,
            "priority_rules_succeeded": self.stats.priority_rules_succeeded,
            "per_priority_source": per_priority_source,
            "n_routes_with_priority": n_routes_with_priority,
            "fraction_routes_with_priority": fraction_routes_with_priority,
            # Search dynamics
            "first_solution_iteration": self.stats.first_solution_iteration,
            "first_solution_time": self.stats.first_solution_time,
            # Tree shape
            "max_branching_factor": max_bf,
            "mean_branching_factor": mean_bf,
            # Route quality
            "best_route_score": best_score,
            "mean_winning_rule_rank": mean_winning_rank,
        }

    def __getattr__(self, name: str):
        if name in _REMOVED_NODE_ATTRS:
            new_api, version = _REMOVED_NODE_ATTRS[name]
            raise AttributeError(
                f"Tree.{name} was removed in {version}. "
                f"Use {new_api} instead. "
                f"See the {version} CHANGELOG entry."
            )
        raise AttributeError(f"'Tree' object has no attribute {name!r}")
