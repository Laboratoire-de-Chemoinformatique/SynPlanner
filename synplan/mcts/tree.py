"""Module containing a class Tree that used for tree search of retrosynthetic routes."""

import logging
from time import time
from typing import Dict, List, Set, Tuple

from CGRtools.containers import MoleculeContainer
from CGRtools.reactor import Reactor
from tqdm.auto import tqdm

from synplan.chem.precursor import Precursor
from synplan.chem.reaction import Reaction, apply_reaction_rule
from synplan.mcts.evaluation import EvaluationStrategy
from synplan.mcts.expansion import PolicyNetworkFunction
from synplan.mcts.node import Node
from synplan.utils.config import TreeConfig

from .algorithm import (
    Beam,
    BestFirst,
    BreadthFirst,
    LazyNestedMonteCarlo,
    NestedMonteCarlo,
    UCT,
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


class Tree:
    """Tree class with attributes and methods for Monte-Carlo tree search."""

    def __init__(
        self,
        target: MoleculeContainer,
        config: TreeConfig,
        reaction_rules: List[Reactor],
        building_blocks: Set[str],
        expansion_function: PolicyNetworkFunction,
        evaluation_function: EvaluationStrategy = None,
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
        """

        # tree config parameters
        self.config = config

        # building blocks and reaction reaction_rules
        self.reaction_rules = tuple(reaction_rules)
        self.building_blocks = frozenset(building_blocks)

        # policy and evaluation services
        assert expansion_function is not None, "Expansion function is required"
        self.expansion_function = expansion_function

        assert evaluation_function is not None, "Evaluation function is required"
        self.evaluator = evaluation_function

        # tree initialization
        target_node = self._init_target_node(target)
        self.nodes: Dict[int, Node] = {1: target_node}
        self.parents: Dict[int, int] = {1: 0}
        self.redundant_children: Dict[int, Set[int]] = {1: set()}
        self.children: Dict[int, Set[int]] = {1: set()}
        self.winning_nodes: List[int] = []
        self.visited_nodes: Set[int] = set()
        self.expanded_nodes: Set[int] = set()
        self.nodes_visit: Dict[int, int] = {1: 0}
        self.nodes_depth: Dict[int, int] = {1: 0}
        self.nodes_prob: Dict[int, float] = {1: 0.0}
        self.nodes_rules: Dict[int, float] = {}
        self.nodes_init_value: Dict[int, float] = {1: 0.0}
        self.nodes_total_value: Dict[int, float] = {1: 0.0}

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

        # choose search algorithm (normalize key)
        algo_key = str(config.algorithm).lower()
        if algo_key not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{config.algorithm}'. Allowed: {list(ALGORITHMS.keys())}"
            )
        self.algorithm = ALGORITHMS[algo_key](self)

        logger.debug(
            f"Tree init: target={str(target)[:50]}, "
            f"building_blocks={len(self.building_blocks)}, "
            f"reaction_rules={len(self.reaction_rules)}, "
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

    def __next__(self) -> [bool, List[int]]:
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

        return is_solved, last_node_id

    def _init_target_node(self, target: MoleculeContainer):

        assert isinstance(
            target, MoleculeContainer
        ), "Target should be given as MoleculeContainer"
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

        prediction = self.expansion_function.predict_reaction_rules(
            curr_node.curr_precursor, self.reaction_rules
        )

        for prob, rule, rule_id in prediction:
            for products in apply_reaction_rule(
                curr_node.curr_precursor.molecule, rule
            ):
                # check repeated products against previously produced molecules
                if not products or not (set(products) - tmp_products):
                    continue
                tmp_products.update(products)

                for molecule in products:
                    molecule.meta["reactor_id"] = rule_id

                new_precursor = tuple(Precursor(mol) for mol in products)
                scaled_prob = prob * len(
                    list(filter(lambda x: len(x) > self.config.min_mol_size, products))
                )

                if set(prev_precursor).isdisjoint(new_precursor):
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

                    # tree pruning start
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
                            total_expanded += 1
                            expanded = True
                            continue
                        else:
                            self.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks[
                                precursors_to_expand
                            ] = self.curr_tree_size

                    child_node = Node(
                        precursors_to_expand=precursors_to_expand,
                        new_precursors=new_precursor,
                    )

                    for np in new_precursor:
                        np.prev_precursors = [np, *prev_precursor]

                    self._add_node(node_id, child_node, scaled_prob, rule_id)
                    total_expanded += 1
                    expanded = True

                    # TODO: Remove this once we have a better way to handle this
                    if total_expanded > self.config.max_rules_applied and False:
                        break

            # TODO: Remove this once we have a better way to handle this
            if total_expanded > self.config.max_rules_applied and False:
                break

        if not expanded and node_id == 1:
            raise StopIteration("\nThe target molecule was not expanded.")

    def _add_node(
        self,
        node_id: int,
        new_node: Node,
        policy_prob: float = None,
        rule_id: int = None,
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

        self.nodes[new_node_id] = new_node
        self.parents[new_node_id] = node_id
        self.redundant_children[new_node_id] = set()
        self.children[node_id].add(new_node_id)
        self.children[new_node_id] = set()
        self.nodes_visit[new_node_id] = 0
        self.nodes_prob[new_node_id] = policy_prob
        self.nodes_rules[new_node_id] = rule_id
        self.nodes_depth[new_node_id] = self.nodes_depth[node_id] + 1
        self.curr_tree_size += 1

        if self.config.search_strategy == "evaluation_first":
            node_value = self._get_node_value(new_node_id)
        elif self.config.search_strategy == "expansion_first":
            node_value = self.init_node_value

        self.nodes_init_value[new_node_id] = node_value
        self.nodes_total_value[new_node_id] = node_value

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
            nodes_depth=self.nodes_depth,
            nodes_prob=self.nodes_prob,
        )
        return node_value

    def _log_final_stats(self, reason: str = "completed") -> None:
        """Logs final tree statistics after search completes.

        :param reason: Reason for stopping (e.g., "iterations limit", "time limit").
        """
        max_depth = max(self.nodes_depth.values()) if self.nodes_depth else 0
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
            self.nodes_visit[node_id] += 1
            node_id = self.parents[node_id]

    def report(self) -> str:
        """Returns the string representation of the tree."""

        return (
            f"Tree for: {str(self.nodes[1].precursors_to_expand[0])}\n"
            f"Time: {round(self.curr_time, 1)} seconds\n"
            f"Number of nodes: {len(self)}\n"
            f"Number of iterations: {self.curr_iteration}\n"
            f"Number of visited nodes: {len(self.visited_nodes)}\n"
            f"Number of found routes: {len(self.winning_nodes)}"
        )

    def route_score(self, node_id: int) -> float:
        """Calculates the score of a given route from the current node to the root node.
        The score depends on cumulated node values nad the route length.

        :param node_id: The id of the current given node.
        :return: The route score.
        """

        cumulated_nodes_value, route_length = 0, 0
        while node_id:
            route_length += 1

            cumulated_nodes_value += self.nodes_total_value[node_id]
            node_id = self.parents[node_id]

        return cumulated_nodes_value / (route_length**2)

    def route_to_node(self, node_id: int) -> List[Node,]:
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

    def synthesis_route(self, node_id: int) -> Tuple[Reaction,]:
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
            for before, after in zip(nodes, nodes[1:])
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
            assert (
                current_node_id not in visited_nodes
            ), "Error: The tree may not be circular!"
            node_visit = self.nodes_visit[current_node_id]

            visited_nodes.add(current_node_id)
            if self.children[current_node_id]:
                # Nodes
                children = [
                    child
                    for child in list(self.children[current_node_id])
                    if self.nodes_visit[child] >= visits_threshold
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
            node_value = round(self.nodes_total_value[node_id], 3)

            node_synthesisability = round(self.nodes_init_value[node_id])

            visit_in_node = self.nodes_visit[node_id]
            meta[node_id] = (node_value, node_synthesisability, visit_in_node)

        return newick_string, meta
