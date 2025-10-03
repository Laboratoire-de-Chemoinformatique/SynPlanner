"""Module containing a class Tree that used for tree search of retrosynthetic routes."""

import logging
import warnings
from collections import defaultdict, deque
from math import sqrt
from random import choice, uniform
from time import time, sleep
from typing import Dict, List, Set, Tuple

from CGRtools.reactor import Reactor
from CGRtools.containers import MoleculeContainer
from tqdm.auto import tqdm

from synplan.chem.precursor import Precursor
from synplan.chem.reaction import Reaction, apply_reaction_rule
from synplan.mcts.evaluation import ValueNetworkFunction
from synplan.mcts.expansion import PolicyNetworkFunction
from synplan.mcts.node import Node
from synplan.utils.config import TreeConfig

from rdkit.Contrib.SA_Score import sascorer
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
from multiprocessing import Pool, Array, Manager, Queue, Process
import sys
import torch
import psutil
import copy

manager = Manager()
to_predict_queue = Queue()
predicted_queue = Queue()
to_expand_queue = Queue()
expanded_queue = Queue()

def expand_node_worker(arg_curr_node_curr_precursor_molecule, arg_rule, arg_rule_id, arg_prob,arg_min_mol_size):
    new_precursors = []
    scaled_probs = []
    for products in apply_reaction_rule(arg_curr_node_curr_precursor_molecule, arg_rule):
        if not products :
            continue
        for molecule in products:
            molecule.meta["reactor_id"] = arg_rule_id

        new_precursor = tuple(Precursor(mol) for mol in products)
        scaled_prob = arg_prob * len(
            list(filter(lambda x: len(x) > arg_min_mol_size, products))
        )
        new_precursors.append(new_precursor)
        scaled_probs.append(scaled_prob)

    return new_precursors, scaled_probs, arg_rule_id


def expand_node_apply_rules_worker(arg_curr_node_curr_precursor_molecule, arg_rules):
    list_products = []
    for products in apply_reaction_rule(arg_curr_node_curr_precursor_molecule, arg_rules):
        list_products.append(products)



    return list_products

def expand_node_rest_worker(arg_products, arg_rule_id, arg_prob,arg_min_mol_size):

    for molecule in arg_products:
        molecule.meta["reactor_id"] = arg_rule_id

    new_precursor = tuple(Precursor(mol) for mol in arg_products)
    scaled_prob = arg_prob * len(
        list(filter(lambda x: len(x) > arg_min_mol_size, arg_products))
    )


    return new_precursor, scaled_prob, arg_rule_id



class Tree:
    """Tree class with attributes and methods for Monte-Carlo tree search."""

    def __init__(
        self,
        target: MoleculeContainer,
        config: TreeConfig,
        reaction_rules: List[Reactor],
        building_blocks: Set[str],
        expansion_function: PolicyNetworkFunction,
        evaluation_function: ValueNetworkFunction = None,
    ):
        """Initializes a tree object with optional parameters for tree search for target
        molecule.

        :param target: A target molecule for retrosynthetic routes search.
        :param config: A tree configuration.
        :param reaction_rules: A loaded reaction rules.
        :param building_blocks: A loaded building blocks.
        :param expansion_function: A loaded policy function.
        :param evaluation_function: A loaded value function. If None, the rollout is
            used as a default for node evaluation.
        """

        # config parameters
        self.config = config

        assert isinstance(
            target, MoleculeContainer
        ), "Target should be given as MoleculeContainer"
        assert len(target) > 3, "Target molecule has less than 3 atoms"

        target_molecule = Precursor(target)
        target_molecule.prev_precursors.append(Precursor(target))
        target_node = Node(
            precursors_to_expand=(target_molecule,), new_precursors=(target_molecule,)
        )

        # tree structure init
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

        # tree building limits
        self.curr_iteration: int = 0
        self.curr_tree_size: int = 2
        self.start_time: float = 0
        self.curr_time: float = 0

        # building blocks and reaction reaction_rules
        self.reaction_rules = reaction_rules
        self.building_blocks = building_blocks

        # policy and value functions
        self.policy_network = expansion_function
        if self.config.evaluation_type == "gcn":
            if evaluation_function is None:
                raise ValueError(
                    "Value function not specified while evaluation type is 'gcn'"
                )
            if (
                evaluation_function is not None
                and self.config.evaluation_type == "rollout"
            ):
                raise ValueError(
                    "Value function is not None while evaluation type is 'rollout'. What should  be evaluation type ?"
                )
            self.value_network = evaluation_function

        # utils
        self._tqdm = True  # needed to disable tqdm with multiprocessing module

        target_smiles = str(self.nodes[1].curr_precursor.molecule)
        if target_smiles in self.building_blocks:
            self.building_blocks.remove(target_smiles)
            print(
                "Target was found in building blocks and removed from building blocks."
            )

        # other tree search algorithms
        self.stop_at_first = False
        self.found_a_route = False
        self.bfs_table = [] #(node_id, score, depth)
        self.lnmcs_thresholds = [[] for _ in range(100)] #list of scores at depth
        self.pool = Pool(10)
        self.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks = {}
        self.big_dict_of_all_node_ids_NMCS_playout_values = {}

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
        if self.stop_at_first and self.found_a_route :
            raise StopIteration("Already found a route.")

        # start new iteration
        self.curr_iteration += 1
        self.curr_time = time() - self.start_time

        if self._tqdm:
            self._tqdm.update()
        if self.config.algorithm == "UCT":
            curr_depth, node_id = 0, 1  # start from the root node_id

            explore_route = True
            while explore_route:
                self.visited_nodes.add(node_id)

                if self.nodes_visit[node_id]:  # already visited
                    if not self.children[node_id]:  # dead node
                        self._update_visits(node_id)
                        explore_route = False
                    else:
                        node_id = self._select_node(node_id)  # select the child node
                        curr_depth += 1
                else:
                    if self.nodes[node_id].is_solved():  # found route
                        self._update_visits(
                            node_id
                        )  # this prevents expanding of bb node_id
                        if not node_id in self.winning_nodes:
                            self.winning_nodes.append(node_id)
                        self.found_a_route = True
                        return True, [node_id]

                    if (
                        curr_depth < self.config.max_depth
                    ):  # expand node if depth limit is not reached
                        self._expand_node(node_id)
                        self.expanded_nodes.add(node_id)
                        if not self.children[node_id]:  # node was not expanded
                            value_to_backprop = -1.0
                        else:
                            self.expanded_nodes.add(node_id)

                            if self.config.search_strategy == "evaluation_first":
                                # recalculate node value based on children synthesisability and backpropagation
                                child_values = [
                                    self.nodes_init_value[child_id]
                                    for child_id in self.children[node_id]
                                ]

                                if self.config.evaluation_agg == "max":
                                    value_to_backprop = max(child_values)

                                elif self.config.evaluation_agg == "average":
                                    value_to_backprop = sum(child_values) / len(
                                        self.children[node_id]
                                    )

                            elif self.config.search_strategy == "expansion_first":
                                value_to_backprop = self._get_node_value(node_id)

                        # backpropagation
                        self._backpropagate(node_id, value_to_backprop)
                        self._update_visits(node_id)
                        explore_route = False

                        if self.children[node_id]:
                            # found after expansion
                            found_after_expansion = set()
                            for child_id in iter(self.children[node_id]):
                                if self.nodes[child_id].is_solved():
                                    found_after_expansion.add(child_id)
                                    if not child_id in self.winning_nodes:
                                        self.winning_nodes.append(child_id)

                            if found_after_expansion:
                                self.found_a_route = True
                                return True, list(found_after_expansion)

                    else:
                        self._backpropagate(node_id, self.nodes_total_value[node_id])
                        self._update_visits(node_id)
                        explore_route = False

            return False, [node_id]


        if self.config.algorithm == "BEAM":
            nodes_to_open = []
            if self.bfs_table == []:
                if self.curr_iteration > 1 :
                    raise StopIteration("One deterministic beam search is enough")
                nodes_to_open = [(1, 0, 1)]
            else :
                width=10
                if width > len(self.bfs_table) :
                    width = len(self.bfs_table)
                for i in range(width) :
                    nodes_to_open.append(self.bfs_table[i])
            self.bfs_table = []

            for node in nodes_to_open:
                depth = node[2]
                leaf_id = node[0]
                self._expand_node(leaf_id)
                self.expanded_nodes.add(leaf_id)

                if self.children[leaf_id] and depth < self.config.max_depth:
                    for child_id in self.children[leaf_id]:
                        if self.nodes[child_id].is_solved():
                            if not child_id in self.winning_nodes:
                                self.winning_nodes.append(child_id)
                            self.found_a_route = True
                            return True, [child_id]

                        if self.config.evaluation_type == "score":
                            self.insert_dicho_bfs_table(child_id, self._get_node_value(child_id), depth + 1, False)

                        if self.config.evaluation_type == "rollout":
                            nodeFinal, seq = self.nmcs_rollout(child_id, depth + 1, "greedy")
                            if self.nodes[nodeFinal].is_solved():
                                if not nodeFinal in self.winning_nodes:
                                    self.winning_nodes.append(nodeFinal)
                                self.found_a_route = True
                                return True, [nodeFinal]
                            self.insert_dicho_bfs_table(child_id, self._get_node_value(nodeFinal), depth + 1, False)
            return False, [1]



        if self.config.algorithm == "BREADTH":
            if self.bfs_table == []:
                leaf_id = 1
                depth = 1
                if self.curr_iteration > 2:
                    raise StopIteration("breadth exhausted the tree")
            else :
                depth = self.bfs_table[0][2]
                leaf_id = self.bfs_table.pop(0)[0]

            if self.nodes[leaf_id].is_solved():
                if not leaf_id in self.winning_nodes:
                    self.winning_nodes.append(leaf_id)
                self.found_a_route = True
                return  True, [leaf_id]
            self._expand_node(leaf_id)
            self.expanded_nodes.add(leaf_id)
            if self.children[leaf_id] and depth < self.config.max_depth:
                for child_id in self.children[leaf_id]:
                    if self.nodes[child_id].is_solved():
                        if not child_id in self.winning_nodes:
                            self.winning_nodes.append(child_id)
                        self.found_a_route = True
                        return True, [child_id]
                    self.bfs_table.append((child_id, 0, depth+1))

            return False, [leaf_id]



        if self.config.algorithm == "BFS":
            if self.bfs_table == []:
                leaf_id = 1
                depth = 1
                if self.curr_iteration > 1:
                    raise StopIteration("BFS exhausted the tree")
            else :
                depth = self.bfs_table[0][2]
                leaf_id = self.bfs_table.pop(0)[0]

            if self.nodes[leaf_id].is_solved():
                if not leaf_id in self.winning_nodes:
                    self.winning_nodes.append(leaf_id)
                self.found_a_route = True
                return  True, [leaf_id]
            self._expand_node(leaf_id, depth = depth)
            self.expanded_nodes.add(leaf_id)
            if self.children[leaf_id] and depth < self.config.max_depth:
                for child_id in self.children[leaf_id]:
                    if self.nodes[child_id].is_solved():
                        if not child_id in self.winning_nodes:
                            self.winning_nodes.append(child_id)
                        self.found_a_route = True
                        return True, [child_id]

                    if self.config.evaluation_type == "score" :
                        self.insert_dicho_bfs_table(child_id, self._get_node_value(child_id), depth+1, False)

                    if self.config.evaluation_type == "rollout" :
                        nodeFinal, seq = self.nmcs_rollout(child_id, depth + 1, "greedy" )
                        if self.nodes[nodeFinal].is_solved():
                            if not nodeFinal in self.winning_nodes:
                                self.winning_nodes.append(nodeFinal)
                            self.found_a_route = True
                            return True, [nodeFinal]
                        self.insert_dicho_bfs_table(child_id, self._get_node_value(nodeFinal), depth +1, False)
            return False, [leaf_id]

        if self.config.algorithm == "paBFS":
            if self.bfs_table == []:
                leaf_id = 1
                depth = 1
                self._expand_node(1)
                self.expanded_nodes.add(1)
                if self.curr_iteration > 1:
                    raise StopIteration("BFS exhausted the tree")
            else :
                to_expand = []
                to_expand_bfstable_index = []
                for i in range(min(150, len(self.bfs_table))):
                    if not self.bfs_table[i][3] :
                        to_expand.append(self.bfs_table[i][0])
                        to_expand_bfstable_index.append(i)
                        if len(to_expand) >= 100 :
                            break
                if len(to_expand) >= min(100, len(self.bfs_table)) :
                    self._expand_multiple_nodes(to_expand)
                    for i in to_expand_bfstable_index :
                        self.bfs_table[i][3] = True


                leaf_id = -1
                for i in range(min(150, len(self.bfs_table))) :
                    if self.bfs_table[i][3]:
                        depth = self.bfs_table[i][2]
                        leaf_id = self.bfs_table.pop(i)[0]
                        break




            if self.children[leaf_id] and depth < self.config.max_depth:
                for child_id in self.children[leaf_id]:
                    if self.nodes[child_id].is_solved():
                        if not child_id in self.winning_nodes:
                            self.winning_nodes.append(child_id)
                        self.found_a_route = True
                        return True, [child_id]

                    if self.config.evaluation_type == "score" :
                        self.insert_dicho_bfs_table(child_id, self._get_node_value(child_id), depth+1, False)

                    if self.config.evaluation_type == "rollout" :
                        nodeFinal, seq = self.nmcs_rollout(child_id, depth + 1, "greedy" )
                        if self.nodes[nodeFinal].is_solved():
                            if not nodeFinal in self.winning_nodes:
                                self.winning_nodes.append(nodeFinal)
                            self.found_a_route = True
                            return True, [nodeFinal]
                        self.insert_dicho_bfs_table(child_id, self._get_node_value(nodeFinal), depth +1, False)
            return False, [leaf_id]

        if self.config.algorithm == "BDFS":
            if self.bfs_table == []:
                leaf_id = 1
                depth = 1
            else :
                depth = self.bfs_table[0][2]
                leaf_id = self.bfs_table.pop(0)[0]

            if self.nodes[leaf_id].is_solved():
                if not leaf_id in self.winning_nodes:
                    self.winning_nodes.append(leaf_id)
                self.found_a_route = True
                return  True, [leaf_id]
            self._expand_node(leaf_id)
            self.expanded_nodes.add(leaf_id)
            while self.children[leaf_id] and depth < self.config.max_depth:
                chil = []
                best_sc = 0
                best_chil_id = -1
                for child_id in self.children[leaf_id]:
                    if self.nodes[child_id].is_solved():
                        if not child_id in self.winning_nodes:
                            self.winning_nodes.append(child_id)
                        self.found_a_route = True
                        return True, [child_id]

                    chil.append(child_id)
                    sc = self._get_node_value(child_id)

                    if sc > best_sc :
                        best_sc = sc
                        best_chil_id = child_id

                for child_id in chil:
                    if child_id != best_chil_id :
                        self.insert_dicho_bfs_table(child_id, self._get_node_value(child_id), depth+1, False)
                depth += 1
                leaf_id = best_chil_id

            return False, [leaf_id]

        if self.config.algorithm == "NMCS":
            if self.curr_iteration > 1:
                raise StopIteration("One deterministic NMCS iteration is enough.")
            node_id = 1
            leaf, _ = self.NMCS(node_id, self.config.NMCS_level, 1)
            if self.nodes[leaf].is_solved() :
                if not leaf in self.winning_nodes:
                    self.winning_nodes.append(leaf)
                self.found_a_route = True
                return True, [leaf]
            return False, [leaf]

        if self.config.algorithm == "LNMCS":
            self.bfs_table = []
            self.lnmcs_thresholds = [[] for _ in range(100)]
            if self.curr_iteration > 1:
                raise StopIteration("One deterministic LNMCS iteration is enough.")
            node_id = 1
            leaf, _ = self.LNMCS(node_id, self.config.NMCS_level, 1, self.config.LNMCS_ratio)
            if self.nodes[leaf].is_solved() :
                if not leaf in self.winning_nodes:
                    self.winning_nodes.append(leaf)
                self.found_a_route = True
                return True, [leaf]
            return False, [leaf]

        if self.config.algorithm == "SCATTER":
            if self.curr_iteration > 1:
                raise StopIteration("One deterministic SCATTER iteration is enough.")
            open_nodes = [1]
            for i in range(self.config.NMCS_level) :
                open_nodes_temp = []
                for n in open_nodes:
                    if len(open_nodes_temp) > 1000 :
                        open_nodes_temp = open_nodes + open_nodes_temp
                        break
                    if self.nodes[n].is_solved():
                        if not n in self.winning_nodes:
                            self.winning_nodes.append(n)
                        self.found_a_route = True
                        return True, [n]
                    if not n in self.expanded_nodes:
                        self._expand_node(n)
                        self.expanded_nodes.add(n)
                    for child_id in self.children[n]:
                        open_nodes_temp.append(child_id)
                open_nodes = open_nodes_temp

            for n in open_nodes:
                if time() - self.start_time > self.config.max_time:
                    return False, [0]
                if self.nodes[n].is_solved():
                    if not n in self.winning_nodes:
                        self.winning_nodes.append(n)
                    self.found_a_route = True
                    return True, [n]
                if not n in self.expanded_nodes:
                    self._expand_node(n)
                    self.expanded_nodes.add(n)
                s, _ = self.nmcs_rollout(n, self.config.NMCS_level+1, "greedy")
                if self.nodes[s].is_solved():
                    if not n in self.winning_nodes:
                        self.winning_nodes.append(n)
                    self.found_a_route = True
                    return True, [s]
            return False, [0]



        print("unknown algorithm, please edit the configuration with a known algorithm: UCT, BFS, BDFS, NMCS, LNMCS, BREADTH, BEAM")
        return False, [0]

    def _ucb(self, node_id: int) -> float:
        """Calculates the Upper Confidence Bound (UCB) statistics for a given node.

        :param node_id: The id of the node.
        :return: The calculated UCB.
        """

        prob = self.nodes_prob[node_id]  # predicted by policy network score
        visit = self.nodes_visit[node_id]

        if self.config.ucb_type == "puct":
            u = (
                self.config.c_ucb * prob * sqrt(self.nodes_visit[self.parents[node_id]])
            ) / (visit + 1)
            ucb_value = self.nodes_total_value[node_id] + u

        if self.config.ucb_type == "uct":
            u = (
                self.config.c_ucb
                * sqrt(self.nodes_visit[self.parents[node_id]])
                / (visit + 1)
            )
            ucb_value = self.nodes_total_value[node_id] + u

        if self.config.ucb_type == "value":
            ucb_value = self.nodes_init_value[node_id] / (visit + 1)

        return ucb_value

    def _select_node(self, node_id: int) -> int:
        """Selects a node based on its UCB value and returns the id of the node with the
        highest UCB.

        :param node_id: The id of the node.
        :return: The id of the node with the highest UCB.
        """

        if self.config.epsilon > 0:
            n = uniform(0, 1)
            if n < self.config.epsilon:
                return choice(list(self.children[node_id]))

        best_score, best_children = None, []
        for child_id in self.children[node_id]:
            score = self._ucb(child_id)
            if best_score is None or score > best_score:
                best_score, best_children = score, [child_id]
            elif score == best_score:
                best_children.append(child_id)

        # is needed for tree search reproducibility, when all child nodes has the same score
        return best_children[0]

    def _expand_node(self, node_id: int) -> None:
        """Expands the node by generating new precursor with policy (expansion) function.

        :param node_id: The id the node to be expanded.
        :return: None.
        """
        total_expanded = 0
        curr_node = self.nodes[node_id]
        prev_precursor = curr_node.curr_precursor.prev_precursors

        tmp_precursor = set()
        expanded = False
        SINGLE_CORE = self.config.single_core
        SINGLE_WORKER = True
        args_to_launch_single = []
        args_to_launch = []
        args_to_launch2_part1 = []
        args_to_launch2 = []

        prediction = self.policy_network.predict_reaction_rules(
            curr_node.curr_precursor, self.reaction_rules
        )

        for prob, rule, rule_id in prediction :
            if SINGLE_CORE :
                for products in apply_reaction_rule(curr_node.curr_precursor.molecule, rule):

                    # check repeated products
                    if not products or not set(products) - tmp_precursor:
                        continue
                    tmp_precursor.update(products)

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

                        child_node = Node(
                            precursors_to_expand=precursors_to_expand,
                            new_precursors=new_precursor,
                        )


                        for new_precursor in new_precursor:
                            new_precursor.prev_precursors = [new_precursor, *prev_precursor]


                        self._add_node(node_id, child_node, scaled_prob, rule_id)
                        total_expanded += 1
                        expanded = True

                        if total_expanded > self.config.max_rules_applied and False:
                            break
                    if total_expanded > self.config.max_rules_applied and False:
                        break
            else:
                if SINGLE_WORKER:
                    args_to_launch_single.append((curr_node.curr_precursor.molecule, rule, rule_id, prob, self.config.min_mol_size))
                else:
                    args_to_launch.append((curr_node.curr_precursor.molecule, rule))
                    args_to_launch2_part1.append((rule_id, prob, self.config.min_mol_size))

        if not SINGLE_CORE :
            if SINGLE_WORKER:
                res = self.pool.starmap(expand_node_worker, args_to_launch_single)
                for r in res:
                    for i in range(len(r[0])):
                        new_precursors = r[0][i]

                        temp = [p for p in new_precursors if not p.is_building_block(self.building_blocks, self.config.min_mol_size)]
                        if not new_precursors or (not set(temp) - tmp_precursor and temp):
                            continue
                        else:
                            tmp_precursor.update(temp)


                        if set(prev_precursor).isdisjoint(new_precursors):
                            precursors_to_expand = (
                                *curr_node.next_precursor,
                                *(
                                    x
                                    for x in new_precursors
                                    if not x.is_building_block(self.building_blocks, self.config.min_mol_size)
                                ),
                            )

                            if precursors_to_expand != () and precursors_to_expand in self.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks:
                                id = self.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks[precursors_to_expand]
                                self.redundant_children[node_id].add(id)
                                total_expanded += 1
                                expanded = True
                                if total_expanded > self.config.max_rules_applied and False:
                                    break
                                continue

                            else:
                                self.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks[precursors_to_expand] = self.curr_tree_size


                            child_node = Node(precursors_to_expand=precursors_to_expand, new_precursors=new_precursors)

                            for new_precursor in new_precursors:
                                new_precursor.prev_precursors = [new_precursor, *prev_precursor]

                            self._add_node(node_id, child_node, r[1][i], r[2])
                            total_expanded += 1
                            expanded = True
                            if total_expanded > self.config.max_rules_applied and False :
                                break

                    if total_expanded > self.config.max_rules_applied and False :
                        break
            else:
                list_list_products = self.pool.starmap(expand_node_apply_rules_worker, args_to_launch)

                for i in range(len(list_list_products)):
                    for j in range(len(list_list_products[i])):

                        products = list_list_products[i][j]
                        if not products or not set(products) - tmp_precursor:
                            continue
                        else:
                            tmp_precursor.update(products)
                            args_to_launch2.append((products, args_to_launch2_part1[i][0], args_to_launch2_part1[i][1], args_to_launch2_part1[i][2]))

                res = self.pool.starmap(expand_node_rest_worker, args_to_launch2)

                for r in res:
                    if set(prev_precursor).isdisjoint(r[0]):
                        precursors_to_expand = (
                            *curr_node.next_precursor,
                            *(
                                x
                                for x in r[0]
                                if not x.is_building_block(
                                self.building_blocks, self.config.min_mol_size
                            )
                            ),
                        )

                        child_node = Node(
                            precursors_to_expand=precursors_to_expand,
                            new_precursors=r[0],
                        )

                        for new_precursor in r[0]:
                            new_precursor.prev_precursors = [new_precursor, *prev_precursor]

                        self._add_node(node_id, child_node, r[1], r[0])
                        total_expanded += 1
                        expanded = True

                        if total_expanded > self.config.max_rules_applied and False:
                            break

                    if total_expanded > self.config.max_rules_applied and False:
                        break

        if not expanded and node_id == 1:
            raise StopIteration("\nThe target molecule was not expanded.")


    def _expand_multiple_nodes(self, nodes_ids: [int]) -> None:
        """Expands the nodes by generating new precursor with policy (expansion) function.

        :param nodes_ids: The ids the nodes to be expanded.
        :return: None.
        """

        for node_id in nodes_ids:
            curr_node = self.nodes[node_id]
            to_predict_queue.put((curr_node.curr_precursor, len(self.reaction_rules), node_id))
        total_predicted = 0
        total_to_expand = 0
        while total_predicted != len(nodes_ids) :
            if not predicted_queue.empty():

                pred_res = predicted_queue.get()
                total_predicted += 1
                node_id = pred_res[2]
                curr_node = self.nodes[node_id]
                for j in range(len(pred_res[0])):
                    total_to_expand +=1
                    prob = pred_res[0][j]
                    rule_id = pred_res[1][j]
                    rule = self.reaction_rules[rule_id]
                    to_expand_queue.put((curr_node.curr_precursor.molecule, rule, rule_id, prob, self.config.min_mol_size, node_id))

        total_expanded = 0

        while total_expanded != total_to_expand:
            if not expanded_queue.empty() :
                total_expanded +=1
                res = expanded_queue.get()
                rule_id = res[2]
                node_id = res[3]
                curr_node = self.nodes[node_id]
                prev_precursor = curr_node.curr_precursor.prev_precursors
                tmp_precursor = set()

                for i in range(len(res[0])):
                    scaled_prob = res[1][i]
                    new_precursors = res[0][i]

                    temp = [p for p in new_precursors if not p.is_building_block(self.building_blocks, self.config.min_mol_size)]
                    if not new_precursors or (not set(temp) - tmp_precursor and temp ):
                        continue
                    else:
                        tmp_precursor.update(temp)

                    if set(prev_precursor).isdisjoint(new_precursors):
                        precursors_to_expand = (
                            *curr_node.next_precursor,
                            *(
                                x
                                for x in new_precursors
                                if not x.is_building_block(
                                self.building_blocks, self.config.min_mol_size
                            )
                            ),
                        )

                        child_node = Node(
                            precursors_to_expand=precursors_to_expand,
                            new_precursors=new_precursors,
                        )

                        for new_precursor in new_precursors:
                            new_precursor.prev_precursors = [new_precursor, *prev_precursor]
                        self._add_node(node_id, child_node, scaled_prob, rule_id)

        for node_id in nodes_ids:
            self.expanded_nodes.add(node_id)


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
            node_value = self.config.init_node_value

        self.nodes_init_value[new_node_id] = node_value
        self.nodes_total_value[new_node_id] = node_value

    def _get_node_value(self, node_id: int) -> float:
        """Calculates the value for the given node (for example with rollout or value
        network).

        :param node_id: The id of the node to be evaluated.
        :return: The estimated value of the node.
        """

        node = self.nodes[node_id]

        if self.config.score_function == "random":
            node_value = uniform(0, 1)

        elif self.config.score_function == "rollout":
            node_value = min(
                (
                    self._rollout_node(
                        precursor, current_depth=self.nodes_depth[node_id]
                    )
                    for precursor in node.precursors_to_expand
                ),
                default=1.0,
            )

        elif self.config.score_function == "sascore":
            meanPrecursorSAS = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    meanPrecursorSAS += sascorer.calculateScore(m)
                except:
                    meanPrecursorSAS += 10.0
            meanPrecursorSAS = meanPrecursorSAS / len(node.precursors_to_expand)
            node_value = 1.0-meanPrecursorSAS / 10.0

        elif self.config.score_function == "policy":
            node_value = self.nodes_prob[node_id]

        elif self.config.score_function == "heavyAtomCount":
            totalHeavy = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    totalHeavy += CalcNumHeavyAtoms(m)
                except:
                    totalHeavy += 100.0

            node_value = 1000 - totalHeavy
            if node_value < 0:
                node_value = 0

        elif self.config.score_function == "weight":
            totalWeight = 0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    totalWeight += ExactMolWt(m)
                except:
                    totalWeight += 1000.0

            node_value = 10000 - totalWeight
            if node_value < 0:
                node_value = 0

        elif self.config.score_function == "weightXsascore":
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    total += ExactMolWt(m) * sascorer.calculateScore(m)
                except:
                    total += 10000.0
            if total == 0:
                return 1
            node_value = 1 / total
            if node_value < 0:
                node_value = 0

        elif self.config.score_function == "WxWxSAS":
            total = 0.0
            for p in node.precursors_to_expand:
                try:
                    m = Chem.MolFromSmiles(str(p.molecule))
                    total += ExactMolWt(m) ** 2 * sascorer.calculateScore(m)
                except:
                    total += 10000.0
            if total == 0:
                return 1
            node_value = 1 / total
            if node_value < 0:
                node_value = 0

        elif self.config.score_function == "gcn":
            node_value = self.value_network.predict_value(node.new_precursors)

        return node_value

    def _update_visits(self, node_id: int) -> None:
        """Updates the number of visits from the current node to the root node.

        :param node_id: The id of the current node.
        :return: None.
        """

        while node_id:
            self.nodes_visit[node_id] += 1
            node_id = self.parents[node_id]

    def _backpropagate(self, node_id: int, value: float) -> None:
        """Backpropagates the value through the tree from the current.

        :param node_id: The id of the node from which to backpropagate the value.
        :param value: The value to backpropagate.
        :return: None.
        """
        while node_id:
            if self.config.backprop_type == "muzero":
                self.nodes_total_value[node_id] = (
                    self.nodes_total_value[node_id] * self.nodes_visit[node_id] + value
                ) / (self.nodes_visit[node_id] + 1)
            elif self.config.backprop_type == "cumulative":
                self.nodes_total_value[node_id] += value
            node_id = self.parents[node_id]

    def _rollout_node(self, precursor: Precursor, current_depth: int = None) -> float:
        """Performs a rollout simulation from a given node in the tree. Given the
        current precursor, find the first successful reaction and return the new precursor.

        If the precursor is a building_block, return 1.0, else check the
        first successful reaction.

        If the reaction is not successful, return -1.0.

        If the reaction is successful, but the generated precursor are not
        the building_blocks and the precursor cannot be generated without
        exceeding current_depth threshold, return -0.5.

        If the reaction is successful, but the precursor are not the
        building_blocks and the precursor cannot be generated, return
        -1.0.

        :param precursor: The precursor to be evaluated.
        :param current_depth: The current depth of the tree.
        :return: The reward (value) assigned to the precursor.
        """

        max_depth = self.config.max_depth - current_depth

        # precursor checking
        if precursor.is_building_block(self.building_blocks, self.config.min_mol_size):
            return 1.0

        if max_depth == 0:
            print("max depth reached in the beginning")

        # precursor simulating
        occurred_precursor = set()
        precursor_to_expand = deque([precursor])
        history = defaultdict(dict)
        rollout_depth = 0
        while precursor_to_expand:
            # Iterate through reactors and pick first successful reaction.
            # Check products of the reaction if you can find them in in-building_blocks data
            # If not, then add missed products to precursor_to_expand and try to decompose them
            if len(history) >= max_depth:
                reward = -0.5
                return reward

            current_precursor = precursor_to_expand.popleft()
            history[rollout_depth]["target"] = current_precursor
            occurred_precursor.add(current_precursor)

            # Pick the first successful reaction while iterating through reactors
            reaction_rule_applied = False
            for prob, rule, rule_id in self.policy_network.predict_reaction_rules(
                current_precursor, self.reaction_rules
            ):
                for products in apply_reaction_rule(current_precursor.molecule, rule):
                    if products:
                        reaction_rule_applied = True
                        break

                if reaction_rule_applied:
                    history[rollout_depth]["rule_index"] = rule_id
                    break

            if not reaction_rule_applied:
                reward = -1.0
                return reward

            products = tuple(Precursor(product) for product in products)
            history[rollout_depth]["products"] = products

            # check loops
            if any(x in occurred_precursor for x in products) and products:
                # sometimes manual can create a loop, when
                # print('occurred_precursor')
                reward = -1.0
                return reward

            if occurred_precursor.isdisjoint(products):
                # added number of atoms check
                precursor_to_expand.extend(
                    [
                        x
                        for x in products
                        if not x.is_building_block(
                            self.building_blocks, self.config.min_mol_size
                        )
                    ]
                )
                rollout_depth += 1

        reward = 1.0
        return reward

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

    def insert_dicho_bfs_table(self, elt, value, depth, expanded):
        if len(self.bfs_table) == 0:
            self.bfs_table.append([elt, value, depth, expanded])
            return
        i1 = 0
        i2 = len(self.bfs_table)-1
        i = len(self.bfs_table)//2
        while i1 != i2 and i != i1 and i!=i2:
            if value < self.bfs_table[i][1]:
                i1 = i+1
            if value > self.bfs_table[i][1]:
                i2 = i
            if value == self.bfs_table[i][1]:
                self.bfs_table.insert(i, [elt, value, depth, expanded])
                return
            i = (i1+i2)//2
        self.bfs_table.insert(i, [elt, value, depth, expanded])
        return
    def insert_dicho_LNMCS_tresh(self, value, depth):
        if len(self.lnmcs_thresholds[depth]) == 0:
            self.lnmcs_thresholds[depth].append(value)
            return
        i1 = 0
        i2 = len(self.lnmcs_thresholds[depth])-1
        i = len(self.lnmcs_thresholds[depth])//2
        while i1 != i2 and i != i1 and i!=i2:
            if value < self.lnmcs_thresholds[depth][i]:
                i1 = i+1
            if value > self.lnmcs_thresholds[depth][i]:
                i2 = i
            if value == self.lnmcs_thresholds[depth][i]:
                self.lnmcs_thresholds[depth].insert(i, value)
                return
            i = (i1+i2)//2
        self.lnmcs_thresholds[depth].insert(i, value)
        return

    def NMCS(self, node_id, n, depth):
        best_node_id = node_id
        best_sequence = []
        return_sequence = []
        if not node_id in self.expanded_nodes :
            self._expand_node(node_id)
            self.expanded_nodes.add(node_id)
        best_score = -100000
        all_child = self.children[node_id].union(self.redundant_children[node_id])
        while all_child :
            if time() - self.start_time > self.config.max_time :
                return best_node_id, return_sequence + best_sequence
            if self.nodes[node_id].is_solved():
                if not node_id in self.winning_nodes:
                    self.winning_nodes.append(node_id)
                self.found_a_route = True
                if self.stop_at_first :
                    return node_id, return_sequence

            for child_id in all_child:
                s1 = child_id
                if self.nodes[child_id].is_solved():
                    if not child_id in self.winning_nodes:
                        self.winning_nodes.append(child_id)
                    self.found_a_route = True
                    return_sequence.append(s1)
                    if self.stop_at_first:
                        return s1, return_sequence
                    continue
                if not child_id in self.expanded_nodes:
                    self._expand_node(child_id)
                    self.expanded_nodes.add(child_id)
                if n == 1 :
                    sequence = []
                    if self.config.evaluation_type == "score" :
                        sequence = []
                    if self.config.evaluation_type == "rollout" :
                        if s1 in self.big_dict_of_all_node_ids_NMCS_playout_values :
                            (s1, sequence) = self.big_dict_of_all_node_ids_NMCS_playout_values[s1]
                        else:
                            s1key = s1
                            s1, sequence = self.nmcs_rollout(s1, depth+1,"greedy")
                            self.big_dict_of_all_node_ids_NMCS_playout_values[s1key] = (s1, sequence)
                    sequence.insert(0, child_id)
                else:
                    s1, sequence = self.NMCS(s1, n-1, depth+1)
                    sequence.insert(0, child_id)
                if self.nodes[s1].is_solved():
                    if not s1 in self.winning_nodes:
                        self.winning_nodes.append(s1)
                    self.found_a_route = True
                    if self.stop_at_first:
                        return s1, return_sequence + sequence

                score = self._get_node_value(s1)

                if score > best_score:
                    best_sequence = sequence
                    best_score = score
                    best_node_id = s1
                if time() - self.start_time > self.config.max_time:
                    return best_node_id, return_sequence + best_sequence
            if len(best_sequence) == 0:
                return node_id, return_sequence
            node_id = best_sequence[0]
            all_child = self.children[node_id].union(self.redundant_children[node_id])
            depth += 1
            return_sequence.append(best_sequence.pop(0))

            if not node_id in self.expanded_nodes and not node_id in self.winning_nodes:
                self._expand_node(node_id)
                self.expanded_nodes.add(node_id)

        return best_node_id, return_sequence

    def LNMCS(self, node_id, n, depth, ratio):
        best_node_id = node_id
        best_sequence = []
        return_sequence = []
        if not node_id in self.expanded_nodes :
            self._expand_node(node_id)
            self.expanded_nodes.add(node_id)
        best_score = -100000
        while self.children[node_id] :
            if time() - self.start_time > self.config.max_time :
                return best_node_id, return_sequence + best_sequence
            if self.nodes[node_id].is_solved():
                if not node_id in self.winning_nodes:
                    self.winning_nodes.append(node_id)
                self.found_a_route = True
                if self.stop_at_first:
                    return node_id, return_sequence

            self.bfs_table = []
            candidates = []

            self.lnmcs_thresholds = [[] for _ in range(100)]

            for child_id in self.children[node_id]:
                if self.nodes[child_id].is_solved():
                    if not child_id in self.winning_nodes:
                        self.winning_nodes.append(child_id)
                    self.found_a_route = True
                    if self.stop_at_first:
                        return_sequence.append(child_id)
                        return child_id, return_sequence
                    continue
                if not child_id in self.expanded_nodes:
                    self._expand_node(child_id)
                    self.expanded_nodes.add(child_id)

                st_to_eval, seq = self.nmcs_rollout(child_id, depth + 1, "greedy")
                if self.nodes[st_to_eval].is_solved():
                    if not st_to_eval in self.winning_nodes:
                        self.winning_nodes.append(st_to_eval)
                    self.found_a_route = True
                    if self.stop_at_first:
                        return_sequence.append(st_to_eval)
                        return st_to_eval, return_sequence + seq
                eval_score = self._get_node_value(st_to_eval)

                self.insert_dicho_LNMCS_tresh(eval_score, depth)
                self.insert_dicho_bfs_table(child_id, eval_score, 0, False)

            best_can = -1
            max_score_can = -1
            for e in self.bfs_table :
                if e[1] > max_score_can :
                    max_score_can = e[1]
                    best_can = e[0]
                if e[1] >= self.lnmcs_thresholds[depth][int(ratio*(len(self.lnmcs_thresholds[depth])-1))] :
                    candidates.append(e[0])
            if best_can != -1 and candidates == []:
                candidates = [best_can]

            for child_id in candidates:
                s1 = child_id
                if n == 1 :
                    sequence = []
                    if self.config.evaluation_type == "score" :
                        sequence = []
                    if self.config.evaluation_type == "rollout" :
                        s1, sequence = self.nmcs_rollout(s1, depth+1, "policy")
                    sequence.insert(0, child_id)
                else:
                    s1, sequence = self.LNMCS(s1, n - 1, depth + 1, ratio)
                    sequence.insert(0, child_id)
                if self.nodes[s1].is_solved():
                    if not s1 in self.winning_nodes:
                        self.winning_nodes.append(s1)
                    self.found_a_route = True
                    if self.stop_at_first:
                        return s1, return_sequence + sequence

                score = self._get_node_value(s1)

                if score > best_score:
                    best_sequence = sequence
                    best_score = score
                    best_node_id = s1
                if time() - self.start_time > self.config.max_time:
                    return best_node_id, return_sequence + best_sequence
            if len(best_sequence) == 0:
                return node_id, return_sequence
            node_id = best_sequence[0]
            depth += 1
            return_sequence.append(best_sequence.pop(0))

            if not node_id in self.expanded_nodes and not node_id in self.winning_nodes:
                self._expand_node(node_id)
                self.expanded_nodes.add(node_id)

        return best_node_id, return_sequence

    def nmcs_rollout(self, node_id, node_depth, mode):
        depth = node_depth
        sequence = []
        if time() - self.start_time > self.config.max_time :
            return node_id, sequence
        if not node_id in self.expanded_nodes:
            self._expand_node(node_id)
            self.expanded_nodes.add(node_id)

        all_child = self.children[node_id].union(self.redundant_children[node_id])
        while all_child and depth < self.config.max_depth:

            if self.nodes[node_id].is_solved():
                if not node_id in self.winning_nodes:
                    self.winning_nodes.append(node_id)
                self.found_a_route = True
                return node_id, sequence
            max = -1
            i = -1
            if mode == "greedy" :
                vals = []
                for e in range(len(self.children[node_id])):
                    vals.append(0.00001)
                    if self.nodes[list(self.children[node_id])[e]].is_solved():
                        if not list(self.children[node_id])[e] in self.winning_nodes:
                            self.winning_nodes.append(list(self.children[node_id])[e])
                        self.found_a_route = True
                        sequence.append(list(self.children[node_id])[e])
                        return list(self.children[node_id])[e], sequence

                    vals[e] += self._get_node_value(list(self.children[node_id])[e])
                    if vals[e] > max :
                        max = vals[e]
                        i = e
            if mode == "random":
                i = int(uniform(0, len(self.children[node_id])))
                if self.nodes[list(self.children[node_id])[i]].is_solved():
                    if not list(self.children[node_id])[i] in self.winning_nodes:
                        self.winning_nodes.append(list(self.children[node_id])[i])
                    self.found_a_route = True
                    sequence.append(list(self.children[node_id])[i])
                    return list(self.children[node_id])[i], sequence
            if mode == "policy":
                vals = []
                for e in range(len(self.children[node_id])):
                    vals.append(0.00001)
                    if self.nodes[list(self.children[node_id])[e]].is_solved():
                        if not list(self.children[node_id])[e] in self.winning_nodes:
                            self.winning_nodes.append(list(self.children[node_id])[e])
                        self.found_a_route = True
                        sequence.append(list(self.children[node_id])[e])
                        return list(self.children[node_id])[e], sequence

                    vals[e] += self.nodes_prob[list(self.children[node_id])[e]]
                    if vals[e] > max:
                        max = vals[e]
                        i = e
            if i != -1:
                selected = list(self.children[node_id])[i]
                sequence.append(selected)
                node_id = selected
                all_child = self.children[node_id].union(self.redundant_children[node_id])
                if not node_id in self.expanded_nodes:
                    self._expand_node(node_id)
                    self.expanded_nodes.add(node_id)
            else:
                return node_id, sequence
            depth += 1
        return node_id, sequence

