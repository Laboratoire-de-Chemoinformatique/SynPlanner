from time import time, sleep
from random import choice, uniform
from math import sqrt

class BaseSearchAlgorithm:
    def __init__(self, tree):
        self.tree = tree  # reference to Tree, so algorithms can access nodes, config, etc.

    def step(self) -> tuple[bool, list[int]]:
        raise NotImplementedError

    def nmcs_rollout(self, node_id, node_depth, mode):
        depth = node_depth
        sequence = []
        if time() - self.tree.start_time > self.tree.config.max_time :
            return node_id, sequence
        if not node_id in self.tree.expanded_nodes:
            self.tree._expand_node(node_id)
            self.tree.expanded_nodes.add(node_id)

        all_child = self.tree.children[node_id].union(self.tree.redundant_children[node_id])
        while all_child and depth < self.tree.config.max_depth:

            if self.tree.nodes[node_id].is_solved():
                if not node_id in self.tree.winning_nodes:
                    self.tree.winning_nodes.append(node_id)
                self.tree.found_a_route = True
                return node_id, sequence
            max = -1
            i = -1
            if mode == "greedy" :
                vals = []
                for e in range(len(self.tree.children[node_id])):
                    vals.append(0.00001)
                    if self.tree.nodes[list(self.tree.children[node_id])[e]].is_solved():
                        if not list(self.tree.children[node_id])[e] in self.tree.winning_nodes:
                            self.tree.winning_nodes.append(list(self.tree.children[node_id])[e])
                        self.tree.found_a_route = True
                        sequence.append(list(self.tree.children[node_id])[e])
                        return list(self.tree.children[node_id])[e], sequence

                    vals[e] += self.tree._get_node_value(list(self.tree.children[node_id])[e])
                    if vals[e] > max :
                        max = vals[e]
                        i = e
            if mode == "random":
                i = int(uniform(0, len(self.tree.children[node_id])))
                if self.tree.nodes[list(self.tree.children[node_id])[i]].is_solved():
                    if not list(self.tree.children[node_id])[i] in self.tree.winning_nodes:
                        self.tree.winning_nodes.append(list(self.tree.children[node_id])[i])
                    self.tree.found_a_route = True
                    sequence.append(list(self.tree.children[node_id])[i])
                    return list(self.tree.children[node_id])[i], sequence
            if mode == "policy":
                vals = []
                for e in range(len(self.tree.children[node_id])):
                    vals.append(0.00001)
                    if self.tree.nodes[list(self.tree.children[node_id])[e]].is_solved():
                        if not list(self.tree.children[node_id])[e] in self.tree.winning_nodes:
                            self.tree.winning_nodes.append(list(self.tree.children[node_id])[e])
                        self.tree.found_a_route = True
                        sequence.append(list(self.tree.children[node_id])[e])
                        return list(self.tree.children[node_id])[e], sequence

                    vals[e] += self.tree.nodes_prob[list(self.tree.children[node_id])[e]]
                    if vals[e] > max:
                        max = vals[e]
                        i = e
            if i != -1:
                selected = list(self.tree.children[node_id])[i]
                sequence.append(selected)
                node_id = selected
                all_child = self.tree.children[node_id].union(self.tree.redundant_children[node_id])
                if not node_id in self.tree.expanded_nodes:
                    self.tree._expand_node(node_id)
                    self.tree.expanded_nodes.add(node_id)
            else:
                return node_id, sequence
            depth += 1
        return node_id, sequence


class BreadthFirstSearch(BaseSearchAlgorithm):
    def __init__(self, tree):
        super().__init__(tree)
        self.bfs_table = []  # (node_id, score, depth)

    def step(self):
        if self.bfs_table == []:
            leaf_id = 1
            depth = 1
            if self.tree.curr_iteration > 2:
                raise StopIteration("breadth exhausted the tree")
        else:
            depth = self.bfs_table[0][2]
            leaf_id = self.bfs_table.pop(0)[0]

        if self.tree.nodes[leaf_id].is_solved():
            if not leaf_id in self.tree.winning_nodes:
                self.tree.winning_nodes.append(leaf_id)
            self.tree.found_a_route = True
            return True, [leaf_id]
        self.tree._expand_node(leaf_id)
        self.tree.expanded_nodes.add(leaf_id)
        if self.tree.children[leaf_id] and depth < self.tree.config.max_depth:
            for child_id in self.tree.children[leaf_id]:
                if self.tree.nodes[child_id].is_solved():
                    if not child_id in self.tree.winning_nodes:
                        self.tree.winning_nodes.append(child_id)
                    self.tree.found_a_route = True
                    return True, [child_id]
                self.bfs_table.append((child_id, 0, depth + 1))

        return False, [leaf_id]


class BestFirstSearch(BaseSearchAlgorithm):
    def __init__(self, tree):
        super().__init__(tree)
        self.bfs_table = []  # (node_id, score, depth)

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

    def step(self):
        if self.bfs_table == []:
            leaf_id = 1
            depth = 1
            if self.tree.curr_iteration > 1:
                raise StopIteration("BFS exhausted the tree")
        else:
            depth = self.bfs_table[0][2]
            leaf_id = self.bfs_table.pop(0)[0]

        if self.tree.nodes[leaf_id].is_solved():
            if not leaf_id in self.tree.winning_nodes:
                self.tree.winning_nodes.append(leaf_id)
            self.tree.found_a_route = True
            return True, [leaf_id]
        self.tree._expand_node(leaf_id)
        self.tree.expanded_nodes.add(leaf_id)
        if self.tree.children[leaf_id] and depth < self.tree.config.max_depth:
            for child_id in self.tree.children[leaf_id]:
                if self.tree.nodes[child_id].is_solved():
                    if not child_id in self.tree.winning_nodes:
                        self.tree.winning_nodes.append(child_id)
                    self.tree.found_a_route = True
                    return True, [child_id]

                # Always evaluate using unified node value
                self.insert_dicho_bfs_table(child_id, self.tree._get_node_value(child_id), depth + 1, False)
        return False, [leaf_id]


class BeamSearch(BaseSearchAlgorithm):
    def __init__(self, tree):
        super().__init__(tree)
        self.bfs_table = []  # (node_id, score, depth)

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

    def step(self):
        nodes_to_open = []
        if self.bfs_table == []:
            if self.tree.curr_iteration > 1:
                raise StopIteration("One deterministic beam search is enough")
            nodes_to_open = [(1, 0, 1)]
        else:
            width = 10
            if width > len(self.bfs_table):
                width = len(self.bfs_table)
            for i in range(width):
                nodes_to_open.append(self.bfs_table[i])
        self.bfs_table = []

        for node in nodes_to_open:
            depth = node[2]
            leaf_id = node[0]
            self.tree._expand_node(leaf_id)
            self.tree.expanded_nodes.add(leaf_id)

            if self.tree.children[leaf_id] and depth < self.tree.config.max_depth:
                for child_id in self.tree.children[leaf_id]:
                    if self.tree.nodes[child_id].is_solved():
                        if not child_id in self.tree.winning_nodes:
                            self.tree.winning_nodes.append(child_id)
                        self.tree.found_a_route = True
                        return True, [child_id]

                    # Always evaluate using unified node value
                    self.insert_dicho_bfs_table(child_id, self.tree._get_node_value(child_id), depth + 1, False)

        return False, [1]


class UpperConfidenceSearch(BaseSearchAlgorithm):

    def __init__(self, tree):
        super().__init__(tree)
        self.c_ucb = 0.1
        self.epsilon = 0
        self.ucb_type = "uct" # ["uct", "puct", "value"]
        self.backprop_type = "muzero" # ["muzero", "cumulative"]
        self.evaluation_agg = "max"

    def _ucb(self, node_id: int) -> float:
        """Calculates the Upper Confidence Bound (UCB) statistics for a given node.

        :param node_id: The id of the node.
        :return: The calculated UCB.
        """

        prob = self.tree.nodes_prob[node_id]  # predicted by policy network score
        visit = self.tree.nodes_visit[node_id]

        if self.ucb_type == "puct":
            u = (
                self.c_ucb * prob * sqrt(self.tree.nodes_visit[self.tree.parents[node_id]])
            ) / (visit + 1)
            ucb_value = self.tree.nodes_total_value[node_id] + u

        if self.ucb_type == "uct":
            u = (
                self.c_ucb
                * sqrt(self.tree.nodes_visit[self.tree.parents[node_id]])
                / (visit + 1)
            )
            ucb_value = self.tree.nodes_total_value[node_id] + u

        if self.ucb_type == "value":
            ucb_value = self.tree.nodes_init_value[node_id] / (visit + 1)

        return ucb_value

    def _select_node(self, node_id: int) -> int:
        """Selects a node based on its UCB value and returns the id of the node with the
        highest UCB.

        :param node_id: The id of the node.
        :return: The id of the node with the highest UCB.
        """

        if self.epsilon > 0:
            n = uniform(0, 1)
            if n < self.epsilon:
                return choice(list(self.tree.children[node_id]))

        best_score, best_children = None, []
        for child_id in self.tree.children[node_id]:
            score = self._ucb(child_id)
            if best_score is None or score > best_score:
                best_score, best_children = score, [child_id]
            elif score == best_score:
                best_children.append(child_id)

        # is needed for tree search reproducibility, when all child nodes has the same score
        return best_children[0]

    def _backpropagate(self, node_id: int, value: float) -> None:
        """Backpropagates the value through the tree from the current.

        :param node_id: The id of the node from which to backpropagate the value.
        :param value: The value to backpropagate.
        :return: None.
        """
        while node_id:
            if self.backprop_type == "muzero":
                self.tree.nodes_total_value[node_id] = (
                    self.tree.nodes_total_value[node_id] * self.tree.nodes_visit[node_id] + value
                ) / (self.tree.nodes_visit[node_id] + 1)
            elif self.backprop_type == "cumulative":
                self.tree.nodes_total_value[node_id] += value
            node_id = self.tree.parents[node_id]

    def step(self):
        curr_depth, node_id = 0, 1  # start from the root node_id

        explore_route = True
        while explore_route:
            self.tree.visited_nodes.add(node_id)

            if self.tree.nodes_visit[node_id]:  # already visited
                if not self.tree.children[node_id]:  # dead node
                    self.tree._update_visits(node_id)
                    explore_route = False
                else:
                    node_id = self._select_node(node_id)  # select the child node
                    curr_depth += 1
            else:
                if self.tree.nodes[node_id].is_solved():  # found route
                    self.tree._update_visits(
                        node_id
                    )  # this prevents expanding of bb node_id
                    if not node_id in self.tree.winning_nodes:
                        self.tree.winning_nodes.append(node_id)
                    self.tree.found_a_route = True
                    return True, [node_id]

                if (
                        curr_depth < self.tree.config.max_depth
                ):  # expand node if depth limit is not reached
                    self.tree._expand_node(node_id)
                    self.tree.expanded_nodes.add(node_id)
                    if not self.tree.children[node_id]:  # node was not expanded
                        value_to_backprop = -1.0
                    else:
                        self.tree.expanded_nodes.add(node_id)

                        if self.tree.config.search_strategy == "evaluation_first":
                            # recalculate node value based on children synthesisability and backpropagation
                            child_values = [
                                self.tree.nodes_init_value[child_id]
                                for child_id in self.tree.children[node_id]
                            ]

                            if self.evaluation_agg == "max":
                                value_to_backprop = max(child_values)

                            elif self.evaluation_agg == "average":
                                value_to_backprop = sum(child_values) / len(
                                    self.tree.children[node_id]
                                )

                        elif self.tree.config.search_strategy == "expansion_first":
                            value_to_backprop = self.tree._get_node_value(node_id)

                    # backpropagation
                    self._backpropagate(node_id, value_to_backprop)
                    self.tree._update_visits(node_id)
                    explore_route = False

                    if self.tree.children[node_id]:
                        # found after expansion
                        found_after_expansion = set()
                        for child_id in iter(self.tree.children[node_id]):
                            if self.tree.nodes[child_id].is_solved():
                                found_after_expansion.add(child_id)
                                if not child_id in self.tree.winning_nodes:
                                    self.tree.winning_nodes.append(child_id)

                        if found_after_expansion:
                            self.tree.found_a_route = True
                            return True, list(found_after_expansion)

                else:
                    self._backpropagate(node_id, self.tree.nodes_total_value[node_id])
                    self.tree._update_visits(node_id)
                    explore_route = False

        return False, [node_id]


class NestedMonteCarloSearch(BaseSearchAlgorithm):
    def __init__(self, tree):
        super().__init__(tree)
        self.NMCS_level = 2
        self.big_dict_of_all_node_ids_NMCS_playout_values = {}

    def step(self):

        if self.tree.curr_iteration > 1:
            raise StopIteration("One deterministic NMCS iteration is enough.")

        node_id = 1
        best_node_id, _ = self.NMCS(node_id, self.NMCS_level, 1)

        if self.tree.nodes[best_node_id].is_solved():
            if not best_node_id in self.tree.winning_nodes:
                self.tree.winning_nodes.append(best_node_id)
            self.tree.found_a_route = True

            return True, [best_node_id]

        return False, [best_node_id]

    def NMCS(self, node_id, n, depth):
        best_node_id = node_id
        best_sequence = []
        return_sequence = []
        if not node_id in self.tree.expanded_nodes :
            self.tree._expand_node(node_id)
            self.tree.expanded_nodes.add(node_id)
        best_score = -100000
        all_child = self.tree.children[node_id].union(self.tree.redundant_children[node_id])
        while all_child:
            if time() - self.tree.start_time > self.tree.config.max_time :
                return best_node_id, return_sequence + best_sequence
            if self.tree.nodes[node_id].is_solved():
                if not node_id in self.tree.winning_nodes:
                    self.tree.winning_nodes.append(node_id)
                self.tree.found_a_route = True
                if self.tree.stop_at_first:
                    return node_id, return_sequence

            for child_id in all_child:
                s1 = child_id
                if self.tree.nodes[child_id].is_solved():
                    if not child_id in self.tree.winning_nodes:
                        self.tree.winning_nodes.append(child_id)
                    self.tree.found_a_route = True
                    return_sequence.append(s1)
                    if self.tree.stop_at_first:
                        return s1, return_sequence
                    continue
                if not child_id in self.tree.expanded_nodes:
                    self.tree._expand_node(child_id)
                    self.tree.expanded_nodes.add(child_id)
                if n == 1 :
                    sequence = []
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
                if self.tree.nodes[s1].is_solved():
                    if not s1 in self.tree.winning_nodes:
                        self.tree.winning_nodes.append(s1)
                    self.tree.found_a_route = True
                    if self.tree.stop_at_first:
                        return s1, return_sequence + sequence

                score = self.tree._get_node_value(s1)

                if score > best_score:
                    best_sequence = sequence
                    best_score = score
                    best_node_id = s1
                if time() - self.tree.start_time > self.tree.config.max_time:
                    return best_node_id, return_sequence + best_sequence
            if len(best_sequence) == 0:
                return node_id, return_sequence
            node_id = best_sequence[0]
            all_child = self.tree.children[node_id].union(self.tree.redundant_children[node_id])
            depth += 1
            return_sequence.append(best_sequence.pop(0))

            if not node_id in self.tree.expanded_nodes and not node_id in self.tree.winning_nodes:
                self.tree._expand_node(node_id)
                self.tree.expanded_nodes.add(node_id)

        return best_node_id, return_sequence

class LazyNestedMonteCarloSearch(BaseSearchAlgorithm):
    def __init__(self, tree, level=2):
        super().__init__(tree)
        self.bfs_table = []  # (node_id, score, depth)
        self.lnmcs_thresholds = [[] for _ in range(100)]  # list of scores at depth
        self.LNMCS_ratio = 0.2
        self.NMCS_level = 2

    def step(self):
        self.bfs_table = []
        self.lnmcs_thresholds = [[] for _ in range(100)]
        if self.tree.curr_iteration > 1:
            raise StopIteration("One deterministic LNMCS iteration is enough.")
        node_id = 1
        leaf, _ = self.LNMCS(node_id, self.NMCS_level, 1, self.LNMCS_ratio)
        if self.tree.nodes[leaf].is_solved():
            if not leaf in self.tree.winning_nodes:
                self.tree.winning_nodes.append(leaf)
            self.tree.found_a_route = True
            return True, [leaf]
        return False, [leaf]

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

    def insert_dicho_LNMCS_tresh(self, value, depth):
        if len(self.lnmcs_thresholds[depth]) == 0:
            self.lnmcs_thresholds[depth].append(value)
            return

        i1 = 0
        i2 = len(self.lnmcs_thresholds[depth]) - 1
        i = len(self.lnmcs_thresholds[depth]) // 2
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

    def LNMCS(self, node_id, n, depth, ratio):
        best_node_id = node_id
        best_sequence = []
        return_sequence = []
        if not node_id in self.tree.expanded_nodes :
            self.tree._expand_node(node_id)
            self.tree.expanded_nodes.add(node_id)
        best_score = -100000
        while self.tree.children[node_id] :
            if time() - self.tree.start_time > self.tree.config.max_time :
                return best_node_id, return_sequence + best_sequence
            if self.tree.nodes[node_id].is_solved():
                if not node_id in self.tree.winning_nodes:
                    self.tree.winning_nodes.append(node_id)
                self.tree.found_a_route = True
                if self.tree.stop_at_first:
                    return node_id, return_sequence

            self.bfs_table = []
            candidates = []

            self.lnmcs_thresholds = [[] for _ in range(100)]

            for child_id in self.tree.children[node_id]:
                if self.tree.nodes[child_id].is_solved():
                    if not child_id in self.tree.winning_nodes:
                        self.tree.winning_nodes.append(child_id)
                    self.tree.found_a_route = True
                    if self.tree.stop_at_first:
                        return_sequence.append(child_id)
                        return child_id, return_sequence
                    continue
                if not child_id in self.tree.expanded_nodes:
                    self.tree._expand_node(child_id)
                    self.tree.expanded_nodes.add(child_id)

                st_to_eval, seq = self.nmcs_rollout(child_id, depth + 1, "greedy")
                if self.tree.nodes[st_to_eval].is_solved():
                    if not st_to_eval in self.tree.winning_nodes:
                        self.tree.winning_nodes.append(st_to_eval)
                    self.tree.found_a_route = True
                    if self.tree.stop_at_first:
                        return_sequence.append(st_to_eval)
                        return st_to_eval, return_sequence + seq
                eval_score = self.tree._get_node_value(st_to_eval)

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
                    s1, sequence = self.nmcs_rollout(s1, depth+1, "policy")
                    sequence.insert(0, child_id)
                else:
                    s1, sequence = self.LNMCS(s1, n - 1, depth + 1, ratio)
                    sequence.insert(0, child_id)
                if self.tree.nodes[s1].is_solved():
                    if not s1 in self.tree.winning_nodes:
                        self.tree.winning_nodes.append(s1)
                    self.tree.found_a_route = True
                    if self.tree.stop_at_first:
                        return s1, return_sequence + sequence

                score = self.tree._get_node_value(s1)

                if score > best_score:
                    best_sequence = sequence
                    best_score = score
                    best_node_id = s1
                if time() - self.tree.start_time > self.tree.config.max_time:
                    return best_node_id, return_sequence + best_sequence
            if len(best_sequence) == 0:
                return node_id, return_sequence
            node_id = best_sequence[0]
            depth += 1
            return_sequence.append(best_sequence.pop(0))

            if not node_id in self.tree.expanded_nodes and not node_id in self.tree.winning_nodes:
                self.tree._expand_node(node_id)
                self.tree.expanded_nodes.add(node_id)

        return best_node_id, return_sequence
