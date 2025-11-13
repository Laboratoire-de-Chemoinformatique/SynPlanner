from time import time
from random import choice, uniform
from math import sqrt
from abc import ABC, abstractmethod
from typing import List, Tuple, Literal, TYPE_CHECKING
from bisect import bisect_right

if TYPE_CHECKING:
    from .tree import Tree


class BaseSearchStrategy(ABC):
    """Minimal abstract base for search strategies operating on a retrosynthesis `Tree`.

    Contract:
    - Holds a reference to `tree` with fields like `children`, `parents`,
      `nodes_visit`, `nodes_total_value`, etc.
    - Subclasses must implement `step()` and return `(found_route, node_ids)`.
    """

    def __init__(self, tree: "Tree"):
        self.tree = tree

    @abstractmethod
    def step(self) -> Tuple[bool, List[int]]:
        """Perform a single algorithm-specific iteration."""
        raise NotImplementedError

    def _mark_solved(self, node_id: int) -> None:
        """Mark a node as solved, updating winners and the found flag."""
        if node_id not in self.tree.winning_nodes:
            self.tree.winning_nodes.append(node_id)
        self.tree.found_a_route = True


class ScoredFrontierMixin:
    def insert_sorted_frontier(
        self, node_id: int, score: float, depth: int, is_expanded: bool
    ) -> None:
        """Insert a node keeping frontier ordered by (score DESC, node_id ASC)."""
        keys = [(-entry[1], entry[0]) for entry in self.frontier]
        idx = bisect_right(keys, (-score, node_id))
        self.frontier.insert(idx, [node_id, score, depth, is_expanded])


class DepthThresholdsMixin:
    def reset_depth_thresholds(self, max_depth_hint: int | None = None) -> None:
        """(Re)initialize depth thresholds list; size driven by config.max_depth when available."""
        if max_depth_hint is not None:
            size = max(2, int(max_depth_hint) + 5)
        else:
            size = 100
            tree = getattr(self, "tree", None)
            if tree is not None and getattr(tree, "config", None) is not None:
                try:
                    max_depth = int(getattr(tree.config, "max_depth"))
                    size = max(2, max_depth + 5)
                except Exception:
                    size = 100
        self.depth_thresholds = [[] for _ in range(size)]

    def insert_sorted_threshold(self, score: float, depth: int) -> None:
        """Insert an evaluation score into the sorted list for a given depth."""
        if depth >= len(self.depth_thresholds):
            # extend to accommodate deeper levels than initially sized
            for _ in range(depth - len(self.depth_thresholds) + 1):
                self.depth_thresholds.append([])
        arr = self.depth_thresholds[depth]
        idx = bisect_right(arr, score)
        arr.insert(idx, score)


class NMCSPlayoutMixin:
    def select_nmcs_path(
        self, node_id: int, node_depth: int, mode: Literal["greedy", "random", "policy"]
    ) -> Tuple[int, List[int]]:
        """Run a simple playout used by NMCS variants starting at `node_id`."""
        depth = node_depth
        sequence: List[int] = []

        if time() - self.tree.start_time > self.tree.config.max_time:
            return node_id, sequence

        if node_id not in self.tree.expanded_nodes:
            self.tree._expand_node(node_id)
            self.tree.expanded_nodes.add(node_id)

        all_children = self.tree.children[node_id].union(
            self.tree.redundant_children[node_id]
        )

        while all_children and depth < self.tree.config.max_depth:
            if self.tree.nodes[node_id].is_solved():
                self._mark_solved(node_id)
                return node_id, sequence

            selected_id = None
            if mode == "greedy":
                best_value = -float("inf")
                for cid in all_children:
                    if self.tree.nodes[cid].is_solved():
                        self._mark_solved(cid)
                        sequence.append(cid)
                        return cid, sequence
                    value = self.tree._get_node_value(cid)
                    if value > best_value:
                        best_value = value
                        selected_id = cid

            elif mode == "random":
                candidate = choice(list(all_children))
                if self.tree.nodes[candidate].is_solved():
                    self._mark_solved(candidate)
                    sequence.append(candidate)
                    return candidate, sequence
                selected_id = candidate

            elif mode == "policy":
                best_value = -float("inf")
                for cid in all_children:
                    if self.tree.nodes[cid].is_solved():
                        self._mark_solved(cid)
                        sequence.append(cid)
                        return cid, sequence
                    value = self.tree.nodes_prob[cid]
                    if value > best_value:
                        best_value = value
                        selected_id = cid

            if selected_id is not None:
                sequence.append(selected_id)
                node_id = selected_id
                all_children = self.tree.children[node_id].union(
                    self.tree.redundant_children[node_id]
                )
                if node_id not in self.tree.expanded_nodes:
                    self.tree._expand_node(node_id)
                    self.tree.expanded_nodes.add(node_id)
            else:
                return node_id, sequence

            depth += 1

        return node_id, sequence


class BreadthFirst(BaseSearchStrategy):
    """Breadth-first search (BFS) over the tree frontier.

    Maintains a FIFO queue of nodes to expand in increasing depth order. On each
    `step`, the next node is popped, expanded once, and its children are enqueued.
    If a solved node is encountered either at the current node or among its
    immediate children, the step returns success immediately. The algorithm does
    not terminate internally; external limits in the `Tree` control stopping.
    """

    def __init__(self, tree):
        """Initialize BFS with an empty frontier."""
        super().__init__(tree)
        self.frontier: List[Tuple[int, int]] = []  # (node_id, depth)
        self.is_seeded: bool = False

    def step(self) -> Tuple[bool, List[int]]:
        """Expand one node in breadth-first order and enqueue its children.

        Returns (found_route, node_ids) where `node_ids` contains the last
        visited node id or the solved child id when discovered.
        """
        if not self.frontier:
            if not self.is_seeded:
                node_id, depth = 1, 1
                self.frontier.append((node_id, depth))
                self.is_seeded = True
            else:
                return False, [1]
        node_id, depth = self.frontier.pop(0)

        if self.tree.nodes[node_id].is_solved():
            self._mark_solved(node_id)
            return True, [node_id]

        if node_id not in self.tree.expanded_nodes:
            self.tree._expand_node(node_id)
            self.tree.expanded_nodes.add(node_id)

        if self.tree.children[node_id] and depth < self.tree.config.max_depth:
            for child_id in self.tree.children[node_id]:
                if self.tree.nodes[child_id].is_solved():
                    self._mark_solved(child_id)
                    return True, [child_id]
                self.frontier.append((child_id, depth + 1))

        return False, [node_id]


class BestFirst(ScoredFrontierMixin, BaseSearchStrategy):
    """Best-first search ordered by node evaluation.

    Keeps a globally sorted frontier (highest score first) according to the
    node value computed by `Tree._get_node_value`. Each `step` expands the top
    frontier node and re-inserts its children with their scores, prioritizing
    the most promising parts of the tree.
    """

    def __init__(self, tree):
        """Initialize best-first search with a scored frontier."""
        super().__init__(tree)
        self.frontier: List[List[int | float]] = (
            []
        )  # (node_id, score, depth, is_expanded)

    def step(self) -> Tuple[bool, List[int]]:
        """Expand the highest-scored node and re-enqueue evaluated children.

        Children are scored via `Tree._get_node_value` and inserted into the
        frontier to preserve the best-first ordering.
        """
        if not self.frontier:
            node_id, depth = 1, 1
        else:
            depth = self.frontier[0][2]
            node_id = self.frontier.pop(0)[0]

        if self.tree.nodes[node_id].is_solved():
            if node_id not in self.tree.winning_nodes:
                self.tree.winning_nodes.append(node_id)
            self.tree.found_a_route = True
            return True, [node_id]

        if node_id not in self.tree.expanded_nodes:
            self.tree._expand_node(node_id)
            self.tree.expanded_nodes.add(node_id)

        if self.tree.children[node_id] and depth < self.tree.config.max_depth:
            for child_id in self.tree.children[node_id]:
                if self.tree.nodes[child_id].is_solved():
                    self._mark_solved(child_id)
                    return True, [child_id]

                # Always evaluate using unified node value
                self.insert_sorted_frontier(
                    child_id, self.tree._get_node_value(child_id), depth + 1, False
                )
        return False, [node_id]


class Beam(ScoredFrontierMixin, BaseSearchStrategy):
    """Beam search over a scored frontier with a fixed width.

    Similar to best-first but expands the top-`beam_width` nodes at each step,
    trading breadth for speed. Children of the opened nodes are evaluated and
    merged back into the frontier. A smaller beam favors exploitation; a larger
    beam explores more alternatives.
    """

    def __init__(self, tree):
        """Initialize beam search with an empty scored frontier."""
        super().__init__(tree)
        self.frontier: List[List[int | float]] = []  # (node_id, score, depth, expanded)

    def step(self) -> Tuple[bool, List[int]]:
        """Open up to `beam_width` best nodes, expand them, and enqueue children.

        The frontier is cleared and rebuilt from the newly generated children
        at each step, reflecting the sliding nature of beam search.
        """
        batch: List[Tuple[int, float, int]] = []
        if not self.frontier:
            batch = [(1, 0.0, 1)]
        else:
            beam_w = getattr(self.tree.config, "beam_width", 10)
            beam_w = max(1, int(beam_w))
            beam_w = min(beam_w, len(self.frontier))
            for i in range(beam_w):
                batch.append(
                    (self.frontier[i][0], self.frontier[i][1], self.frontier[i][2])
                )
        self.frontier = []

        for entry in batch:
            node_id, _score, depth = entry
            if node_id not in self.tree.expanded_nodes:
                self.tree._expand_node(node_id)
                self.tree.expanded_nodes.add(node_id)

            if self.tree.children[node_id] and depth < self.tree.config.max_depth:
                for child_id in self.tree.children[node_id]:
                    if self.tree.nodes[child_id].is_solved():
                        self._mark_solved(child_id)
                        return True, [child_id]

                    # Always evaluate using unified node value
                    self.insert_sorted_frontier(
                        child_id, self.tree._get_node_value(child_id), depth + 1, False
                    )

        return False, [1]


class UCT(BaseSearchStrategy):
    """Upper Confidence Tree (UCT/PUCT/value) search with epsilon-greedy.

    Selection uses one of:
    - UCT: Q(s) + c * sqrt(N(parent)) / (N(s)+1)
    - PUCT: Q(s) + c * P(s) * sqrt(N(parent)) / (N(s)+1)
    - value-only: V_init(s) / (N(s)+1)

    Where Q is the running estimate of node value (`nodes_total_value`), P is the
    policy prior (`nodes_prob`), N are visit counts, and c is an exploration
    constant. With probability `epsilon`, selection chooses a random child to
    encourage exploration.
    """

    def __init__(self, tree):
        """Initialize UCT parameters from the tree configuration."""
        super().__init__(tree)
        cfg = self.tree.config
        self.ucb_type = getattr(cfg, "ucb_type", "uct").strip().lower()
        if self.ucb_type not in ("uct", "puct", "value"):
            self.ucb_type = "uct"
        try:
            self.c_ucb = max(0.0, float(getattr(cfg, "c_ucb", 0.1)))
        except (TypeError, ValueError):
            self.c_ucb = 0.1
        self.backprop_type = getattr(cfg, "backprop_type", "muzero").strip().lower()
        if self.backprop_type not in ("muzero", "cumulative"):
            self.backprop_type = "muzero"
        self.evaluation_agg = getattr(cfg, "evaluation_agg", "max").strip().lower()
        if self.evaluation_agg not in ("max", "average"):
            self.evaluation_agg = "max"
        try:
            self.epsilon = min(1.0, max(0.0, float(getattr(cfg, "epsilon", 0.0))))
        except (TypeError, ValueError):
            self.epsilon = 0.0

    def _ucb(self, node_id: int) -> float:
        """Calculate the selection score for a node based on the configured rule.

        Returns the UCT/PUCT/value-based score as described in the class doc.
        """
        prob = self.tree.nodes_prob[node_id]
        visit = self.tree.nodes_visit[node_id]

        if self.ucb_type == "puct":
            u = (
                self.c_ucb
                * prob
                * sqrt(self.tree.nodes_visit[self.tree.parents[node_id]])
            ) / (visit + 1)
            return self.tree.nodes_total_value[node_id] + u

        if self.ucb_type == "uct":
            u = (
                self.c_ucb
                * sqrt(self.tree.nodes_visit[self.tree.parents[node_id]])
                / (visit + 1)
            )
            return self.tree.nodes_total_value[node_id] + u

        # value-based
        return self.tree.nodes_init_value[node_id] / (visit + 1)

    def _select_node(self, node_id: int) -> int:
        """Pick the child with the highest selection score (ties broken deterministically).

        With probability `epsilon`, a random child is returned to explore. When
        multiple children share the same score, the smallest id is returned for
        reproducibility.
        """

        if self.epsilon > 0:
            n = uniform(0, 1)
            if n < self.epsilon:
                return choice(list(self.tree.children[node_id]))

        best_score = None
        best_children: List[int] = []
        for child_id in self.tree.children[node_id]:
            score = self._ucb(child_id)
            if best_score is None or score > best_score:
                best_score, best_children = score, [child_id]
            elif score == best_score:
                best_children.append(child_id)
        return best_children[0]

    def _backpropagate(self, node_id: int, value: float) -> None:
        """Backpropagate a scalar value from `node_id` up to the root.

        Modes:
        - "muzero": incremental mean update of Q-values
        - "cumulative": running sum update (Monte Carlo return style)
        """
        while node_id:
            if self.backprop_type == "muzero":
                self.tree.nodes_total_value[node_id] = (
                    self.tree.nodes_total_value[node_id]
                    * self.tree.nodes_visit[node_id]
                    + value
                ) / (self.tree.nodes_visit[node_id] + 1)
            else:  # cumulative
                self.tree.nodes_total_value[node_id] += value
            node_id = self.tree.parents[node_id]

    def step(self) -> Tuple[bool, List[int]]:
        """Run one UCT iteration: selection → expansion/evaluation → backpropagation.

        On an unvisited node, either returns a solved node immediately or expands
        it (respecting `max_depth`). The value used for backpropagation depends on
        `search_strategy` (evaluation-first aggregates children; expansion-first
        evaluates the current node via the evaluator).
        """
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
                    )  # prevents expanding of bb node_id
                    self._mark_solved(node_id)
                    return True, [node_id]

                if (
                    curr_depth < self.tree.config.max_depth
                ):  # expand node if depth limit is not reached
                    self.tree._expand_node(node_id)
                    self.tree.expanded_nodes.add(node_id)

                    value_to_backprop = -1.0  # node was not expanded
                    if self.tree.children[node_id]:
                        if self.tree.config.search_strategy == "evaluation_first":
                            # recalculate node value based on children synthesisability and backpropagation
                            child_values = [
                                self.tree.nodes_init_value[child_id]
                                for child_id in self.tree.children[node_id]
                            ]
                            if self.evaluation_agg == "max":
                                value_to_backprop = max(child_values)
                            else:  # average
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
                                if child_id not in self.tree.winning_nodes:
                                    self.tree.winning_nodes.append(child_id)

                        if found_after_expansion:
                            self.tree.found_a_route = True
                            return True, list(found_after_expansion)
                else:
                    self._backpropagate(node_id, self.tree.nodes_total_value[node_id])
                    self.tree._update_visits(node_id)
                    explore_route = False

        return False, [node_id]


class NestedMonteCarlo(NMCSPlayoutMixin, BaseSearchStrategy):
    """Nested Monte Carlo Search (NMCS) over the tree.

    Performs recursive rollouts with increasing nesting depth. At level 1, a
    greedy rollout is executed; at higher levels, each child is recursively
    explored at a lower level and the best outcome is selected, building a
    sequence of improving choices. Results of base rollouts are memoized to
    avoid recomputation across siblings.
    """

    def __init__(self, tree):
        """Initialize NMCS with default nesting level and caches."""
        super().__init__(tree)
        self.nmcs_level = 2
        self.rollout_cache = {}

    def step(self) -> Tuple[bool, List[int]]:
        """Perform a single NMCS pass from the root (level = `NMCS_level`)."""
        if self.tree.curr_iteration > 1:
            # Deterministic NMCS single pass
            return False, [1]

        node_id = 1
        best_node_id, _ = self.NMCS(node_id, self.nmcs_level, 1)

        if self.tree.nodes[best_node_id].is_solved():
            self._mark_solved(best_node_id)
            return True, [best_node_id]

        return False, [best_node_id]

    def NMCS(self, node_id, level, depth):
        """Recursive nested Monte Carlo search.

        Args:
            node_id: Current node id.
            level: Nesting level. If 1, a greedy rollout is used; otherwise, the
               function recursively evaluates children with level n-1.
            depth: Current depth for respecting `max_depth` and thresholds.

        Returns:
            Tuple[int, List[int]]: Best leaf id found and the improving sequence
            of choices (node ids) that led there.
        """
        best_node_id = node_id
        best_path: List[int] = []
        chosen_path: List[int] = []
        if node_id not in self.tree.expanded_nodes:
            self.tree._expand_node(node_id)
            self.tree.expanded_nodes.add(node_id)
        best_score = -float("inf")
        all_children = self.tree.children[node_id].union(
            self.tree.redundant_children[node_id]
        )
        while all_children:
            if time() - self.tree.start_time > self.tree.config.max_time:
                return best_node_id, chosen_path + best_path
            if self.tree.nodes[node_id].is_solved():
                self._mark_solved(node_id)
                if self.tree.stop_at_first:
                    return node_id, chosen_path

            for child_id in all_children:
                candidate_id = child_id
                if self.tree.nodes[child_id].is_solved():
                    self._mark_solved(child_id)
                    chosen_path.append(candidate_id)
                    if self.tree.stop_at_first:
                        return candidate_id, chosen_path
                    continue
                if child_id not in self.tree.expanded_nodes:
                    self.tree._expand_node(child_id)
                    self.tree.expanded_nodes.add(child_id)
                if level == 1:
                    path: List[int] = []
                    if candidate_id in self.rollout_cache:
                        (candidate_id, path) = self.rollout_cache[candidate_id]
                    else:
                        candidate_key = candidate_id
                        candidate_id, path = self.select_nmcs_path(
                            candidate_id, depth + 1, "greedy"
                        )
                        self.rollout_cache[candidate_key] = (
                            candidate_id,
                            path,
                        )
                    path.insert(0, child_id)
                else:
                    candidate_id, path = self.NMCS(candidate_id, level - 1, depth + 1)
                    path.insert(0, child_id)
                if self.tree.nodes[candidate_id].is_solved():
                    self._mark_solved(candidate_id)
                    if self.tree.stop_at_first:
                        return candidate_id, chosen_path + path

                score = self.tree._get_node_value(candidate_id)

                if score > best_score:
                    best_path = path
                    best_score = score
                    best_node_id = candidate_id
                if time() - self.tree.start_time > self.tree.config.max_time:
                    return best_node_id, chosen_path + best_path
            if len(best_path) == 0:
                return node_id, chosen_path
            node_id = best_path[0]
            all_children = self.tree.children[node_id].union(
                self.tree.redundant_children[node_id]
            )
            depth += 1
            chosen_path.append(best_path.pop(0))

            if (node_id not in self.tree.expanded_nodes) and (
                node_id not in self.tree.winning_nodes
            ):
                self.tree._expand_node(node_id)
                self.tree.expanded_nodes.add(node_id)

        return best_node_id, chosen_path


class LazyNestedMonteCarlo(
    NMCSPlayoutMixin, ScoredFrontierMixin, DepthThresholdsMixin, BaseSearchStrategy
):
    """Lazy NMCS variant using percentile thresholds to restrict candidates.

    For each node, it estimates rollout scores for children and keeps only those
    above a configurable percentile (ratio) — a cheap approximation limiting the
    branching factor. It then applies NMCS on that reduced set.
    """

    def __init__(self, tree, level=2):
        """Initialize lazy NMCS with thresholds and nesting level."""
        super().__init__(tree)
        self.frontier: List[List[int | float]] = (
            []
        )  # (node_id, score, depth, is_expanded)
        self.reset_depth_thresholds()
        self.lnmcs_ratio = 0.2
        self.nmcs_level = 2

    def step(self) -> Tuple[bool, List[int]]:
        """Perform one lazy NMCS iteration using percentile-based candidate pruning."""
        self.frontier = []
        self.reset_depth_thresholds()
        if self.tree.curr_iteration > 1:
            return False, [1]
        node_id = 1
        leaf_id, _ = self.LNMCS(node_id, self.nmcs_level, 1, self.lnmcs_ratio)
        if self.tree.nodes[leaf_id].is_solved():
            self._mark_solved(leaf_id)
            return True, [leaf_id]
        return False, [leaf_id]

    def LNMCS(self, node_id, level, depth, prune_ratio):
        """Lazy NMCS recursion selecting candidates above a percentile cut-off.

        Args:
            node_id: Current node id.
            level: Nesting level; at 1, uses policy/greedy rollout; otherwise recurses.
            depth: Current depth in the tree.
            prune_ratio: Percentile ratio (0..1) used to filter candidates at this depth.

        Returns:
            Tuple[int, List[int]]: Best leaf id and chosen sequence.
        """
        best_node_id = node_id
        best_path: List[int] = []
        chosen_path: List[int] = []
        if node_id not in self.tree.expanded_nodes:
            self.tree._expand_node(node_id)
            self.tree.expanded_nodes.add(node_id)
        best_score = -float("inf")
        while self.tree.children[node_id]:
            if time() - self.tree.start_time > self.tree.config.max_time:
                return best_node_id, chosen_path + best_path
            if self.tree.nodes[node_id].is_solved():
                self._mark_solved(node_id)
                if self.tree.stop_at_first:
                    return node_id, chosen_path

            self.frontier = []
            candidates: List[int] = []

            self.reset_depth_thresholds()

            for child_id in self.tree.children[node_id]:
                if self.tree.nodes[child_id].is_solved():
                    self._mark_solved(child_id)
                    if self.tree.stop_at_first:
                        chosen_path.append(child_id)
                        return child_id, chosen_path
                    continue
                if child_id not in self.tree.expanded_nodes:
                    self.tree._expand_node(child_id)
                    self.tree.expanded_nodes.add(child_id)

                eval_id, path = self.select_nmcs_path(child_id, depth + 1, "greedy")
                if self.tree.nodes[eval_id].is_solved():
                    self._mark_solved(eval_id)
                    if self.tree.stop_at_first:
                        chosen_path.append(eval_id)
                        return eval_id, chosen_path + path
                eval_score = self.tree._get_node_value(eval_id)

                self.insert_sorted_threshold(eval_score, depth)
                self.insert_sorted_frontier(child_id, eval_score, 0, False)

            best_candidate_id = -1
            best_candidate_score = -float("inf")
            for entry in self.frontier:
                if entry[1] > best_candidate_score:
                    best_candidate_score = entry[1]
                    best_candidate_id = entry[0]
                if (
                    entry[1]
                    >= self.depth_thresholds[depth][
                        int(prune_ratio * (len(self.depth_thresholds[depth]) - 1))
                    ]
                ):
                    candidates.append(entry[0])
            if best_candidate_id != -1 and candidates == []:
                candidates = [best_candidate_id]

            for child_id in candidates:
                candidate_id = child_id
                if level == 1:
                    path: List[int] = []
                    candidate_id, path = self.select_nmcs_path(
                        candidate_id, depth + 1, "policy"
                    )
                    path.insert(0, child_id)
                else:
                    candidate_id, path = self.LNMCS(
                        candidate_id, level - 1, depth + 1, prune_ratio
                    )
                    path.insert(0, child_id)
                if self.tree.nodes[candidate_id].is_solved():
                    self._mark_solved(candidate_id)
                    if self.tree.stop_at_first:
                        return candidate_id, chosen_path + path

                score = self.tree._get_node_value(candidate_id)

                if score > best_score:
                    best_path = path
                    best_score = score
                    best_node_id = candidate_id
                if time() - self.tree.start_time > self.tree.config.max_time:
                    return best_node_id, chosen_path + best_path
            if len(best_path) == 0:
                return node_id, chosen_path
            node_id = best_path[0]
            depth += 1
            chosen_path.append(best_path.pop(0))

            if (node_id not in self.tree.expanded_nodes) and (
                node_id not in self.tree.winning_nodes
            ):
                self.tree._expand_node(node_id)
                self.tree.expanded_nodes.add(node_id)

        return best_node_id, chosen_path
