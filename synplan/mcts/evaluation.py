"""Module containing evaluation strategies for node evaluation in tree search.

This module implements the Strategy pattern for different evaluation methods:
- Rollout simulation
- Value network (GCN)
- RDKit-based scores
- Policy probabilities
- Random scores
"""

import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from random import uniform
from typing import Dict, List, Set, Tuple

import torch

from synplan.chem.precursor import Precursor, compose_precursors
from synplan.chem.rdkit_utils import RDKitScore
from synplan.mcts.expansion import PolicyNetworkFunction
from synplan.ml.networks.value import ValueNetwork
from synplan.ml.training import mol_to_pyg


class ValueNetworkFunction:
    """Value function implemented as a value neural network for node evaluation
    (synthesisability prediction) in tree search."""

    def __init__(self, weights_path: str) -> None:
        """The value function predicts the probability to synthesize the target molecule
        with available building blocks starting from a given precursor.

        :param weights_path: The value network weights file path.
        """

        value_net = ValueNetwork.load_from_checkpoint(
            weights_path, map_location=torch.device("cpu")
        )
        self.value_network = value_net.eval()

    def predict_value(self, precursors: List[Precursor,]) -> float:
        """Predicts a value based on the given precursors from the node. For prediction,
        precursors must be composed into a single molecule (product).

        :param precursors: The list of precursors.
        :return: The predicted float value ("synthesisability") of the node.
        """

        molecule = compose_precursors(precursors=precursors, exclude_small=True)
        pyg_graph = mol_to_pyg(molecule)
        if pyg_graph:
            with torch.no_grad():
                value_pred = self.value_network.forward(pyg_graph)[0].item()
        else:
            value_pred = -1e6

        return value_pred


class RolloutSimulator:
    """Unified rollout simulator used for node evaluation and NMCS helpers.

    Returns rollout rewards in [-1, 1] following the existing semantics in Tree._rollout_node.
    Normalization to [0, 1] is done by EvaluationService if requested.

    Supports two modes:
    - Greedy (default): Takes the first successful reaction rule
    - Stochastic: Samples from valid rules proportionally to policy probabilities
    """

    def __init__(
        self,
        policy_network: PolicyNetworkFunction,
        reaction_rules,
        building_blocks: Set[str],
        min_mol_size: int,
        max_depth: int,
        stochastic: bool = False,
    ) -> None:
        """Initialize the rollout simulator.

        :param policy_network: Policy network for selecting reactions.
        :param reaction_rules: Available reaction rules.
        :param building_blocks: Set of building block molecules.
        :param min_mol_size: Minimum molecule size.
        :param max_depth: Maximum rollout depth.
        :param stochastic: If True, sample from valid rules using policy probabilities.
            If False (default), use greedy selection (first successful rule).
        """
        self.policy_network = policy_network
        self.reaction_rules = reaction_rules
        self.building_blocks = building_blocks
        self.min_mol_size = min_mol_size
        self.max_depth = max_depth
        self.stochastic = stochastic

    def _select_reaction(self, current_precursor: Precursor) -> Tuple[bool, any, int]:
        """Select a reaction rule to apply.

        :param current_precursor: The precursor to expand.
        :return: Tuple of (success, products, rule_id).
        """
        if self.stochastic:
            return self._select_reaction_stochastic(current_precursor)
        else:
            return self._select_reaction_greedy(current_precursor)

    def _select_reaction_greedy(
        self, current_precursor: Precursor
    ) -> Tuple[bool, any, int]:
        """Greedy selection: take first successful rule.

        :param current_precursor: The precursor to expand.
        :return: Tuple of (success, products, rule_id).
        """
        for _, rule, rule_id in self.policy_network.predict_reaction_rules(
            current_precursor, self.reaction_rules
        ):
            for prods in self._apply_rule(current_precursor, rule):
                if prods:
                    return True, prods, rule_id
        return False, None, -1

    def _select_reaction_stochastic(
        self, current_precursor: Precursor
    ) -> Tuple[bool, any, int]:
        """Stochastic selection: sample from rules until a valid one is found.

        Instead of applying all rules upfront, samples from the policy
        distribution and tries to apply. If the sampled rule fails,
        it's removed from candidates and we sample again.

        :param current_precursor: The precursor to expand.
        :return: Tuple of (success, products, rule_id).
        """
        # Collect all candidate rules with their probabilities
        candidates = [
            (prob, rule, rule_id)
            for prob, rule, rule_id in self.policy_network.predict_reaction_rules(
                current_precursor, self.reaction_rules
            )
        ]

        if not candidates:
            return False, None, -1

        # Sample until we find a valid rule or exhaust all candidates
        while candidates:
            # Sample one rule proportionally to probabilities
            probs = [c[0] for c in candidates]
            idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
            prob, rule, rule_id = candidates[idx]

            # Try to apply the sampled rule
            for prods in self._apply_rule(current_precursor, rule):
                if prods:
                    return True, prods, rule_id

            # Rule failed, remove from candidates and try again
            candidates.pop(idx)

        return False, None, -1

    def simulate_precursor(self, precursor: Precursor, current_depth: int = 0) -> float:
        """Performs a rollout simulation from a given node in the tree.

        If stochastic mode is enabled, samples from valid rules using policy
        probabilities instead of always taking the first successful rule.

        Rewards:
        - 1.0: All precursors are building blocks (success)
        - -0.5: Max depth exceeded
        - -1.0: No valid reaction found or cycle detected

        :param precursor: The precursor to be evaluated.
        :param current_depth: The current depth of the tree.
        :return: The reward (value) assigned to the precursor.
        """
        max_depth = self.max_depth - current_depth

        if precursor.is_building_block(self.building_blocks, self.min_mol_size):
            return 1.0

        occurred_precursor = set()
        precursor_to_expand = deque([precursor])
        history = defaultdict(dict)
        rollout_depth = 0

        while precursor_to_expand:
            if len(history) >= max_depth:
                return -0.5

            current_precursor = precursor_to_expand.popleft()
            history[rollout_depth]["target"] = current_precursor
            occurred_precursor.add(current_precursor)

            # Select reaction (greedy or stochastic based on self.stochastic)
            reaction_applied, products, rule_id = self._select_reaction(
                current_precursor
            )

            if not reaction_applied:
                return -1.0

            history[rollout_depth]["rule_index"] = rule_id
            products = tuple(Precursor(product) for product in products)
            history[rollout_depth]["products"] = products

            if any(x in occurred_precursor for x in products) and products:
                return -1.0

            if occurred_precursor.isdisjoint(products):
                precursor_to_expand.extend(
                    [
                        x
                        for x in products
                        if not x.is_building_block(
                            self.building_blocks, self.min_mol_size
                        )
                    ]
                )
                rollout_depth += 1

        return 1.0

    @staticmethod
    def _apply_rule(precursor_mol, rule):
        # Local import to avoid circular dependency
        from synplan.chem.reaction import apply_reaction_rule

        return apply_reaction_rule(precursor_mol.molecule, rule)


class EvaluationStrategy(ABC):
    """Abstract base class for all evaluation strategies.

    Implements the Strategy pattern for node evaluation in tree search.
    Each concrete strategy implements a different evaluation method.
    """

    @abstractmethod
    def evaluate_node(
        self,
        node,
        node_id: int,
        nodes_depth: Dict[int, int],
        nodes_prob: Dict[int, float],
    ) -> float:
        """Evaluate a node and return its score.

        :param node: The node to evaluate.
        :param node_id: ID of the node.
        :param nodes_depth: Dictionary mapping node IDs to depths.
        :param nodes_prob: Dictionary mapping node IDs to policy probabilities.
        :return: Evaluation score for the node.
        """
        pass

    @staticmethod
    def _to_01(value: float, *, src_range: Tuple[float, float] = (0.0, 1.0)) -> float:
        """Normalize value to [0, 1] range.

        :param value: Value to normalize.
        :param src_range: Source range (min, max).
        :return: Normalized value in [0, 1].
        """
        a, b = src_range
        if b == a:
            return 0.0
        x = (value - a) / (b - a)
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        return float(x)


class RolloutEvaluationStrategy(EvaluationStrategy):
    """Evaluation strategy using rollout simulation.

    Performs Monte Carlo rollout from the node to estimate synthesizability.
    Supports both greedy and stochastic (probability-weighted sampling) modes.
    """

    def __init__(
        self,
        policy_network: PolicyNetworkFunction,
        reaction_rules,
        building_blocks: Set[str],
        min_mol_size: int,
        max_depth: int,
        normalize: bool = False,
        stochastic: bool = False,
    ) -> None:
        """Initialize rollout evaluation strategy.

        :param policy_network: Policy network for selecting reactions during rollout.
        :param reaction_rules: Available reaction rules.
        :param building_blocks: Set of building block molecules.
        :param min_mol_size: Minimum molecule size.
        :param max_depth: Maximum rollout depth.
        :param normalize: Whether to normalize scores to [0, 1].
        :param stochastic: If True, sample from valid rules using policy probabilities.
            If False (default), use greedy selection (first successful rule).
        """
        self.rollout = RolloutSimulator(
            policy_network=policy_network,
            reaction_rules=reaction_rules,
            building_blocks=building_blocks,
            min_mol_size=min_mol_size,
            max_depth=max_depth,
            stochastic=stochastic,
        )
        self.normalize = normalize

    def evaluate_node(
        self,
        node,
        node_id: int,
        nodes_depth: Dict[int, int],
        nodes_prob: Dict[int, float],
    ) -> float:
        """Evaluate node using rollout simulation."""
        current_depth = nodes_depth[node_id]
        raw = min(
            (
                self.rollout.simulate_precursor(precursor, current_depth=current_depth)
                for precursor in node.precursors_to_expand
            ),
            default=1.0,
        )
        if self.normalize:
            # Map [-1, 1] -> [0, 1]
            score = 0.5 * (raw + 1.0)
            return self._to_01(score)
        return raw


class ValueNetworkEvaluationStrategy(EvaluationStrategy):
    """Evaluation strategy using a trained value neural network (GCN).

    Predicts synthesizability using a graph convolutional network.
    """

    def __init__(
        self,
        value_network: ValueNetworkFunction,
        normalize: bool = False,
    ) -> None:
        """Initialize value network evaluation strategy.

        :param value_network: Trained value network function.
        :param normalize: Whether to normalize scores to [0, 1].
        """
        self.value_network = value_network
        self.normalize = normalize

    def evaluate_node(
        self,
        node,
        node_id: int,
        nodes_depth: Dict[int, int],
        nodes_prob: Dict[int, float],
    ) -> float:
        """Evaluate node using value network."""
        score = float(self.value_network.predict_value(node.new_precursors))
        return self._to_01(score) if self.normalize else score


class RDKitEvaluationStrategy(EvaluationStrategy):
    """Evaluation strategy using RDKit molecular descriptors.

    Uses scores like SA score, molecular weight, etc.
    """

    def __init__(
        self,
        score_function: str,
        normalize: bool = False,
    ) -> None:
        """Initialize RDKit evaluation strategy.

        :param score_function: Name of the RDKit scoring function.
        :param normalize: Whether to normalize scores to [0, 1].
        """
        self.scorer = RDKitScore(score_function=score_function)
        self.normalize = normalize

    def evaluate_node(
        self,
        node,
        node_id: int,
        nodes_depth: Dict[int, int],
        nodes_prob: Dict[int, float],
    ) -> float:
        """Evaluate node using RDKit scorer."""
        score = float(self.scorer(node))
        return self._to_01(score) if self.normalize else score


class PolicyEvaluationStrategy(EvaluationStrategy):
    """Evaluation strategy using policy network probabilities.

    Uses the policy probability of the node as its evaluation score.
    """

    def __init__(self, normalize: bool = False) -> None:
        """Initialize policy evaluation strategy.

        :param normalize: Whether to normalize scores to [0, 1].
        """
        self.normalize = normalize

    def evaluate_node(
        self,
        node,
        node_id: int,
        nodes_depth: Dict[int, int],
        nodes_prob: Dict[int, float],
    ) -> float:
        """Evaluate node using policy probability."""
        score = nodes_prob.get(node_id, 0.0)
        return self._to_01(score) if self.normalize else score


class RandomEvaluationStrategy(EvaluationStrategy):
    """Evaluation strategy using random scores.

    Useful for testing and baseline comparisons.
    """

    def __init__(self, normalize: bool = False) -> None:
        """Initialize random evaluation strategy.

        :param normalize: Whether to normalize scores to [0, 1] (already in [0,1]).
        """
        self.normalize = normalize

    def evaluate_node(
        self,
        node,
        node_id: int,
        nodes_depth: Dict[int, int],
        nodes_prob: Dict[int, float],
    ) -> float:
        """Evaluate node using random score."""
        score = uniform(0.0, 1.0)
        return self._to_01(score) if self.normalize else score
