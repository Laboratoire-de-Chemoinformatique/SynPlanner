"""Module containing a class that represents a value function for prediction of
synthesisablity of new nodes in the tree search."""

from typing import List, Optional, Set, Dict, Tuple

import torch

from synplan.chem.precursor import Precursor, compose_precursors
from synplan.mcts.expansion import PolicyNetworkFunction
from synplan.ml.networks.value import ValueNetwork
from synplan.ml.training import mol_to_pyg
from synplan.chem.rdkit_utils import RDKitScore



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
    """

    def __init__(
        self,
        policy_network: PolicyNetworkFunction,
        reaction_rules,
        building_blocks: Set[str],
        min_mol_size: int,
        max_depth: int,
    ) -> None:
        self.policy_network = policy_network
        self.reaction_rules = reaction_rules
        self.building_blocks = building_blocks
        self.min_mol_size = min_mol_size
        self.max_depth = max_depth

    def simulate_precursor(self, precursor: Precursor, current_depth: int = 0) -> float:
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

        from collections import defaultdict, deque

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

            reaction_rule_applied = False
            products = None
            for prob, rule, rule_id in self.policy_network.predict_reaction_rules(
                current_precursor, self.reaction_rules
            ):
                for prods in self._apply_rule(current_precursor, rule):
                    if prods:
                        products = prods
                        reaction_rule_applied = True
                        break
                if reaction_rule_applied:
                    history[rollout_depth]["rule_index"] = rule_id
                    break

            if not reaction_rule_applied:
                return -1.0

            products = tuple(Precursor(product) for product in products)
            history[rollout_depth]["products"] = products

            if any(x in occurred_precursor for x in products) and products:
                return -1.0

            if occurred_precursor.isdisjoint(products):
                precursor_to_expand.extend(
                    [
                        x
                        for x in products
                        if not x.is_building_block(self.building_blocks, self.min_mol_size)
                    ]
                )
                rollout_depth += 1

        return 1.0

    @staticmethod
    def _apply_rule(precursor_mol, rule):
        # Local import to avoid circular dependency
        from synplan.chem.reaction import apply_reaction_rule

        return apply_reaction_rule(precursor_mol.molecule, rule)


class EvaluationService:
    """Single entry-point to evaluate a node using configured scoring function.

    Produces values in [0, 1] if normalize=True.
    """

    def __init__(
        self,
        *,
        score_function: str,
        policy_network: PolicyNetworkFunction,
        reaction_rules,
        building_blocks: Set[str],
        min_mol_size: int,
        max_depth: int,
        value_network: Optional[ValueNetworkFunction] = None,
        normalize: bool = False,
    ) -> None:
        # Canonical naming: evaluation_function
        self.score_function = score_function
        self.evaluation_function = score_function
        self.value_network = value_network
        self.normalize = normalize
        self.policy_network = policy_network
        self.reaction_rules = reaction_rules
        self.building_blocks = building_blocks
        self.min_mol_size = min_mol_size
        self.max_depth = max_depth
        self.rollout = None 

    @staticmethod
    def _to_01(value: float, *, src_range: Tuple[float, float] = (0.0, 1.0)) -> float:
        a, b = src_range
        if b == a:
            return 0.0
        x = (value - a) / (b - a)
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        return float(x)

    def evaluate_node(
        self,
        *,
        node,
        node_id: int,
        nodes_depth: Dict[int, int],
        nodes_prob: Dict[int, float],
    ) -> float:
        """Evaluate and return a score. If normalize=True, output is in [0,1]."""

        sf = self.evaluation_function

        if sf == "random":
            from random import uniform

            score = uniform(0.0, 1.0)
            return self._to_01(score) if self.normalize else score

        if sf == "policy":
            score = nodes_prob.get(node_id, 0.0)
            return self._to_01(score) if self.normalize else score

        if sf == "gcn":
            if not self.value_network:
                raise ValueError("Value network not provided but score_function='gcn'.")
            score = float(self.value_network.predict_value(node.new_precursors))
            # Value net is expected to be in [0,1]; clamp defensively
            return self._to_01(score) if self.normalize else score

        if sf == "rollout":
            if self.rollout is None:
                self.rollout = RolloutSimulator(
                    policy_network=self.policy_network,
                    reaction_rules=self.reaction_rules,
                    building_blocks=self.building_blocks,
                    min_mol_size=self.min_mol_size,
                    max_depth=self.max_depth,
                )
            current_depth = nodes_depth[node_id]
            raw = min(
                (
                    self.rollout.simulate_precursor(precursor, current_depth=current_depth)
                    for precursor in node.precursors_to_expand
                ),
                default=1.0,
            )
            if self.normalize:
                # Map [-1, 1] -> [0, 1] only when normalization is requested
                score = 0.5 * (raw + 1.0)
                return self._to_01(score)
            return raw

        # RDKit and other chemistry-derived scores are handled via RDKitScore

        scorer = RDKitScore(score_function=sf)
        score = float(scorer(node))
        return self._to_01(score) if self.normalize else score
