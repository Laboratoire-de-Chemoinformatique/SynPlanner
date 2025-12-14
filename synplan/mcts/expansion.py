"""Module containing a class that represents a policy function for node expansion in the
tree search."""

from typing import Iterator, List, Optional, Tuple, Union

import torch
import torch_geometric
from CGRtools.reactor.reactor import Reactor

from synplan.chem.precursor import Precursor
from synplan.ml.networks.policy import PolicyNetwork
from synplan.ml.training import mol_to_pyg
from synplan.utils.config import PolicyNetworkConfig


class PolicyNetworkFunction:
    """Policy function implemented as a policy neural network for node expansion in tree
    search."""

    def __init__(
        self, policy_config: PolicyNetworkConfig, compile: bool = False
    ) -> None:
        """Initializes the expansion function (ranking or filter policy network).

        :param policy_config: An expansion policy configuration.
        :param compile: Is supposed to speed up the training with model compilation.
        """

        self.config = policy_config

        policy_net = PolicyNetwork.load_from_checkpoint(
            self.config.weights_path,
            map_location=torch.device("cpu"),
            batch_size=1,
            dropout=0,
        )

        policy_net = policy_net.eval()
        if compile:
            self.policy_net = torch_geometric.compile(policy_net, dynamic=True)
        else:
            self.policy_net = policy_net

    def _get_graph(self, precursor: Precursor) -> Optional[torch_geometric.data.Data]:
        """Convert precursor molecule to PyG graph.

        :param precursor: The precursor molecule.
        :return: PyG graph or None if conversion fails.
        """
        return mol_to_pyg(precursor.molecule, canonicalize=False)

    def _get_embedding(self, pyg_graph: torch_geometric.data.Data) -> torch.Tensor:
        """Get molecule embedding from the network.

        :param pyg_graph: PyG graph of the molecule.
        :return: Embedding tensor.
        """
        return self.policy_net.embedder(pyg_graph, self.policy_net.batch_size)

    def get_logits(self, precursor: Precursor) -> Optional[torch.Tensor]:
        """Get raw logits from the policy network (before sigmoid/softmax).

        Works for both filtering and ranking networks.

        :param precursor: The current precursor.
        :return: Raw logits tensor for all rules, or None if graph conversion fails.
        """
        pyg_graph = self._get_graph(precursor)
        if not pyg_graph:
            return None

        with torch.no_grad():
            x = self._get_embedding(pyg_graph)
            logits = self.policy_net.y_predictor(x)
            logits = logits[0].double()

        del pyg_graph
        return logits

    def get_probs(self, precursor: Precursor) -> Optional[torch.Tensor]:
        """Get probability tensor from the policy network.

        For filtering: returns sigmoid(logits) combined with priority if configured.
        For ranking: returns softmax(logits).

        :param precursor: The current precursor.
        :return: Probability tensor for all rules, or None if graph conversion fails.
        """
        pyg_graph = self._get_graph(precursor)
        if not pyg_graph:
            return None

        with torch.no_grad():
            if self.policy_net.policy_type == "filtering":
                probs, priority = self.policy_net.forward(pyg_graph)
                probs = probs[0].double()
                priority = priority[0].double()
                priority_coef = self.config.priority_rules_fraction
                probs = (1 - priority_coef) * probs + priority_coef * priority
            else:  # ranking
                probs = self.policy_net.forward(pyg_graph)
                probs = probs[0].double()

        del pyg_graph
        return probs

    def _predict_rules_common(
        self, precursor: Precursor, n_rules: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Common logic for predicting reaction rules.

        :param precursor: The current precursor.
        :param n_rules: Expected number of reaction rules.
        :return: Tuple of (sorted_probs, sorted_rule_ids) or None if prediction fails.
        """
        out_dim = list(self.policy_net.modules())[-1].out_features
        if out_dim != n_rules:
            raise Exception(
                f"The policy network output dimensionality is {out_dim}, but the number of reaction rules is {n_rules}. "
                "Probably you use a different version of the policy network. Be sure to retain the policy network "
                "with the current set of reaction rules"
            )

        probs = self.get_probs(precursor)
        if probs is None:
            return None

        sorted_probs, sorted_rules = torch.sort(probs, descending=True)
        sorted_probs, sorted_rules = (
            sorted_probs[: self.config.top_rules],
            sorted_rules[: self.config.top_rules],
        )

        if self.policy_net.policy_type == "filtering":
            sorted_probs = torch.softmax(sorted_probs, -1)

        return sorted_probs, sorted_rules

    def predict_reaction_rules(
        self, precursor: Precursor, reaction_rules: List[Reactor]
    ) -> Iterator[Union[Iterator, Iterator[Tuple[float, Reactor, int]]]]:
        """The policy function predicts the list of reaction rules for a given precursor.

        :param precursor: The current precursor for which the reaction rules are predicted.
        :param reaction_rules: The list of reaction rules from which applicable reaction
            rules are predicted and selected.
        :return: Yielding the predicted probability for the reaction rule, reaction rule
            and reaction rule id.
        """
        result = self._predict_rules_common(precursor, len(reaction_rules))
        if result is None:
            return []

        sorted_probs, sorted_rules = result
        sorted_probs, sorted_rules = sorted_probs.tolist(), sorted_rules.tolist()

        for prob, rule_id in zip(sorted_probs, sorted_rules):
            if prob > self.config.rule_prob_threshold:
                yield prob, reaction_rules[rule_id], rule_id

    def predict_reaction_rules_light(
        self, precursor: Precursor, reaction_rules_len: int
    ) -> Iterator[Union[Iterator, Iterator[Tuple[float, int]]]]:
        """The policy function predicts the list of reaction rules for a given precursor.

        Light version that doesn't return Reactor objects.

        :param precursor: The current precursor for which the reaction rules are predicted.
        :param reaction_rules_len: The number of reaction rules.
        :return: Yielding the predicted probability and reaction rule id.
        """
        result = self._predict_rules_common(precursor, reaction_rules_len)
        if result is None:
            return []

        sorted_probs, sorted_rules = result
        sorted_probs, sorted_rules = sorted_probs.tolist(), sorted_rules.tolist()

        for prob, rule_id in zip(sorted_probs, sorted_rules):
            if prob > self.config.rule_prob_threshold:
                yield prob, rule_id

    def get_filtering_probs_only(self, precursor: Precursor) -> Optional[torch.Tensor]:
        """Get filtering probability tensor (y only, no priority mixing).

        :param precursor: The current precursor.
        :return: Raw filtering probability tensor (sigmoid output) for all rules.
        """
        if self.policy_net.policy_type != "filtering":
            raise ValueError("This method is only for filtering policy networks")

        logits = self.get_logits(precursor)
        if logits is None:
            return None
        return torch.sigmoid(logits)

    def get_ranking_logits(self, precursor: Precursor) -> Optional[torch.Tensor]:
        """Get raw logits from ranking policy network (before softmax).

        :param precursor: The current precursor.
        :return: Raw logits tensor for all rules.
        """
        if self.policy_net.policy_type != "ranking":
            raise ValueError("This method is only for ranking policy networks")
        return self.get_logits(precursor)


class CombinedPolicyNetworkFunction:
    """Combined policy function that adds filtering and ranking logits.

    Combines filtering and ranking policies by weighted addition of logits:
        combined_logits = filtering_logits + ranking_weight * ranking_logits
        combined_probs = softmax(combined_logits / temperature)

    Both networks output raw logits (before sigmoid/softmax). The weighting
    allows controlling the balance between:
    - Filtering: "Is this rule applicable?" (trained on multi-label applicability)
    - Ranking: "Is this rule likely to work?" (trained on actual reactions)

    Parameters:
    - ranking_weight > 1.0: More bias toward ranking (better feasibility)
    - ranking_weight < 1.0: More bias toward filtering (more exploration)
    - temperature > 1.0: Softer distribution (more exploration)
    - temperature < 1.0: Sharper distribution (more exploitation)
    """

    def __init__(
        self,
        filtering_config: PolicyNetworkConfig,
        ranking_config: PolicyNetworkConfig,
        top_rules: int = 50,
        rule_prob_threshold: float = 0.0,
        ranking_weight: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        """Initializes the combined policy function with both filtering and ranking networks.

        :param filtering_config: Configuration for the filtering policy network.
        :param ranking_config: Configuration for the ranking policy network.
        :param top_rules: Number of top rules to return.
        :param rule_prob_threshold: Minimum probability threshold for returning a rule.
        :param ranking_weight: Weight for ranking logits (default 1.0).
            Values > 1.0 give more weight to ranking (feasibility).
            Values < 1.0 give more weight to filtering (applicability).
        :param temperature: Temperature for softmax (default 1.0).
            Values > 1.0 produce softer distributions (more exploration).
            Values < 1.0 produce sharper distributions (more exploitation).
        """
        if filtering_config.policy_type != "filtering":
            raise ValueError(
                f"filtering_config must have policy_type='filtering', got '{filtering_config.policy_type}'"
            )
        if ranking_config.policy_type != "ranking":
            raise ValueError(
                f"ranking_config must have policy_type='ranking', got '{ranking_config.policy_type}'"
            )

        self.filtering_net = PolicyNetworkFunction(filtering_config)
        self.ranking_net = PolicyNetworkFunction(ranking_config)
        self.top_rules = top_rules
        self.rule_prob_threshold = rule_prob_threshold
        self.ranking_weight = ranking_weight
        self.temperature = temperature

    @property
    def n_rules(self) -> int:
        """Get the number of reaction rules the networks were trained on."""
        return list(self.filtering_net.policy_net.modules())[-1].out_features

    def _validate_dimensions(self, expected_n_rules: int) -> None:
        """Validate that network output dimensions match expected number of rules."""
        filtering_dim = list(self.filtering_net.policy_net.modules())[-1].out_features
        ranking_dim = list(self.ranking_net.policy_net.modules())[-1].out_features

        if filtering_dim != expected_n_rules or ranking_dim != expected_n_rules:
            raise Exception(
                f"Policy network output dimensions (filtering={filtering_dim}, ranking={ranking_dim}) "
                f"do not match the number of reaction rules ({expected_n_rules}). "
                "Both policy networks must be trained on the same set of reaction rules."
            )

    def _get_combined_probs(self, precursor: Precursor) -> Optional[torch.Tensor]:
        """Compute combined probabilities by weighted addition of logits.

        Formula: softmax((filtering_logits + ranking_weight * ranking_logits) / temperature)

        :param precursor: The current precursor.
        :return: Combined probability tensor or None if inference fails.
        """
        # Get raw logits from both networks
        filtering_logits = self.filtering_net.get_logits(precursor)
        ranking_logits = self.ranking_net.get_logits(precursor)

        if filtering_logits is None or ranking_logits is None:
            return None

        # Weighted combination of logits
        combined_logits = filtering_logits + self.ranking_weight * ranking_logits

        # Temperature-scaled softmax
        return torch.softmax(combined_logits / self.temperature, dim=-1)

    def _predict_rules_common(
        self, precursor: Precursor, n_rules: int
    ) -> Optional[Tuple[List[float], List[int]]]:
        """Common logic for predicting reaction rules.

        :param precursor: The current precursor.
        :param n_rules: Expected number of reaction rules.
        :return: Tuple of (sorted_probs, sorted_rule_ids) as lists, or None.
        """
        self._validate_dimensions(n_rules)

        combined_probs = self._get_combined_probs(precursor)
        if combined_probs is None:
            return None

        # Sort and select top rules
        sorted_probs, sorted_rules = torch.sort(combined_probs, descending=True)
        sorted_probs = sorted_probs[: self.top_rules].tolist()
        sorted_rules = sorted_rules[: self.top_rules].tolist()

        return sorted_probs, sorted_rules

    def predict_reaction_rules(
        self, precursor: Precursor, reaction_rules: List[Reactor]
    ) -> Iterator[Tuple[float, Reactor, int]]:
        """Predicts reaction rules using Bayesian-style log-space combination.

        :param precursor: The current precursor for which the reaction rules are predicted.
        :param reaction_rules: The list of reaction rules.
        :return: Yielding (probability, reaction_rule, rule_id) tuples.
        """
        result = self._predict_rules_common(precursor, len(reaction_rules))
        if result is None:
            return

        sorted_probs, sorted_rules = result
        for prob, rule_id in zip(sorted_probs, sorted_rules):
            if prob > self.rule_prob_threshold:
                yield prob, reaction_rules[rule_id], rule_id

    def predict_reaction_rules_light(
        self, precursor: Precursor, reaction_rules_len: int
    ) -> Iterator[Tuple[float, int]]:
        """Predicts reaction rules using Bayesian-style log-space combination.

        Light version without returning Reactor objects.

        :param precursor: The current precursor for which the reaction rules are predicted.
        :param reaction_rules_len: The number of reaction rules.
        :return: Yielding (probability, rule_id) tuples.
        """
        result = self._predict_rules_common(precursor, reaction_rules_len)
        if result is None:
            return

        sorted_probs, sorted_rules = result
        for prob, rule_id in zip(sorted_probs, sorted_rules):
            if prob > self.rule_prob_threshold:
                yield prob, rule_id
