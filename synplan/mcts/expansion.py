"""Module containing a class that represents a policy function for node expansion in the
tree search."""

from typing import Iterator, List, Tuple, Union

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

        out_dim = list(self.policy_net.modules())[-1].out_features
        if out_dim != len(reaction_rules):
            raise Exception(
                f"The policy network output dimensionality is {out_dim}, but the number of reaction rules is {len(reaction_rules)}. "
                "Probably you use a different version of the policy network. Be sure to retain the policy network "
                "with the current set of reaction rules"
            )

        pyg_graph = mol_to_pyg(precursor.molecule, canonicalize=False)
        if pyg_graph:
            with torch.no_grad():
                if self.policy_net.policy_type == "filtering":
                    probs, priority = self.policy_net.forward(pyg_graph)
                if self.policy_net.policy_type == "ranking":
                    probs = self.policy_net.forward(pyg_graph)
            del pyg_graph
        else:
            return []

        probs = probs[0].double()
        if self.policy_net.policy_type == "filtering":
            priority = priority[0].double()
            priority_coef = self.config.priority_rules_fraction
            probs = (1 - priority_coef) * probs + priority_coef * priority

        sorted_probs, sorted_rules = torch.sort(probs, descending=True)
        sorted_probs, sorted_rules = (
            sorted_probs[: self.config.top_rules],
            sorted_rules[: self.config.top_rules],
        )

        if self.policy_net.policy_type == "filtering":
            sorted_probs = torch.softmax(sorted_probs, -1)

        sorted_probs, sorted_rules = sorted_probs.tolist(), sorted_rules.tolist()

        for prob, rule_id in zip(sorted_probs, sorted_rules):
            if (
                prob > self.config.rule_prob_threshold
            ):  # search may fail if rule_prob_threshold is too low (recommended value is 0.0)
                yield prob, reaction_rules[rule_id], rule_id

    def predict_reaction_rules_light(
        self, precursor: Precursor, reaction_rules_len: int
    ) -> Iterator[Union[Iterator, Iterator[Tuple[float, Reactor, int]]]]:
        """The policy function predicts the list of reaction rules for a given precursor.

        :param precursor: The current precursor for which the reaction rules are predicted.
        :param reaction_rules: The list of reaction rules from which applicable reaction
            rules are predicted and selected.
        :return: Yielding the predicted probability for the reaction rule, reaction rule
            and reaction rule id.
        """

        out_dim = list(self.policy_net.modules())[-1].out_features
        if out_dim != reaction_rules_len:
            raise Exception(
                f"The policy network output dimensionality is {out_dim}, but the number of reaction rules is {reaction_rules_len}. "
                "Probably you use a different version of the policy network. Be sure to retain the policy network "
                "with the current set of reaction rules"
            )

        pyg_graph = mol_to_pyg(precursor.molecule, canonicalize=False)
        if pyg_graph:
            with torch.no_grad():
                if self.policy_net.policy_type == "filtering":
                    probs, priority = self.policy_net.forward(pyg_graph)
                if self.policy_net.policy_type == "ranking":
                    probs = self.policy_net.forward(pyg_graph)
            del pyg_graph
        else:
            return []

        probs = probs[0].double()
        if self.policy_net.policy_type == "filtering":
            priority = priority[0].double()
            priority_coef = self.config.priority_rules_fraction
            probs = (1 - priority_coef) * probs + priority_coef * priority

        sorted_probs, sorted_rules = torch.sort(probs, descending=True)
        sorted_probs, sorted_rules = (
            sorted_probs[: self.config.top_rules],
            sorted_rules[: self.config.top_rules],
        )

        if self.policy_net.policy_type == "filtering":
            sorted_probs = torch.softmax(sorted_probs, -1)

        sorted_probs, sorted_rules = sorted_probs.tolist(), sorted_rules.tolist()

        for prob, rule_id in zip(sorted_probs, sorted_rules):
            if (
                prob > self.config.rule_prob_threshold
            ):  # search may fail if rule_prob_threshold is too low (recommended value is 0.0)
                yield prob, rule_id

    def get_raw_probs(self, precursor: Precursor) -> torch.Tensor:
        """Get raw probability tensor from the policy network without sorting or filtering.

        :param precursor: The current precursor for which the reaction rules are predicted.
        :return: Raw probability tensor for all rules.
        """
        pyg_graph = mol_to_pyg(precursor.molecule, canonicalize=False)
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


class CombinedPolicyNetworkFunction:
    """Combined policy function that multiplies filtering and ranking policy probabilities.

    The filtering policy provides applicability probabilities (sigmoid outputs),
    while the ranking policy provides relative ranking probabilities (softmax outputs).
    The combined probability is their element-wise product, allowing filtering to
    effectively mask out inapplicable rules while ranking prioritizes among applicable ones.
    """

    def __init__(
        self,
        filtering_config: PolicyNetworkConfig,
        ranking_config: PolicyNetworkConfig,
        top_rules: int = 50,
        rule_prob_threshold: float = 0.0,
        priority_rules_fraction: float = 0.5,
    ) -> None:
        """Initializes the combined policy function with both filtering and ranking networks.

        :param filtering_config: Configuration for the filtering policy network.
        :param ranking_config: Configuration for the ranking policy network.
        :param top_rules: Number of top rules to return.
        :param rule_prob_threshold: Minimum probability threshold for returning a rule.
        :param priority_rules_fraction: Coefficient for combining priority rules in filtering.
        """
        # Ensure configs have correct policy types
        if filtering_config.policy_type != "filtering":
            raise ValueError(
                f"filtering_config must have policy_type='filtering', got '{filtering_config.policy_type}'"
            )
        if ranking_config.policy_type != "ranking":
            raise ValueError(
                f"ranking_config must have policy_type='ranking', got '{ranking_config.policy_type}'"
            )

        # Override priority_rules_fraction in filtering config
        filtering_config.priority_rules_fraction = priority_rules_fraction

        self.filtering_net = PolicyNetworkFunction(filtering_config)
        self.ranking_net = PolicyNetworkFunction(ranking_config)
        self.top_rules = top_rules
        self.rule_prob_threshold = rule_prob_threshold

    def predict_reaction_rules(
        self, precursor: Precursor, reaction_rules: List[Reactor]
    ) -> Iterator[Union[Iterator, Iterator[Tuple[float, Reactor, int]]]]:
        """Predicts reaction rules by combining filtering and ranking policy probabilities.

        :param precursor: The current precursor for which the reaction rules are predicted.
        :param reaction_rules: The list of reaction rules from which applicable reaction
            rules are predicted and selected.
        :return: Yielding the predicted probability for the reaction rule, reaction rule
            and reaction rule id.
        """
        # Validate output dimensions match
        filtering_out_dim = list(self.filtering_net.policy_net.modules())[-1].out_features
        ranking_out_dim = list(self.ranking_net.policy_net.modules())[-1].out_features

        if filtering_out_dim != len(reaction_rules) or ranking_out_dim != len(reaction_rules):
            raise Exception(
                f"Policy network output dimensions (filtering={filtering_out_dim}, ranking={ranking_out_dim}) "
                f"do not match the number of reaction rules ({len(reaction_rules)}). "
                "Both policy networks must be trained on the same set of reaction rules."
            )

        # Get raw probabilities from both networks
        filtering_probs = self.filtering_net.get_raw_probs(precursor)
        ranking_probs = self.ranking_net.get_raw_probs(precursor)

        if filtering_probs is None or ranking_probs is None:
            return []

        # Multiply probabilities element-wise
        combined_probs = filtering_probs * ranking_probs

        # Sort and select top rules
        sorted_probs, sorted_rules = torch.sort(combined_probs, descending=True)
        sorted_probs, sorted_rules = (
            sorted_probs[: self.top_rules],
            sorted_rules[: self.top_rules],
        )

        sorted_probs, sorted_rules = sorted_probs.tolist(), sorted_rules.tolist()

        for prob, rule_id in zip(sorted_probs, sorted_rules):
            if prob > self.rule_prob_threshold:
                yield prob, reaction_rules[rule_id], rule_id

    def predict_reaction_rules_light(
        self, precursor: Precursor, reaction_rules_len: int
    ) -> Iterator[Union[Iterator, Iterator[Tuple[float, int]]]]:
        """Predicts reaction rules (light version without returning Reactor objects).

        :param precursor: The current precursor for which the reaction rules are predicted.
        :param reaction_rules_len: The number of reaction rules.
        :return: Yielding the predicted probability and reaction rule id.
        """
        # Validate output dimensions match
        filtering_out_dim = list(self.filtering_net.policy_net.modules())[-1].out_features
        ranking_out_dim = list(self.ranking_net.policy_net.modules())[-1].out_features

        if filtering_out_dim != reaction_rules_len or ranking_out_dim != reaction_rules_len:
            raise Exception(
                f"Policy network output dimensions (filtering={filtering_out_dim}, ranking={ranking_out_dim}) "
                f"do not match the number of reaction rules ({reaction_rules_len}). "
                "Both policy networks must be trained on the same set of reaction rules."
            )

        # Get raw probabilities from both networks
        filtering_probs = self.filtering_net.get_raw_probs(precursor)
        ranking_probs = self.ranking_net.get_raw_probs(precursor)

        if filtering_probs is None or ranking_probs is None:
            return []

        # Sum probabilities element-wise
        combined_probs = filtering_probs + ranking_probs

        # Sort and select top rules
        sorted_probs, sorted_rules = torch.sort(combined_probs, descending=True)
        sorted_probs, sorted_rules = (
            sorted_probs[: self.top_rules],
            sorted_rules[: self.top_rules],
        )

        sorted_probs, sorted_rules = sorted_probs.tolist(), sorted_rules.tolist()

        for prob, rule_id in zip(sorted_probs, sorted_rules):
            if prob > self.rule_prob_threshold:
                yield prob, rule_id
