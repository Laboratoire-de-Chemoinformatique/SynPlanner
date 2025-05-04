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
