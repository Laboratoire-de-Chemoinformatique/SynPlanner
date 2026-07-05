"""Template-based policies selecting among a fixed library of reaction-rule templates."""

from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import torch

from synplan.chem.reaction.rules import rule_query_pattern
from synplan.chem.reaction.rules.representation import (
    rule_representation_digest,
    rule_smarts_from_reactors,
)
from synplan.mcts.policy.base import Policy
from synplan.ml.featurization.fingerprints import rule_fingerprints_from_smarts
from synplan.ml.featurization.molecules import mol_to_pyg
from synplan.ml.featurization.rules import query_cgr_graphs_from_smarts

if TYPE_CHECKING:
    import torch_geometric
    import torch_geometric.data

    from synplan.chem.precursor import Precursor
    from synplan.chem.reaction import CanonicalRetroReactor
    from synplan.ml.networks.policy.linear import (
        FilteringPolicyNetwork,
        RankingPolicyNetwork,
    )
    from synplan.ml.networks.policy.mhnreact import MHNReact

_MAX_RULE_ASSOCIATION_CACHE_SIZE = 4


class TemplateBasedPolicy(Policy):
    """Rank a fixed reaction-rule library for a precursor via a learned network.

    Owns the shared logic the old expansion ``*Function`` layer carried:
    precursor → graph featurization, top-k iteration and probability
    thresholding. Subclasses supply :meth:`get_logits` / :meth:`get_probs`.
    """

    config = None

    def __init__(
        self,
        policy_net,
        *,
        top_rules: int = 50,
        rule_prob_threshold: float = 0.0,
        priority_rules_fraction: float = 0.5,
    ) -> None:
        """Wrap a pure policy network with its selection knobs.

        :param policy_net: A pure policy network in eval mode.
        :param top_rules: Number of top rules to return.
        :param rule_prob_threshold: Minimum probability to yield a rule.
        :param priority_rules_fraction: Filtering rule/priority head mix weight.
        """
        self.policy_net = policy_net.eval()
        self.top_rules = top_rules
        self.rule_prob_threshold = rule_prob_threshold
        self.priority_rules_fraction = priority_rules_fraction

    @property
    def architecture(self) -> str:
        """Return the wrapped network architecture."""
        return getattr(self.policy_net, "architecture", "linear")

    @property
    def n_rules(self) -> int:
        """Return the network output dimensionality."""
        return self.policy_net.n_rules

    def _get_graph(self, precursor: Precursor) -> torch_geometric.data.Data | None:
        """Convert a precursor molecule to a PyG graph."""
        return mol_to_pyg(precursor.molecule, canonicalize=False)

    @abstractmethod
    def get_logits(self, precursor: Precursor) -> torch.Tensor | None:
        """Return raw per-rule logits, or ``None`` if featurization fails."""

    @abstractmethod
    def get_probs(self, precursor: Precursor) -> torch.Tensor | None:
        """Return per-rule probabilities, or ``None`` if featurization fails."""

    def _predict_rules_common(
        self, precursor: Precursor, n_rules: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Top-k probabilities and rule ids for a precursor, or ``None``."""
        out_dim = self.n_rules
        if out_dim != n_rules:
            raise Exception(
                f"The policy network output dimensionality is {out_dim}, but the "
                f"number of reaction rules is {n_rules}. Probably you use a different "
                "version of the policy network. Be sure to retain the policy network "
                "with the current set of reaction rules"
            )
        probs = self.get_probs(precursor)
        if probs is None:
            return None
        k = min(self.top_rules, probs.numel())
        sorted_probs, sorted_rules = torch.topk(probs, k=k, sorted=True)
        if getattr(self.policy_net, "policy_type", "ranking") == "filtering":
            sorted_probs = torch.softmax(sorted_probs, -1)
        return sorted_probs, sorted_rules

    def predict_reaction_rules(
        self,
        precursor: Precursor,
        reaction_rules: Sequence[CanonicalRetroReactor],
    ) -> Iterator[tuple[float, CanonicalRetroReactor, int]]:
        """Yield ``(prob, reaction_rule, rule_id)`` above the threshold."""
        result = self._predict_rules_common(precursor, len(reaction_rules))
        if result is None:
            return
        sorted_probs, sorted_rules = result
        for prob, rule_id in zip(
            sorted_probs.tolist(), sorted_rules.tolist(), strict=True
        ):
            if prob > self.rule_prob_threshold:
                yield prob, reaction_rules[rule_id], rule_id

    def predict_reaction_rules_light(
        self,
        precursor: Precursor,
        reaction_rules_len: int,
    ) -> Iterator[tuple[float, int]]:
        """Reactor-free variant of :meth:`predict_reaction_rules`."""
        result = self._predict_rules_common(precursor, reaction_rules_len)
        if result is None:
            return
        sorted_probs, sorted_rules = result
        for prob, rule_id in zip(
            sorted_probs.tolist(), sorted_rules.tolist(), strict=True
        ):
            if prob > self.rule_prob_threshold:
                yield prob, rule_id


class LinearPolicy(TemplateBasedPolicy):
    """Template policy over a fixed library (ranking or filtering network)."""

    policy_net: RankingPolicyNetwork | FilteringPolicyNetwork

    def get_logits(self, precursor: Precursor) -> torch.Tensor | None:
        """Return raw per-rule logits (before sigmoid/softmax)."""
        pyg_graph = self._get_graph(precursor)
        if not pyg_graph:
            return None
        with torch.no_grad():
            x = self.policy_net.embedder(pyg_graph)
            return self.policy_net.y_predictor(x)[0].double()

    def get_probs(self, precursor: Precursor) -> torch.Tensor | None:
        """Return per-rule probabilities, mixing priority for filtering nets."""
        pyg_graph = self._get_graph(precursor)
        if not pyg_graph:
            return None
        with torch.no_grad():
            if self.policy_net.policy_type == "filtering":
                probs, priority = self.policy_net(pyg_graph)
                probs = probs[0].double()
                priority = priority[0].double()
                coef = self.priority_rules_fraction
                return (1 - coef) * probs + coef * priority
            return self.policy_net(pyg_graph)[0].double()

    def get_filtering_probs_only(self, precursor: Precursor) -> torch.Tensor | None:
        """Return the filtering rule head (sigmoid) without priority mixing."""
        if self.policy_net.policy_type != "filtering":
            raise ValueError("This method is only for filtering policy networks")
        logits = self.get_logits(precursor)
        return torch.sigmoid(logits) if logits is not None else None


class MHNReactPolicy(TemplateBasedPolicy):
    """MHN ranking policy scoring a runtime rule set via cached associations."""

    policy_net: MHNReact

    def __init__(self, policy_net, **kwargs) -> None:
        super().__init__(policy_net, **kwargs)
        if self.architecture != "mhn_ranking":
            raise ValueError("MHNReactPolicy requires an mhn_ranking network")
        self._rule_representation_digest: str | None = None
        self._rule_associations: torch.Tensor | None = None
        self._rule_association_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    @property
    def n_rules(self) -> int:
        """Return the currently bound runtime rule dimensionality."""
        if self._rule_associations is not None:
            return self._rule_associations.shape[0]
        return self.policy_net.n_rules

    def prepare_rule_associations(
        self, reaction_rules: Sequence[CanonicalRetroReactor]
    ) -> None:
        """Encode a runtime rule set once for MHN ranking prediction."""
        rule_smarts = rule_smarts_from_reactors(reaction_rules)
        representation_config = self.policy_net.rule_representation_config
        digest = rule_representation_digest(rule_smarts, representation_config)
        if self._rule_representation_digest == digest:
            return

        associations = self._rule_association_cache.get(digest)
        if associations is None:
            if representation_config.embedding_type == "fingerprint":
                rule_representations = rule_fingerprints_from_smarts(
                    rule_smarts, representation_config.fingerprint_config
                )
            else:
                rule_representations = query_cgr_graphs_from_smarts(
                    rule_smarts,
                    schema_version=representation_config.graph_schema_version,
                )
            with torch.no_grad():
                associations = self.policy_net.encode_rules(
                    rule_representations
                ).detach()
            self._rule_association_cache[digest] = associations
            self._rule_association_cache.move_to_end(digest)
            while len(self._rule_association_cache) > _MAX_RULE_ASSOCIATION_CACHE_SIZE:
                self._rule_association_cache.popitem(last=False)
        else:
            self._rule_association_cache.move_to_end(digest)
        self._rule_associations = associations
        self._rule_representation_digest = digest

    def get_logits(self, precursor: Precursor) -> torch.Tensor | None:
        """Return MHN logits for the currently prepared rule associations."""
        pyg_graph = self._get_graph(precursor)
        if not pyg_graph:
            return None
        if self._rule_associations is None:
            raise ValueError(
                "mhn_ranking rules are prepared by predict_reaction_rules()."
            )
        with torch.no_grad():
            return self.policy_net.get_logits(
                pyg_graph, rule_associations=self._rule_associations
            )[0].double()

    def get_probs(self, precursor: Precursor) -> torch.Tensor | None:
        """Return MHN ranking probabilities for prepared runtime rules."""
        logits = self.get_logits(precursor)
        return torch.softmax(logits, dim=-1) if logits is not None else None

    def predict_reaction_rules(
        self,
        precursor: Precursor,
        reaction_rules: Sequence[CanonicalRetroReactor],
    ) -> Iterator[tuple[float, CanonicalRetroReactor, int]]:
        """Prepare the runtime rule set, then rank it for the precursor."""
        self.prepare_rule_associations(reaction_rules)
        yield from super().predict_reaction_rules(precursor, reaction_rules)


class PriorityPolicy(Policy):
    """Curated-rule selector trying chython substructure matches before learning."""

    config = None

    def __init__(
        self,
        priority_rules: Sequence[CanonicalRetroReactor],
    ) -> None:
        """Build a priority selector from a curated rule set.

        :param priority_rules: The curated reaction rules to try first.
        """
        self.priority_rules: tuple[CanonicalRetroReactor, ...] = tuple(priority_rules)

    @property
    def n_rules(self) -> int:
        """Return the number of curated priority rules."""
        return len(self.priority_rules)

    @staticmethod
    def _rule_applies(rule: CanonicalRetroReactor, precursor: Precursor) -> bool:
        """Return whether a curated rule's LHS query pattern matches ``precursor``."""
        pattern = rule_query_pattern(rule)
        if pattern is None:
            return False
        try:
            return pattern < precursor.molecule
        except TypeError:
            return False

    def predict_reaction_rules(
        self,
        precursor: Precursor,
        reaction_rules: Sequence[CanonicalRetroReactor],
    ) -> Iterator[tuple[float, CanonicalRetroReactor, int]]:
        """Yield applicable curated rules with ``prob=1.0``, in rule order."""
        for rule_id, rule in enumerate(self.priority_rules):
            if self._rule_applies(rule, precursor):
                yield 1.0, rule, rule_id
