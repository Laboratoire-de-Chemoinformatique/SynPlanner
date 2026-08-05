"""Composite policy merging a filtering and a ranking policy over one rule set."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import torch

from synplan.mcts.policy.base import Policy

if TYPE_CHECKING:
    from synplan.chem.molecule.precursor import Precursor
    from synplan.chem.reaction import CanonicalRetroReactor


class CompositePolicy(Policy):
    """Merge two policies over one rule set by aligned per-rule logits.

    Combines a filtering and a ranking policy by weighted addition of logits::

        combined_logits = filtering_logits + ranking_weight * ranking_logits
        combined_probs = softmax(combined_logits / temperature)
    """

    config = None

    def __init__(
        self,
        filtering_policy: Policy,
        ranking_policy: Policy,
        *,
        top_rules: int = 50,
        rule_prob_threshold: float = 0.0,
        ranking_weight: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        """Merge a filtering and a ranking policy over the same rule set.

        :param filtering_policy: Policy exposing per-rule applicability logits.
        :param ranking_policy: Policy exposing per-rule feasibility logits.
        :param top_rules: Number of top rules to return.
        :param rule_prob_threshold: Minimum probability to yield a rule.
        :param ranking_weight: Weight applied to ranking logits.
        :param temperature: Softmax temperature for the merged logits.
        """
        self.filtering_policy = filtering_policy
        self.ranking_policy = ranking_policy
        self.top_rules = top_rules
        self.rule_prob_threshold = rule_prob_threshold
        self.ranking_weight = ranking_weight
        self.temperature = temperature

    @property
    def n_rules(self) -> int:
        """Return the shared rule-set dimensionality."""
        return self.filtering_policy.n_rules

    def _validate_dimensions(self, expected_n_rules: int) -> None:
        filtering_dim = self.filtering_policy.n_rules
        ranking_dim = self.ranking_policy.n_rules
        if filtering_dim != expected_n_rules or ranking_dim != expected_n_rules:
            raise Exception(
                f"Policy network output dimensions (filtering={filtering_dim}, "
                f"ranking={ranking_dim}) do not match the number of reaction rules "
                f"({expected_n_rules}). Both policy networks must be trained on the "
                "same set of reaction rules."
            )

    def prepare_rule_associations(
        self, reaction_rules: Sequence[CanonicalRetroReactor]
    ) -> None:
        """Prepare dynamic ranking-side rule associations when needed."""
        prepare_rules = getattr(self.ranking_policy, "prepare_rule_associations", None)
        if prepare_rules is not None:
            prepare_rules(reaction_rules)

    def _get_combined_probs(self, precursor: Precursor) -> torch.Tensor | None:
        """Weighted-logit softmax merge of the two policies, or ``None``."""
        filtering_logits = self.filtering_policy.get_logits(precursor)
        ranking_logits = self.ranking_policy.get_logits(precursor)
        if filtering_logits is None or ranking_logits is None:
            return None
        combined_logits = filtering_logits + self.ranking_weight * ranking_logits
        return torch.softmax(combined_logits / self.temperature, dim=-1)

    def _predict_rules_common(
        self, precursor: Precursor, n_rules: int
    ) -> tuple[list[float], list[int]] | None:
        self._validate_dimensions(n_rules)
        combined_probs = self._get_combined_probs(precursor)
        if combined_probs is None:
            return None
        sorted_probs, sorted_rules = torch.sort(combined_probs, descending=True)
        return (
            sorted_probs[: self.top_rules].tolist(),
            sorted_rules[: self.top_rules].tolist(),
        )

    def predict_reaction_rules(
        self,
        precursor: Precursor,
        reaction_rules: Sequence[CanonicalRetroReactor],
    ) -> Iterator[tuple[float, CanonicalRetroReactor, int]]:
        """Merge the two policies and yield ranked rules above the threshold."""
        self.prepare_rule_associations(reaction_rules)
        result = self._predict_rules_common(precursor, len(reaction_rules))
        if result is None:
            return
        sorted_probs, sorted_rules = result
        for prob, rule_id in zip(sorted_probs, sorted_rules, strict=True):
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
        for prob, rule_id in zip(sorted_probs, sorted_rules, strict=True):
            if prob > self.rule_prob_threshold:
                yield prob, rule_id
