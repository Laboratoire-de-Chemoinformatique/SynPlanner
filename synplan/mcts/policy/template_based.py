"""Template-based policies selecting among a fixed library of reaction-rule templates."""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from synplan.mcts.policy.base import Policy

if TYPE_CHECKING:
    import torch

    from synplan.chem.precursor import Precursor
    from synplan.chem.reaction import CanonicalRetroReactor
    from synplan.mcts.expansion import PolicyNetworkFunction


class TemplateBasedPolicy(Policy):
    """Thin wrapper turning a ``PolicyNetworkFunction`` into a :class:`Policy`.

    The wrapped object is expected to provide ``predict_reaction_rules``,
    ``predict_reaction_rules_light``, ``get_probs``, ``get_logits``, ``n_rules``
    and ``config`` — i.e. the
    :class:`synplan.mcts.expansion.PolicyNetworkFunction` surface. All calls
    delegate straight through; no prediction logic is re-implemented.
    """

    def __init__(self, expansion_function: "PolicyNetworkFunction") -> None:
        """Wrap an already-built expansion function.

        :param expansion_function: A loaded
            :class:`~synplan.mcts.expansion.PolicyNetworkFunction` (or
            subclass) to delegate to.
        """
        self.expansion_function = expansion_function

    @property
    def config(self) -> Any:
        """Forward the wrapped function's configuration object."""
        return getattr(self.expansion_function, "config", None)

    @property
    def n_rules(self) -> int:
        """Forward the wrapped function's output dimensionality."""
        return self.expansion_function.n_rules

    @property
    def top_rules(self) -> Any:
        """Forward the wrapped function's Top-N rule limit, if exposed."""
        return getattr(self.expansion_function, "top_rules", None)

    def predict_reaction_rules(
        self,
        precursor: "Precursor",
        reaction_rules: "Sequence[CanonicalRetroReactor]",
    ) -> "Iterator[tuple[float, CanonicalRetroReactor, int]]":
        """Delegate ranked-rule prediction to the wrapped expansion function."""
        return self.expansion_function.predict_reaction_rules(precursor, reaction_rules)

    def predict_reaction_rules_light(
        self,
        precursor: "Precursor",
        reaction_rules_len: int,
    ) -> "Iterator[tuple[float, int]]":
        """Delegate the Reactor-free variant to the wrapped expansion function."""
        return self.expansion_function.predict_reaction_rules_light(
            precursor, reaction_rules_len
        )

    def get_probs(self, precursor: "Precursor") -> "torch.Tensor | None":
        """Delegate probability computation to the wrapped expansion function."""
        return self.expansion_function.get_probs(precursor)

    def get_logits(self, precursor: "Precursor") -> "torch.Tensor | None":
        """Delegate logit computation to the wrapped expansion function."""
        return self.expansion_function.get_logits(precursor)


class PriorityPolicy(Policy):
    """Curated-rule selector peer (torch-free).

    Priority rules are tried *before* the learned policy on every node. Exposes
    the chython substructure-applicability check deciding whether a curated
    rule's LHS query pattern matches the current precursor.

    Requires no torch: the heavy import of :mod:`synplan.mcts.expansion` is
    deferred to call time so a rules-only planner can construct it.
    """

    config: Any = None

    def __init__(
        self,
        priority_rules: "Sequence[CanonicalRetroReactor]",
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
    def _rule_applies(rule: "CanonicalRetroReactor", precursor: "Precursor") -> bool:
        """Return whether a curated rule's LHS query pattern matches ``precursor``.

        Reuses the chython substructure seam from
        :mod:`synplan.mcts.expansion` (imported lazily to stay torch-free).
        """
        from synplan.mcts.expansion import rule_query_pattern

        pattern = rule_query_pattern(rule)
        if pattern is None:
            return False
        return pattern < precursor.molecule

    def predict_reaction_rules(
        self,
        precursor: "Precursor",
        reaction_rules: "Sequence[CanonicalRetroReactor]",
    ) -> "Iterator[tuple[float, CanonicalRetroReactor, int]]":
        """Yield applicable curated rules with ``prob=1.0``, in rule order.

        Mirrors the priority-set semantics in
        :class:`synplan.mcts.tree.Tree`, where curated rules enter expansion
        with ``prob=1.0`` so a priority disconnect can outrank learned siblings.
        The ``reaction_rules`` argument is accepted for interface parity with
        the learned policies but is unused: a priority policy ranks its own
        curated set.

        :param precursor: The current precursor to expand.
        :param reaction_rules: Ignored; present for :class:`Policy` parity.
        :return: Yields ``(1.0, reaction_rule, rule_id)`` for applicable rules.
        """
        for rule_id, rule in enumerate(self.priority_rules):
            if self._rule_applies(rule, precursor):
                yield 1.0, rule, rule_id
