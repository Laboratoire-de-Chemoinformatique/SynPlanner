"""Abstract :class:`Policy` interface: the action selector for tree-search node expansion."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from synplan.chem.molecule.precursor import Precursor
    from synplan.chem.reaction import CanonicalRetroReactor


class Policy(ABC):
    """Abstract action selector for tree-search node expansion.

    Concrete policies rank a runtime reaction-rule set for a given precursor and
    yield the top candidates.
    """

    #: Underlying policy configuration object, when one exists; read by the tree
    #: via ``getattr(expansion_function, "config", None)``.
    config: Any

    @property
    @abstractmethod
    def n_rules(self) -> int:
        """Return the fixed output dimensionality (number of reaction rules)."""

    @abstractmethod
    def predict_reaction_rules(
        self,
        precursor: "Precursor",
        reaction_rules: "Sequence[CanonicalRetroReactor]",
    ) -> "Iterator[tuple[float, CanonicalRetroReactor, int]]":
        """Rank ``reaction_rules`` for ``precursor`` and yield the top candidates.

        :param precursor: The current precursor to expand.
        :param reaction_rules: The runtime reaction-rule set to rank.
        :return: Yields ``(probability, reaction_rule, rule_id)`` tuples in
            descending rank order.
        """

    def predict_reaction_rules_light(
        self,
        precursor: "Precursor",
        reaction_rules_len: int,
    ) -> "Iterator[tuple[float, int]]":
        """Reactor-free variant of :meth:`predict_reaction_rules`.

        :param precursor: The current precursor to expand.
        :param reaction_rules_len: The number of reaction rules.
        :return: Yields ``(probability, rule_id)`` tuples in descending rank
            order.
        """
        raise NotImplementedError

    def get_probs(self, precursor: "Precursor") -> "torch.Tensor | None":
        """Return the per-rule probability tensor, or ``None`` on failure.

        :param precursor: The current precursor.
        :return: Probability tensor over all rules, or ``None`` if graph
            conversion / inference fails.
        """
        raise NotImplementedError

    def get_logits(self, precursor: "Precursor") -> "torch.Tensor | None":
        """Return the raw per-rule logits tensor, or ``None`` on failure.

        :param precursor: The current precursor.
        :return: Logits tensor over all rules (before sigmoid/softmax), or
            ``None`` if graph conversion / inference fails.
        """
        raise NotImplementedError

    def __call__(
        self,
        precursor: "Precursor",
        reaction_rules: "Sequence[CanonicalRetroReactor]",
    ) -> "Iterator[tuple[float, CanonicalRetroReactor, int]]":
        """Convenience alias for :meth:`predict_reaction_rules`."""
        return self.predict_reaction_rules(precursor, reaction_rules)
