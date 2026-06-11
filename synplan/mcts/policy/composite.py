"""Composite policy merging a filtering and a ranking network over one rule set."""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from synplan.mcts.policy.base import Policy

if TYPE_CHECKING:
    from synplan.chem.precursor import Precursor
    from synplan.chem.reaction import CanonicalRetroReactor
    from synplan.mcts.expansion import CombinedPolicyNetworkFunction


class CompositePolicy(Policy):
    """Wrap a :class:`CombinedPolicyNetworkFunction` behind the :class:`Policy` seam.

    The wrapped function combines a filtering and a ranking policy over the same
    rule set; this class only forwards calls to it.
    """

    #: No single config object: the combined function holds its weighting on
    #: loose attributes (``top_rules``/``ranking_weight``/``temperature``).
    config: Any = None

    def __init__(self, combined_function: "CombinedPolicyNetworkFunction") -> None:
        """Wrap an already-built combined expansion function.

        :param combined_function: A loaded
            :class:`~synplan.mcts.expansion.CombinedPolicyNetworkFunction`.
        """
        self.combined_function = combined_function

    @property
    def n_rules(self) -> int:
        """Forward the combined function's output dimensionality."""
        return self.combined_function.n_rules

    def predict_reaction_rules(
        self,
        precursor: "Precursor",
        reaction_rules: "Sequence[CanonicalRetroReactor]",
    ) -> "Iterator[tuple[float, CanonicalRetroReactor, int]]":
        """Delegate the same-rule-set merge to the wrapped combined function."""
        return self.combined_function.predict_reaction_rules(precursor, reaction_rules)

    def predict_reaction_rules_light(
        self,
        precursor: "Precursor",
        reaction_rules_len: int,
    ) -> "Iterator[tuple[float, int]]":
        """Delegate the Reactor-free merge variant to the combined function."""
        return self.combined_function.predict_reaction_rules_light(
            precursor, reaction_rules_len
        )
