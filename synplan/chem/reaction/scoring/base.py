"""Base classes and shared constants for reaction-level scoring."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from chython import MoleculeContainer

from synplan.chem.reaction import CanonicalRetroReactor

UNAVAILABLE = float("nan")


@dataclass(frozen=True)
class ReactionScoreContext:
    """Context object carrying all data needed to compute a reaction score.

    :param product: The route product being disconnected.
    :param new_precursors: Reactants of the forward reaction.
    :param available_precursors: Availability verdict aligned with
        ``new_precursors``. ``None`` means that availability is unknown.
    :param rule: The retro reaction rule that was applied.
    """

    product: MoleculeContainer
    new_precursors: tuple[MoleculeContainer, ...]
    available_precursors: tuple[bool, ...] | None = None
    rule: CanonicalRetroReactor | None = None


class AbstractReactionScore(ABC):
    """Abstract base class for all reaction-level scoring functions."""

    @abstractmethod
    def compute(self, context: ReactionScoreContext) -> float:
        """Compute the score for a single reaction.

        :param context: ReactionScoreContext carrying all reaction data.
        :returns: Score as a float. Return UNAVAILABLE if the score
            cannot be computed for this reaction.
        """
        ...
