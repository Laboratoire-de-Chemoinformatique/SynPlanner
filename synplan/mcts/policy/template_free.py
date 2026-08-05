"""Template-free policy seam for generative one-step models (abstract stub)."""

from abc import abstractmethod
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

from synplan.mcts.policy.base import Policy

if TYPE_CHECKING:
    from synplan.chem.molecule.precursor import Precursor
    from synplan.chem.reaction import CanonicalRetroReactor


class TemplateFreePolicy(Policy):
    """Abstract stub for generative, template-free one-step policies.

    Intended for RetroChimera-style models that generate disconnections
    directly rather than ranking a fixed template library. NOT IMPLEMENTED:
    concrete subclasses and the generation-to-:class:`Policy` adapter are future
    work. Defined now only to fix the abstraction boundary.
    """

    @property
    @abstractmethod
    def n_rules(self) -> int:
        """Generative policies have no fixed rule dimensionality.

        Concrete subclasses must define what (if anything) this means for a
        template-free model. NOT IMPLEMENTED.
        """

    @abstractmethod
    def predict_reaction_rules(
        self,
        precursor: "Precursor",
        reaction_rules: "Sequence[CanonicalRetroReactor]",
    ) -> "Iterator[tuple[float, CanonicalRetroReactor, int]]":
        """Generate ranked disconnections for ``precursor``. NOT IMPLEMENTED.

        :param precursor: The current precursor to expand.
        :param reaction_rules: Present for :class:`Policy` parity; a
            template-free model typically ignores a fixed rule library.
        """
