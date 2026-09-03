"""Configuration for :class:`RetrekRouteScorer`."""

import math
from typing import Literal

from pydantic import Field, model_validator

from synplan.utils.config import BaseConfigModel

ScoreName = Literal["cd", "as", "rd", "st"]

DEFAULT_RETREK_WEIGHTS: dict[ScoreName, float] = {
    "cd": 5.0,
    "as": 0.5,
    "rd": 2.0,
    "st": 2.0,
}


class RetrekRouteScoringConfig(BaseConfigModel):
    """Configuration for RetrekRouteScorer.

    :param enabled_scores: Scores applied to every route step. STScore is not
        enabled by default because it requires the reaction-rule collection.
    :param weights: Relative importance of each score. Aggregation divides by
        the sum of weights for scores available on the current step.
    """

    enabled_scores: tuple[ScoreName, ...] = ("cd", "as", "rd")
    weights: dict[ScoreName, float] = Field(
        default_factory=lambda: dict(DEFAULT_RETREK_WEIGHTS)
    )

    @model_validator(mode="after")
    def validate_scores_and_weights(self):
        """Require a usable, unambiguous weighted score selection."""

        if not self.enabled_scores:
            raise ValueError("at least one ReTReK score must be enabled")
        if len(set(self.enabled_scores)) != len(self.enabled_scores):
            raise ValueError("enabled ReTReK scores must be unique")

        missing = set(self.enabled_scores).difference(self.weights)
        if missing:
            raise ValueError(f"missing ReTReK weights for: {sorted(missing)}")

        enabled_weights = [self.weights[name] for name in self.enabled_scores]
        if any(not math.isfinite(weight) or weight < 0.0 for weight in enabled_weights):
            raise ValueError("ReTReK score weights must be finite and non-negative")
        if not any(weight > 0.0 for weight in enabled_weights):
            raise ValueError(
                "at least one enabled ReTReK score needs a positive weight"
            )
        return self
