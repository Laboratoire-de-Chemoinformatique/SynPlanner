"""Settings for gas-phase AIMNet2 reaction thermochemistry."""

from pathlib import Path
from typing import Annotated

from pydantic import ConfigDict, Field

from synplan.utils.config import BaseConfigModel

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class AIMNetConfig(BaseConfigModel):
    """One-conformer RRHO thermochemistry, with exhaustive missing stereo.

    Checkpoint paths are optional local copies of the released AIMNet2 wB97M-D3
    and LoQI Flow v1.1.0 models. Otherwise upstream checksum-verified registries
    resolve them on first use. Symmetry overrides use isomeric SMILES keys.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    device: str = Field(default="cpu", pattern=r"^(cpu|cuda(:[0-9]+)?)$")
    temperature: PositiveFloat = 298.15
    pressure: PositiveFloat = 101325.0
    n_conformers: PositiveInt = 10
    loqi_steps: PositiveInt = 25
    batch_atoms: PositiveInt = 2000
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    fmax: PositiveFloat = 0.01
    max_steps: PositiveInt = 200
    min_confidence: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    symmetry_numbers: dict[str, PositiveInt] = Field(default_factory=dict)
    aimnet_checkpoint: Path | None = None
    loqi_checkpoint: Path | None = None


FILTER_REASONS = frozenset(
    {
        "unsupported_element",
        "unsupported_charge",
        "radical",
        "unsupported_stereo",
        "unsupported_redox",
        "confidence_below_threshold",
    }
)


class MolecularCalculationError(Exception):
    """An expected chemical/numerical failure, safe to report per species."""

    def __init__(
        self, stage: str, reason: str, detail: str, *, category: str | None = None
    ):
        super().__init__(detail)
        self.stage = stage
        self.reason = reason
        self.category = category or (
            "filtered_out" if reason in FILTER_REASONS else "errored"
        )
        self.reaction_context: dict = {}
        self.conformer_failures: list[dict] = []

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "reason": self.reason,
            "detail": str(self),
            "category": self.category,
        }


def normalize_config(config: AIMNetConfig) -> AIMNetConfig:
    """Resolve equivalent symmetry keys once without changing the caller's config."""
    from .species import canonical_species, parse_species

    symmetry = {}
    for text, number in config.symmetry_numbers.items():
        key = canonical_species(parse_species(text))
        if key in symmetry and symmetry[key] != number:
            raise ValueError(f"Conflicting symmetry numbers for {key}")
        symmetry[key] = number
    return config.model_copy(update={"symmetry_numbers": symmetry}, deep=True)
