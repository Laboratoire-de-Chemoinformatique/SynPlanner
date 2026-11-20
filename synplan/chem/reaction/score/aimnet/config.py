"""Phase, molecular calculation and reference settings for AIMNet2 thermochemistry."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from synplan.utils.config import BaseConfigModel

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]


LOQI_CPCM_SHA256 = "32f73fcd68e76f67a6e088841304ca0adc96066cca83216dd9ed992e9e4dec87"


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
    also_calculate_gas: bool = False
    phase: Literal["gas", "cpcm_water"] = "gas"
    cpcm_checkpoint: Path | None = None
    cpcm_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stereo_policy: Literal["enumerate", "conformer"] = "enumerate"
    allow_mirror_image: bool = False
    etkdg_fallback: bool = False
    etkdg_max_heavy_atoms: int = Field(default=0, ge=0, strict=True)
    conformer_selection: Literal["optimize_all", "lowest_seed_energy"] = "optimize_all"
    withhold_gas_ion_separation: bool = False
    solvent_concentration_mol_l: PositiveFloat | None = None
    optimizer: Literal["BFGS", "LBFGS"] = "BFGS"
    concentration_mol_l: PositiveFloat | None = None
    imaginary_cutoff_cm1: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    balance_policy: Literal["infer", "validate"] = "infer"
    symmetry_numbers: dict[str, PositiveInt] = Field(default_factory=dict)
    aimnet_checkpoint: Path | None = None
    loqi_checkpoint: Path | None = None

    @model_validator(mode="after")
    def validate_phase(self):
        if self.also_calculate_gas and self.phase != "cpcm_water":
            raise ValueError("also_calculate_gas requires a water calculation")
        if self.phase == "cpcm_water" and not (
            self.cpcm_sha256
            and (self.cpcm_checkpoint or self.cpcm_sha256 == LOQI_CPCM_SHA256)
        ):
            raise ValueError(
                "CPCM water requires a local checkpoint and its full SHA-256"
            )
        if self.solvent_concentration_mol_l is not None and (
            self.phase != "cpcm_water" or self.concentration_mol_l is None
        ):
            raise ValueError(
                "Solvent concentration requires CPCM water and a solute concentration standard state"
            )
        return self

    def molecular_settings(self):
        """Keep compatible gas cache keys, excluding reaction-level policies."""
        settings = self.model_dump(
            mode="json",
            exclude={
                "balance_policy",
                "also_calculate_gas",
                "withhold_gas_ion_separation",
                "solvent_concentration_mol_l",
            },
        )
        for key in (
            "phase",
            "cpcm_checkpoint",
            "cpcm_sha256",
            "optimizer",
            "concentration_mol_l",
            "imaginary_cutoff_cm1",
            "stereo_policy",
            "allow_mirror_image",
            "etkdg_fallback",
            "etkdg_max_heavy_atoms",
            "conformer_selection",
        ):
            if getattr(self, key) == type(self).model_fields[key].default:
                del settings[key]
        return settings


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
