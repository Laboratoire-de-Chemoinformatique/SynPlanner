"""Configuration for the Synt-On port. Domain configs live next to their code, not in utils."""

import json
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from synplan.utils.config import BaseConfigModel
from synplan.utils.parallel import default_num_workers

_DATA_DIR = Path(__file__).resolve().parent / "data"


class SynthonConfig(BaseConfigModel):
    """Every knob the port exposes. The eight labels are NOT one of them — they are fixed."""

    # data
    classes_path: str = str(_DATA_DIR / "bb_classes.json")
    marks_path: str = str(_DATA_DIR / "bb_marks.json")
    rules_path: str = str(_DATA_DIR / "rules.json")
    # BB synthonisation
    keep_protecting_groups: bool = False  # the reference's keepPG
    ignore_solvents: bool = True
    max_components: int = Field(4, ge=1)
    # fragmentation
    max_stages: int = Field(5, ge=1)  # MaxNumberOfStages
    max_rc_per_fragment: int = Field(3, ge=1)  # maxNumberOfReactionCentersPerFragment
    max_pathways: int = Field(10_000, ge=1)  # NEW - upstream bounds depth, never width
    rule_mode: Literal["use_all", "include_only", "exclude_some", "one_by_one"] = (
        "use_all"
    )
    rules_selection: str = "R1-R13"
    fragments_to_ignore: list[str] = []
    availability_denominator: Literal["target", "pathway"] = "target"
    # enumeration
    max_reacted_synthons: int = Field(6, ge=2)
    max_products: int = Field(1_000, ge=1)  # desiredNumberOfNewMols
    mw_lower: float = Field(100.0, ge=0.0)
    mw_upper: float = Field(1000.0, ge=0.0)
    # ring sizes a heterocyclisation may close; () disables ring closure and restores the
    # acyclic-only behaviour exactly. 5 and 6 cover every azole and azine, 7 the diazepines.
    ring_closure_sizes: tuple[int, ...] = (5, 6, 7)
    # analogues
    find_analogues: bool = False
    similarity_threshold: float = -1.0  # simTh; -1 disables
    pas_removal_direction: bool = (
        True  # upstream's removal branch is unsatisfiable; we fix it
    )
    ro2_filtration: bool = False
    ro2_variant: Literal["paper", "corrected"] = "paper"
    strict_availability: bool = False
    # runtime
    num_workers: int = Field(default_factory=default_num_workers, ge=1)
    time_budget_s: float | None = Field(
        None, gt=0.0
    )  # max_products bounds output, not work
    # CLI audit artifacts
    write_audit_files: bool = False
    audit_overwrite: Literal["error", "replace"] = "error"

    @model_validator(mode="after")
    def _mw_window_is_non_empty(self) -> "SynthonConfig":
        if self.mw_lower > self.mw_upper:
            raise ValueError(f"mw_lower {self.mw_lower} > mw_upper {self.mw_upper}")
        return self


@cache
def load_data(path: str) -> Any:
    """Read one committed data file. Cached: the 2401 patterns are parsed once per process."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["SynthonConfig", "load_data"]
