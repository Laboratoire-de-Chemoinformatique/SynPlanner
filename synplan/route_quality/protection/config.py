"""Configuration for the protection strategy module."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from synplan.utils.config import ConfigABC


_DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class ProtectionConfig(ConfigABC):
    """Configuration for protection-group analysis.

    :param competing_groups_path: Path to YAML file with SMARTS patterns
        for reactive functional groups, organized by category.
    :param incompatibility_path: Path to TSV file with the FG x FG
        incompatibility matrix.
    :param halogen_groups_path: Path to YAML file with halogen SMARTS
        patterns grouped by halogen family.
    :param score_weight: Weight of the protection score S(T) when
        combining with the original route score.  Must be in [0, 1].
    :param enable_reranking: If True, re-rank candidate routes using the
        combined score that includes the protection penalty.
    """

    competing_groups_path: str = str(_DATA_DIR / "competing_groups.yaml")
    incompatibility_path: str = str(_DATA_DIR / "incompatibility_matrix.tsv")
    halogen_groups_path: str = str(_DATA_DIR / "halogen_groups.yaml")
    score_weight: float = 0.5
    enable_reranking: bool = True

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "ProtectionConfig":
        return ProtectionConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> "ProtectionConfig":
        with open(file_path, "r", encoding="utf-8") as fh:
            config_dict = yaml.safe_load(fh)
        return ProtectionConfig.from_dict(config_dict)

    def _validate_params(self, params: Dict[str, Any]) -> None:
        # competing_groups_path
        if not isinstance(params["competing_groups_path"], str):
            raise ValueError("competing_groups_path must be a string.")

        # incompatibility_path
        if not isinstance(params["incompatibility_path"], str):
            raise ValueError("incompatibility_path must be a string.")

        # halogen_groups_path
        if not isinstance(params["halogen_groups_path"], str):
            raise ValueError("halogen_groups_path must be a string.")

        # score_weight
        if not isinstance(params["score_weight"], (int, float)):
            raise ValueError("score_weight must be a float.")
        if not (0.0 <= float(params["score_weight"]) <= 1.0):
            raise ValueError("score_weight must be in [0.0, 1.0].")

        # enable_reranking
        if not isinstance(params["enable_reranking"], bool):
            raise ValueError("enable_reranking must be a boolean.")
