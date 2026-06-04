"""Configuration for the protection strategy module."""

from pathlib import Path

from pydantic import Field

from synplan.utils.config import BaseConfigModel

_DATA_DIR = Path(__file__).resolve().parent / "data"


class ProtectionConfig(BaseConfigModel):
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
    score_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_reranking: bool = True
