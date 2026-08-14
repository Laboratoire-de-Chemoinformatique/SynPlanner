"""Every configs/*.yaml must load through the class its CLI command uses.

All config models are ``extra="forbid"``, so a renamed or removed field turns a
shipped config into a hard ValidationError that no other test would notice.
"""

from pathlib import Path

import pytest
import yaml

from synplan.chem.building_blocks.config import BuildingBlockStockLoadConfig
from synplan.chem.data.filtering import ReactionFilterConfig
from synplan.chem.data.standardizing import ReactionStandardizationConfig
from synplan.chem.synthon.config import SynthonConfig
from synplan.utils.config import (
    CombinedPolicyConfig,
    PolicyNetworkConfig,
    RuleExtractionConfig,
    TreeConfig,
    TuningConfig,
    ValueNetworkConfig,
)

CONFIGS = Path(__file__).resolve().parents[3] / "configs"

# config file -> flat config class (whole file validates as one model)
FLAT = {
    "reactions_standardization.yaml": ReactionStandardizationConfig,
    "reactions_filtration.yaml": ReactionFilterConfig,
    "rules_extraction.yaml": RuleExtractionConfig,
    "extraction_functional_groups.yaml": RuleExtractionConfig,
    "policy_training.yaml": PolicyNetworkConfig,
    "mhn_ranking_policy_training.yaml": PolicyNetworkConfig,
    "building_blocks_stock.yaml": BuildingBlockStockLoadConfig,
    "combined_ranking_filtering_policy.yaml": CombinedPolicyConfig,
    "synthonisation.yaml": SynthonConfig,
}

# config file -> {top-level section: class}, as the CLI splits them
SECTIONED = {
    "planning_standard.yaml": {
        "tree": TreeConfig,
        "node_expansion": PolicyNetworkConfig,
    },
    "planning_value.yaml": {"tree": TreeConfig, "node_expansion": PolicyNetworkConfig},
    "planning_combined_policies.yaml": {
        "tree": TreeConfig,
        "combined_policy": CombinedPolicyConfig,
    },
    "tuning.yaml": {
        "tree": TreeConfig,
        "node_expansion": PolicyNetworkConfig,
        "value_network": ValueNetworkConfig,
        "tuning": TuningConfig,
    },
}


def test_every_shipped_config_is_covered():
    on_disk = {p.name for p in CONFIGS.glob("*.yaml")}
    assert on_disk == set(FLAT) | set(SECTIONED)


@pytest.mark.parametrize("name", sorted(FLAT))
def test_flat_config_loads(name):
    FLAT[name].from_yaml(str(CONFIGS / name))


@pytest.mark.parametrize("name", sorted(SECTIONED))
def test_sectioned_config_loads(name):
    with open(CONFIGS / name, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for section, cls in SECTIONED[name].items():
        cls.from_dict(config[section])
