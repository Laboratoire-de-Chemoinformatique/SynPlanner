"""Every presets/*.yaml must be a manifest ``download_preset`` can hand to the loaders.

A preset pairing a ranking head with a filtering head from a *different* rule
set parses fine, downloads fine, and only fails deep inside planning when
:class:`~synplan.mcts.policy.CompositePolicy` compares logit widths. The
``synplanner-gps`` manifest on HuggingFace does exactly that (11235-rule GPS
ranking head, 24094-rule GCN filtering head), so check the invariant here.
"""

from pathlib import Path

import pytest
import yaml

PRESETS = Path(__file__).resolve().parents[3] / "presets"

# Keys the loaders and tutorials index the download_preset() result by.
COMPONENTS = {
    "reaction_rules",
    "ranking_policy",
    "filtering_policy",
    "value_network",
    "building_blocks",
}

NAMES = sorted(p.stem for p in PRESETS.glob("*.yaml"))


def _load(name):
    with open(PRESETS / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_every_shipped_preset_is_covered():
    assert {p.name for p in PRESETS.glob("*.yaml")} == {f"{n}.yaml" for n in NAMES}
    assert NAMES, "no preset manifests on disk"


@pytest.mark.parametrize("name", NAMES)
def test_preset_manifest_shape(name):
    preset = _load(name)
    assert preset["name"] == name
    assert preset["description"]
    files = preset["files"]
    assert set(files) <= COMPONENTS, (
        f"unknown component keys: {set(files) - COMPONENTS}"
    )
    assert "reaction_rules" in files


@pytest.mark.parametrize("name", NAMES)
def test_preset_policies_share_the_rule_set(name):
    """Both policy heads must come from the folder that owns the rule file."""
    files = _load(name)["files"]
    rules_dir = str(Path(files["reaction_rules"]).parent)
    for key in ("ranking_policy", "filtering_policy"):
        if key in files:
            assert str(files[key]).startswith(rules_dir + "/"), (
                f"{name}: {key} lives outside {rules_dir}, so it was trained on "
                "another rule set and its logits cannot be combined"
            )
