"""The shipped SAScore benchmark must call the library with arguments it accepts.

``sascore-benchmark`` is a console script nothing else exercises: every run
raised ``TypeError: load_combined_policy_function() got an unexpected keyword
argument 'priority_rules_fraction'`` right after the file checks. ``autospec``
re-checks the real signature, so the same drift fails here instead of at the
user's first run.
"""

from pathlib import Path
from unittest import mock

import yaml

from scripts.sascore_bench import run_benchmark
from synplan.utils import loading

CONFIG_PATH = Path(run_benchmark.DEFAULT_CONFIG_PATH)


def _config_with_stub_data(tmp_path):
    """Rewrite the shipped config so its data paths point at empty local files."""
    config = run_benchmark.load_config(CONFIG_PATH)
    data = tmp_path / "data"
    for key in (
        "ranking_policy",
        "filtering_policy",
        "reaction_rules",
        "building_blocks",
    ):
        path = data / config["paths"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    config["paths"]["data_folder"] = str(data)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _autospec_loaders(monkeypatch):
    policy = mock.create_autospec(loading.load_combined_policy_function)
    monkeypatch.setattr(run_benchmark, "load_combined_policy_function", policy)
    monkeypatch.setattr(
        run_benchmark,
        "load_reaction_rules",
        mock.create_autospec(loading.load_reaction_rules),
    )
    monkeypatch.setattr(
        run_benchmark,
        "load_building_blocks",
        mock.create_autospec(loading.load_building_blocks),
    )
    return policy


def test_shipped_config_builds_the_benchmark_resources(tmp_path, monkeypatch):
    policy = _autospec_loaders(monkeypatch)

    resources = run_benchmark.load_resources_from_config(
        _config_with_stub_data(tmp_path)
    )

    assert policy.call_count == 1
    # main() takes its whole setup from here, so this covers the console script
    assert resources["tree_config"].search_strategy == "expansion_first"


def test_shipped_config_builds_the_policy_alone(tmp_path, monkeypatch):
    policy = _autospec_loaders(monkeypatch)

    run_benchmark.load_policy_from_config(_config_with_stub_data(tmp_path))

    assert policy.call_count == 1
