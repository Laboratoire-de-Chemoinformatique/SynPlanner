"""Release helper: links must follow the last tag, not the pyproject version."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.release import (
    last_released_version,
    update_changelog,
    update_switcher,
)

REPO = "https://github.com/Laboratoire-de-Chemoinformatique/SynPlanner"
CHANGELOG = f"""# Changelog

## [Unreleased]

## [1.5.1] - 2026-06-04

[Unreleased]: {REPO}/compare/v1.5.1...HEAD
[1.5.1]: {REPO}/compare/v1.5.0...v1.5.1
"""
SWITCHER = json.dumps(
    [
        {"name": "dev", "version": "latest", "url": "/en/latest/"},
        {"name": "1.5.1 (stable)", "version": "stable", "url": "/en/stable/"},
        {"name": "1.5.0", "version": "v1.5.0", "url": "/en/v1.5.0/"},
    ]
)


def test_compare_link_uses_previous_tag_not_untagged_bump():
    # pyproject was bumped to 1.5.2 but never tagged; prev must stay 1.5.1
    out = update_changelog(CHANGELOG, new="1.6.0", prev="1.5.1", today="2026-08-03")
    assert f"[1.6.0]: {REPO}/compare/v1.5.1...v1.6.0" in out
    assert "v1.5.2" not in out


def test_switcher_keeps_the_last_real_release():
    entries = json.loads(update_switcher(SWITCHER, new="1.6.0", prev="1.5.1"))
    assert entries[1] == {
        "name": "1.6.0 (stable)",
        "version": "stable",
        "url": "/en/stable/",
    }
    assert entries[2]["version"] == "v1.5.1"


def test_last_released_version_matches_git():
    try:
        tags = subprocess.run(
            ["git", "tag", "--list", "v[0-9]*.[0-9]*.[0-9]*", "--sort=-v:refname"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no usable git checkout")
    if not tags:
        pytest.skip("no release tags in this checkout")
    assert last_released_version() == tags[0].removeprefix("v")
