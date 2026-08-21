"""Configuration contracts for opt-in Synthon CLI audit artifacts."""

import pytest
from pydantic import ValidationError

from synplan.synthon.config import SynthonConfig


def test_audit_artifacts_are_opt_in() -> None:
    config = SynthonConfig()

    assert config.write_audit_files is False
    assert config.audit_overwrite == "error"


def test_audit_replace_policy_loads_from_yaml(tmp_path) -> None:
    path = tmp_path / "audit.yaml"
    path.write_text(
        "write_audit_files: true\naudit_overwrite: replace\n",
        encoding="utf-8",
    )

    config = SynthonConfig.from_yaml(str(path))

    assert config.write_audit_files is True
    assert config.audit_overwrite == "replace"


def test_unknown_audit_overwrite_policy_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SynthonConfig(audit_overwrite="append")
