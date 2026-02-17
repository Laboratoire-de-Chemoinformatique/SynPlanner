"""Tests for the reaction mapping module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from synplan.chem.data.mapping import MappingConfig, _parse_one
from synplan.interfaces.cli import synplan


class TestMappingConfig:

    def test_defaults(self):
        cfg = MappingConfig()
        assert cfg.batch_size == 16
        assert cfg.chunk_size == 5000
        assert cfg.device is None
        assert cfg.no_amp is False

    def test_custom_values(self):
        cfg = MappingConfig(batch_size=32, chunk_size=1000, device="cpu", no_amp=True)
        assert cfg.batch_size == 32
        assert cfg.device == "cpu"

    def test_bad_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            MappingConfig(batch_size=0)

    def test_bad_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            MappingConfig(chunk_size=-1)

    def test_bad_device(self):
        with pytest.raises(ValueError, match="device"):
            MappingConfig(device="tpu")

    def test_round_trip_yaml(self, tmp_path: Path):
        cfg = MappingConfig(batch_size=4, device="cpu")
        yaml_path = tmp_path / "mapping.yaml"
        cfg.to_yaml(str(yaml_path))

        loaded = MappingConfig.from_yaml(str(yaml_path))
        assert loaded.batch_size == 4
        assert loaded.device == "cpu"
        assert loaded.chunk_size == cfg.chunk_size

    def test_from_yaml_minimal(self, tmp_path: Path):
        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text(yaml.dump({"batch_size": 128}))
        cfg = MappingConfig.from_yaml(str(yaml_path))
        assert cfg.batch_size == 128
        assert cfg.chunk_size == 5000


class TestParseOne:

    def test_valid_reaction(self):
        rxn, err = _parse_one("[CH3:1][OH:2]>>[CH2:1]=[O:2]")
        assert rxn is not None
        assert err is None

    def test_tab_suffix_preserved(self):
        rxn, err = _parse_one("[CH3:1][OH:2]>>[CH2:1]=[O:2]\tUS1234567")
        assert rxn is not None
        assert err is None

    def test_molecule_smiles_rejected(self):
        rxn, err = _parse_one("CCO")
        assert rxn is None
        assert "not a reaction" in err

    def test_invalid_smiles(self):
        rxn, err = _parse_one("NOT_A_SMILES")
        assert rxn is None
        assert err is not None

    def test_empty_string(self):
        rxn, err = _parse_one("")
        assert rxn is None
        assert err is not None


class TestMappingCLI:

    def test_help(self):
        result = CliRunner().invoke(synplan, ["reaction_mapping", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--device" in result.output

    def test_missing_input(self):
        result = CliRunner().invoke(synplan, ["reaction_mapping", "--output", "out.smi"])
        assert result.exit_code != 0

    def test_listed_in_help(self):
        result = CliRunner().invoke(synplan, ["--help"])
        assert "reaction_mapping" in result.output
