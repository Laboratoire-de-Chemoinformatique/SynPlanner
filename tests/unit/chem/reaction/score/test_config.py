"""Shipped YAML, CLI precedence and stable public imports without model loading."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts import assess_forward_routes, assess_routes, assess_thermochemistry
from synplan.chem.reaction.score import AIMNetConfig, AIMNetReactionAssessor
from synplan.chem.reaction.score.aimnet import AIMNetConfig as PublicConfig
from synplan.chem.reaction.score.aimnet import AIMNetReactionAssessor as PublicAssessor

from .test_aimnet import Calculator, route

REPOSITORY = Path(__file__).resolve().parents[5]


def test_public_imports_and_compatibility_entry_points():
    assert PublicConfig is AIMNetConfig
    assert PublicAssessor is AIMNetReactionAssessor
    assert (
        assess_routes.assess_collection
        is assess_forward_routes.assess_forward_collection
    )
    assert assess_routes.main is assess_forward_routes.main
    assert (
        assess_thermochemistry.run_assessment
        is assess_thermochemistry.run_thermochemistry_assessment
    )


def test_shipped_yaml_to_assessor(tmp_path):
    config = AIMNetConfig.from_yaml(str(REPOSITORY / "configs/aimnet_thermo.yaml"))
    assert config == AIMNetConfig()
    calculator = Calculator()
    assessor = AIMNetReactionAssessor(
        config, tmp_path / "cache.json", calculator=calculator
    )
    assert (
        assessor.assess([route("O>>O")])[0].steps[0].thermochemistry["status"] == "ok"
    )
    assert calculator.calls == ["O"]


def test_yaml_round_trip_and_cli_overrides(tmp_path):
    config = AIMNetConfig(
        device="cuda:1",
        temperature=350,
        pressure=200000,
        symmetry_numbers={"[OH2]": 2},
        aimnet_checkpoint=Path("weights/aimnet.pt"),
        loqi_checkpoint=Path("weights/loqi.ckpt"),
    )
    path = tmp_path / "settings.yaml"
    config.to_yaml(str(path))
    assert AIMNetConfig.from_yaml(str(path)) == config
    args = Namespace(config=path, device=None, aimnet_model=None, loqi_model=None)
    assert assess_thermochemistry.config_from_arguments(args) == config
    args.device = "cpu"
    args.aimnet_model = tmp_path / "override.pt"
    merged = assess_thermochemistry.config_from_arguments(args)
    assert merged.device == "cpu"
    assert merged.aimnet_checkpoint == args.aimnet_model
    assert merged.loqi_checkpoint == Path("weights/loqi.ckpt")
    assert merged.temperature == 350
    assert merged.symmetry_numbers == {"[OH2]": 2}
    args.device = "invalid"
    with pytest.raises(ValidationError):
        assess_thermochemistry.config_from_arguments(args)


def test_cli_without_yaml_keeps_cpu_defaults():
    args = Namespace(config=None, device=None, aimnet_model=None, loqi_model=None)
    assert assess_thermochemistry.config_from_arguments(args) == AIMNetConfig()


@pytest.mark.parametrize(
    "contents", ["unknown: true", "temperature: -1", "n_conformers: 0"]
)
def test_invalid_yaml_is_rejected_before_execution(tmp_path, monkeypatch, contents):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assess_thermochemistry",
            "--config",
            str(path),
            "--input",
            "unused.json",
            "--output",
            str(tmp_path / "results"),
            "--cache",
            str(tmp_path / "cache.json"),
        ],
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid configuration reached assessment")

    monkeypatch.setattr(
        assess_thermochemistry, "run_thermochemistry_assessment", unexpected
    )
    with pytest.raises(ValidationError):
        assess_thermochemistry.main()
    assert not (tmp_path / "results").exists()


def test_cli_passes_loaded_yaml_to_runner(tmp_path, monkeypatch):
    import torch

    path = tmp_path / "settings.yaml"
    AIMNetConfig(temperature=310, device="cuda").to_yaml(str(path))
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assess_thermochemistry",
            "--config",
            str(path),
            "--device",
            "cpu",
            "--input",
            "routes.json",
            "--output",
            "results",
            "--cache",
            "cache.json",
        ],
    )
    captured = []

    def run(source, output, config, cache, **kwargs):
        captured.append(config)
        return {}

    monkeypatch.setattr(assess_thermochemistry, "run_thermochemistry_assessment", run)
    assess_thermochemistry.main()
    assert captured == [AIMNetConfig(temperature=310)]
