"""Configuration contracts for building-block preparation."""

import pytest
from pydantic import ValidationError

from synplan.chem.building_blocks.config import (
    BuildingBlockPreparationConfig,
    BuildingBlockStockLoadConfig,
)
from synplan.utils.config import TreeConfig


def test_stock_load_config_defaults_to_auto_detection() -> None:
    config = BuildingBlockStockLoadConfig()
    assert config.identity_format == "auto"
    assert config.standardize is True
    assert config.chemistry_column is None
    assert config.delimiter is None


def test_stock_load_config_yaml_round_trip(tmp_path) -> None:
    expected = BuildingBlockStockLoadConfig(
        identity_format="inchikey",
        chemistry_column="Identity",
        delimiter=";",
    )
    path = tmp_path / "stock.yaml"
    expected.to_yaml(str(path))
    assert BuildingBlockStockLoadConfig.from_yaml(str(path)) == expected


def test_nonstandardizing_stock_requires_explicit_smiles_identity() -> None:
    config = BuildingBlockStockLoadConfig(
        identity_format="smiles",
        standardize=False,
    )

    assert config.standardize is False
    with pytest.raises(ValidationError, match="requires identity_format='smiles'"):
        BuildingBlockStockLoadConfig(standardize=False)


@pytest.mark.parametrize(
    "values",
    [{"delimiter": "::"}, {"chemistry_column": " "}, {"identity_format": "inchi"}],
)
def test_invalid_stock_load_options_fail(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BuildingBlockStockLoadConfig.model_validate(values)


def test_tree_config_does_not_own_stock_source_format() -> None:
    assert "building_blocks_format" not in TreeConfig.model_fields


def test_defaults_are_backward_compatible() -> None:
    config = BuildingBlockPreparationConfig()
    assert not config.deprotect
    assert config.deprotect_policy == "conservative"
    assert config.deprotect_output == "replace"
    assert not config.write_inchikey_stock
    assert not config.write_audit_files
    assert config.num_workers is None
    assert config.batch_size == 500


@pytest.mark.parametrize(
    "values",
    [
        {"protected_output_file": "protected.smi"},
        {"deprotect_policy": "aggressive"},
        {"deprotect_output": "append"},
        {"inchikey_file": "stock.inchikey"},
        {"identity_reference_file": "identity.tsv"},
        {"price_reference_file": "prices.tsv"},
        {"collisions_file": "collisions.tsv"},
        {"audit_overwrite": "replace"},
        {"unknown_option": True},
    ],
)
def test_invalid_dependencies_and_unknown_fields_fail(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        BuildingBlockPreparationConfig.model_validate(values)


def test_duplicate_report_can_be_requested_without_deprotection() -> None:
    config = BuildingBlockPreparationConfig(duplicates_file="duplicates.tsv")
    assert config.duplicates_file == "duplicates.tsv"


def test_yaml_round_trip(tmp_path) -> None:
    expected = BuildingBlockPreparationConfig(
        deprotect=True,
        deprotect_policy="aggressive",
        deprotect_output="append",
        write_inchikey_stock=True,
        write_audit_files=True,
        audit_overwrite="replace",
        num_workers=2,
        batch_size=9,
    )
    path = tmp_path / "config.yaml"
    expected.to_yaml(str(path))
    assert BuildingBlockPreparationConfig.from_yaml(str(path)) == expected
