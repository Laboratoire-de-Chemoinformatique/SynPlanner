"""Strict configuration for ordinary building-block stocks and preparation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from synplan.utils.config import BaseConfigModel

BuildingBlockInputFormat = Literal[
    "auto", "smi", "smiles", "cxsmiles", "sdf", "csv", "tsv"
]
BuildingBlockStockInputFormat = Literal["auto", "smiles", "inchikey"]
DeprotectionPolicy = Literal["conservative", "aggressive"]
DeprotectionOutput = Literal["replace", "append"]
AuditOverwrite = Literal["error", "replace"]


class BuildingBlockStockLoadConfig(BaseConfigModel):
    """Configuration for decoding an ordinary building-block stock source.

    This configuration belongs to the stock-loading boundary, not to tree-search
    behavior. identity_format describes the canonical SMILES or full Standard
    InChIKeys stored in the input. Setting standardize to false trusts an explicitly
    declared plain SMILES stock as already canonical and skips molecular parsing.
    """

    identity_format: BuildingBlockStockInputFormat = "auto"
    standardize: bool = True
    chemistry_column: str | None = None
    delimiter: str | None = None

    @model_validator(mode="after")
    def _validate_source_options(self) -> BuildingBlockStockLoadConfig:
        if not self.standardize and self.identity_format != "smiles":
            raise ValueError(
                "standardize=False requires identity_format='smiles'"
            )
        if self.chemistry_column is not None and not self.chemistry_column.strip():
            raise ValueError("chemistry_column must not be empty")
        if self.delimiter is not None and len(self.delimiter) != 1:
            raise ValueError("delimiter must be exactly one character")
        return self


class BuildingBlockPreparationConfig(BaseConfigModel):
    """Configuration for preparing an ordinary planner building-block stock.

    The model deliberately rejects unknown fields.  Optional output paths are only
    valid when their owning feature is enabled, preventing configurations which look
    effective but are silently ignored.
    """

    input_format: BuildingBlockInputFormat = "auto"
    smiles_column: str = "SMILES"

    deprotect: bool = False
    deprotect_policy: DeprotectionPolicy = "conservative"
    deprotect_output: DeprotectionOutput = "replace"

    write_inchikey_stock: bool = False
    protected_output_file: str | None = None
    inchikey_file: str | None = None
    identity_reference_file: str | None = None
    price_reference_file: str | None = None
    duplicates_file: str | None = None
    collisions_file: str | None = None
    stereo_file: str | None = None

    write_audit_files: bool = False
    audit_overwrite: AuditOverwrite = "error"
    num_workers: int | None = Field(None, ge=1)
    batch_size: int = Field(500, ge=1)

    @model_validator(mode="after")
    def _validate_dependencies(self) -> BuildingBlockPreparationConfig:
        if not self.smiles_column.strip():
            raise ValueError("smiles_column must not be empty")
        if not self.deprotect:
            if self.deprotect_policy != "conservative":
                raise ValueError(
                    "deprotect_policy can only be changed when deprotect is true"
                )
            if self.deprotect_output != "replace":
                raise ValueError(
                    "deprotect_output can only be changed when deprotect is true"
                )
            if self.protected_output_file is not None:
                raise ValueError(
                    "protected_output_file can only be used when deprotect is true"
                )
        if not self.write_inchikey_stock:
            if self.inchikey_file is not None:
                raise ValueError(
                    "inchikey_file requires write_inchikey_stock to be true"
                )
            if self.identity_reference_file is not None:
                raise ValueError(
                    "identity_reference_file requires write_inchikey_stock to be true"
                )
            if self.price_reference_file is not None:
                raise ValueError(
                    "price_reference_file requires write_inchikey_stock to be true"
                )
            if self.collisions_file is not None:
                raise ValueError(
                    "collisions_file requires write_inchikey_stock to be true"
                )
        if not self.write_audit_files and self.audit_overwrite != "error":
            raise ValueError(
                "audit_overwrite can only be changed when write_audit_files is true"
            )
        return self


__all__ = [
    "AuditOverwrite",
    "BuildingBlockInputFormat",
    "BuildingBlockPreparationConfig",
    "BuildingBlockStockInputFormat",
    "BuildingBlockStockLoadConfig",
    "DeprotectionOutput",
    "DeprotectionPolicy",
]
