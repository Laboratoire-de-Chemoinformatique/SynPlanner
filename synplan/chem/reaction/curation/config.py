"""Shared configuration models for reaction-data processing."""

from pydantic import Field

from synplan.utils.config import BaseConfigModel


class SmallMoleculesConfig(BaseConfigModel):
    """Configure the heavy-atom threshold for small-molecule handling."""

    mol_max_size: int = Field(default=6, ge=1)


__all__ = ["SmallMoleculesConfig"]
