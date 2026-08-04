"""Deprecated launcher for protection source-data conversion."""

from __future__ import annotations

import warnings

from scripts.convert_protection_source_data import (
    main,
)

warnings.warn(
    "Run synplan-convert-protection-data or scripts/convert_protection_source_data.py instead",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
