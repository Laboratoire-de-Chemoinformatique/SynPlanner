"""Helpers for backward-compatibility shim modules."""

import warnings


def deprecated_module(old_name: str, new_path: str) -> None:
    warnings.warn(
        f"{old_name} is deprecated; import from {new_path} instead",
        DeprecationWarning,
        stacklevel=2,
    )
