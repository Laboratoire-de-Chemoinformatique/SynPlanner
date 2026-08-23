"""Back-compat shim; import from `synplan.chem.synthon` instead."""

from importlib import import_module

from synplan._compat import deprecated_module
from synplan.chem.synthon import __all__ as __all__

deprecated_module(__name__, "synplan.chem.synthon")

# Resolved lazily: the eager form would pull the whole package on import and
# undo `chem.synthon`'s no-torch guarantee.
_MOVED = {
    "audit": "synplan.interfaces.synthon_audit",
    "authoring": "synplan.chem.synthon.rules.validate",
    "cli": "synplan.interfaces.synthon_commands",
    "data": "synplan.chem.synthon.rules",
    "enumeration": "synplan.chem.synthon.enumerate",
    "priority": "synplan.chem.reaction.rules.synthon",
    "reactor": "synplan.chem.synthon.transformer",
}


def __getattr__(name: str):
    if name in _MOVED:
        deprecated_module(f"{__name__}.{name}", _MOVED[name])
        return import_module(_MOVED[name])
    return getattr(import_module("synplan.chem.synthon"), name)


def __dir__():
    return sorted({*__all__, *_MOVED})
