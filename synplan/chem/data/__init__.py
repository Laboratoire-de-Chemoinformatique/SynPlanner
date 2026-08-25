"""Back-compat shim; import from `synplan.chem.reaction.curation` instead."""

from importlib import import_module

from synplan._compat import deprecated_module

deprecated_module(__name__, "synplan.chem.reaction.curation")

_SUBMODULES = (
    "config",
    "filtering",
    "mapping",
    "pipeline",
    "reaction_result",
    "rebalancing",
    "standardizing",
)


def __getattr__(name: str):
    target = "synplan.chem.reaction.curation"
    if name in _SUBMODULES:
        deprecated_module(f"{__name__}.{name}", f"{target}.{name}")
        return import_module(f"{target}.{name}")
    return getattr(import_module(target), name)


def __dir__():
    return sorted(_SUBMODULES)
