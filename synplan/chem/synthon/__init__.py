"""Synt-On port: building-block classification, synthonisation, fragmentation and enumeration.

Lazy exports — importing this package must not pull torch, matplotlib or synplan.mcts.
"""

from importlib import import_module

__all__ = [
    "BBClassifier",
    "BBSynthoniser",
    "Enumerator",
    "Fragmenter",
    "SynthonConfig",
    "SynthonTransformer",
    "find_analogues",
    "load_synthon_stock",
    "murcko_scaffold",
]

_EXPORTS = {
    "BBClassifier": "classify",
    "BBSynthoniser": "synthonise",
    "Enumerator": "enumeration",
    "Fragmenter": "fragment",
    "SynthonConfig": "config",
    "SynthonTransformer": "reactor",
    "find_analogues": "analogues",
    "load_synthon_stock": "stock",
    "murcko_scaffold": "scaffolds",
}


def __getattr__(name: str):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__():
    return sorted(__all__)
