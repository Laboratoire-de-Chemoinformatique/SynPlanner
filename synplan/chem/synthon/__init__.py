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
    "classify_coverage",
    "find_analogues",
    "load_coverage_rules",
    "load_synthon_stock",
]

_EXPORTS = {
    "BBClassifier": "classify",
    "BBSynthoniser": "synthonise",
    "Enumerator": "enumerate",
    "Fragmenter": "fragment",
    "SynthonConfig": "config",
    "SynthonTransformer": "transformer",
    "classify_coverage": "coverage",
    "find_analogues": "analogues",
    "load_coverage_rules": "coverage",
    "load_synthon_stock": "stock",
}


def __getattr__(name: str):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__():
    return sorted(__all__)
