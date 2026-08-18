"""Combinatorial library enumeration. Enumeration is the capability, `synthon` is one method.

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
    "BBClassifier": "synthon",
    "BBSynthoniser": "synthon",
    "Enumerator": "synthon",
    "Fragmenter": "synthon",
    "SynthonConfig": "synthon",
    "SynthonTransformer": "synthon",
    "classify_coverage": "synthon",
    "find_analogues": "synthon",
    "load_coverage_rules": "synthon",
    "load_synthon_stock": "synthon",
}


def __getattr__(name: str):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__():
    return sorted(__all__)
