"""Compatibility wrapper for :mod:`synplan.routes.analysis`."""

import synplan.routes.analysis as _analysis

__all__ = [name for name in vars(_analysis) if not name.startswith("_")]

globals().update({name: getattr(_analysis, name) for name in __all__})

del _analysis
