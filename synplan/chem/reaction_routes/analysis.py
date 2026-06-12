"""Compatibility wrapper for :mod:`synplan.chem.reaction.routes.analysis`."""

import synplan.chem.reaction.routes.analysis as _analysis

__all__ = [name for name in vars(_analysis) if not name.startswith("_")]

globals().update({name: getattr(_analysis, name) for name in __all__})

del _analysis
