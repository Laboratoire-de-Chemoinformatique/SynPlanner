"""Compatibility wrapper for :mod:`synplan.chem.reaction.routes.representation.state`."""

from synplan.chem.reaction.routes.representation import state as _state
from synplan.chem.reaction.routes.representation.state import *


def __getattr__(name):
    return getattr(_state, name)
