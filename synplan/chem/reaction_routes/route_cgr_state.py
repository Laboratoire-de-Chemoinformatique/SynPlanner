"""Compatibility wrapper for :mod:`synplan.routes.route_cgr.state`."""

from synplan.routes.route_cgr import state as _state
from synplan.routes.route_cgr.state import *


def __getattr__(name):
    return getattr(_state, name)
