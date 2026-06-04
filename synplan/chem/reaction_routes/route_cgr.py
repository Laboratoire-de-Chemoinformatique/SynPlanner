"""Compatibility wrapper for :mod:`synplan.routes.route_cgr`."""

from typing import TYPE_CHECKING

from chython.containers import CGRContainer, MoleculeContainer, ReactionContainer
from chython.containers.bonds import DynamicBond

from synplan.routes.route_cgr import *
from synplan.routes.route_cgr import __all__ as _route_cgr_all
from synplan.routes.route_cgr.builder import _bond_key

if TYPE_CHECKING:
    from synplan.mcts.tree import Tree
else:
    # Declared for static export checks; resolved lazily by ``__getattr__``.
    Tree: object

__all__ = [
    *_route_cgr_all,
    "CGRContainer",
    "DynamicBond",
    "MoleculeContainer",
    "ReactionContainer",
    "Tree",
    "_bond_key",
]


def __getattr__(name):
    if name == "Tree":
        from synplan.mcts.tree import Tree as _Tree

        globals()["Tree"] = _Tree
        return _Tree
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
