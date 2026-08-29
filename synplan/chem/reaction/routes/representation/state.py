"""RouteCGR atom and bond state helpers."""

from __future__ import annotations

from chython.containers.bonds import DynamicBond
from chython.periodictable import DynamicElement

__all__ = [
    "RouteDynamicBond",
    "bond_key",
    "remove_transient_bonds",
    "route_atom",
    "transient_bond",
]


def bond_key(atom1: int, atom2: int) -> tuple[int, int]:
    """Return the canonical key for an undirected atom pair."""

    return (atom1, atom2) if atom1 <= atom2 else (atom2, atom1)


def set_symmetric_bond(cgr, atom1, atom2, bond):
    """Store one bond object in both directions of a CGR adjacency map."""

    cgr._bonds.setdefault(atom1, {})[atom2] = bond
    cgr._bonds.setdefault(atom2, {})[atom1] = bond


class RouteDynamicBond(DynamicBond):
    """DynamicBond carrying RouteCGR route-order and deconvolution metadata."""

    __slots__ = ("route_bond_step_states", "route_order", "route_step_order")

    def __init__(
        self,
        order=None,
        p_order=None,
        route_order=None,
        route_step_order=None,
    ):
        if order is None and p_order is None:
            self._order = self._p_order = None
        else:
            super().__init__(order, p_order)
        self.route_order = route_order
        self.route_step_order = metadata_set(route_step_order)
        self.route_bond_step_states = {}

    @classmethod
    def from_bond(cls, bond: DynamicBond, route_order=None, route_step_order=None):
        copy = object.__new__(cls)
        copy._order = bond.order
        copy._p_order = bond.p_order
        copy.route_order = (
            getattr(bond, "route_order", None) if route_order is None else route_order
        )
        copy.route_step_order = metadata_set(
            getattr(bond, "route_step_order", None)
            if route_step_order is None
            else route_step_order
        )
        copy.route_bond_step_states = dict(getattr(bond, "route_bond_step_states", {}))
        return copy

    def copy(self, *args, **kwargs):
        return self.from_bond(self, self.route_order, self.route_step_order)


_ROUTE_ATOM_CLASSES = {}


def metadata_set(value):
    if value is None:
        return set()
    if isinstance(value, (set, frozenset, list, tuple)):
        return set(value)
    return {value}


def _route_atom_class_from_class(atom_class, symbol):
    if atom_class in _ROUTE_ATOM_CLASSES:
        return _ROUTE_ATOM_CLASSES[atom_class]

    class_name = f"Route{atom_class.__name__}"

    def atomic_symbol(self):
        return symbol

    def copy(self, *args, **kwargs):
        duplicate = atom_class.copy(self)
        duplicate.route_order = metadata_set(getattr(self, "route_order", None))
        duplicate.route_step_order = metadata_set(
            getattr(self, "route_step_order", None)
        )
        duplicate.route_atom_step_states = dict(
            getattr(self, "route_atom_step_states", {})
        )
        return duplicate

    route_atom_class = type(
        class_name,
        (atom_class,),
        {
            "__module__": __name__,
            "__slots__": (
                "route_order",
                "route_step_order",
                "route_atom_step_states",
            ),
            "atomic_symbol": property(atomic_symbol),
            "copy": copy,
        },
    )
    globals()[class_name] = route_atom_class
    _ROUTE_ATOM_CLASSES[atom_class] = route_atom_class
    return route_atom_class


def __getattr__(name):
    if name.startswith("RouteDynamic"):
        atom_class_name = name[5:]
        for atom_class in DynamicElement.__subclasses__():
            if atom_class.__name__ == atom_class_name:
                return _route_atom_class_from_class(atom_class, atom_class_name[7:])
    raise AttributeError(name)


def _route_atom_class(atom):
    if hasattr(atom, "route_order") and hasattr(atom, "route_step_order"):
        return atom.__class__
    return _route_atom_class_from_class(atom.__class__, atom.atomic_symbol)


def route_atom(atom, route_orders, route_step_orders=None):
    """Return an atom copy carrying route-order and step-order metadata."""

    route_orders = metadata_set(route_orders)
    route_step_orders = metadata_set(route_step_orders)

    if hasattr(atom, "route_order") and hasattr(atom, "route_step_order"):
        atom.route_order.update(route_orders)
        atom.route_step_order.update(route_step_orders)
        if not hasattr(atom, "route_atom_step_states"):
            atom.route_atom_step_states = {}
        return atom

    route_atom_class = _route_atom_class(atom)
    new_atom = object.__new__(route_atom_class)
    new_atom._isotope = atom.isotope
    new_atom._charge = atom.charge
    new_atom._is_radical = atom.is_radical
    new_atom._p_is_radical = atom.p_is_radical
    new_atom._p_charge = atom.p_charge
    new_atom._xy = atom._xy.__class__(atom._xy.x, atom._xy.y)
    new_atom.route_order = metadata_set(getattr(atom, "route_order", None))
    new_atom.route_order.update(route_orders)
    new_atom.route_step_order = metadata_set(getattr(atom, "route_step_order", None))
    new_atom.route_step_order.update(route_step_orders)
    new_atom.route_atom_step_states = dict(getattr(atom, "route_atom_step_states", {}))
    return new_atom


def transient_bond() -> RouteDynamicBond:
    """The route-only marker for a bond that forms and later breaks.

    chython's ``DynamicBond`` rejects ``(None, None)``; ``RouteDynamicBond``
    accepts it, which is the whole reason the subclass exists.
    """

    return RouteDynamicBond()


def remove_transient_bonds(cgr):
    """Delete transient ``DynamicBond(None, None)`` markers from a CGR."""

    for atom1, atom2, bond in list(cgr.bonds()):
        if bond.order is None and bond.p_order is None:
            cgr.delete_bond(atom1, atom2)
    return cgr
