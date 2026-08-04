from chython.periodictable import At, DynamicElement

__all__ = ["DynamicX", "Marked", "MarkedAt", "MarkedY"]


class Marked(At):
    """Chython-based atom that adds route marks and pseudo-element symbols.

    Chython still owns isotope validation, charge/radical state, valence rules,
    coordinates, and copying; this class only adds SynPlanner's route mark,
    display symbol, and standalone hash compatible with previous marker use.
    """

    marker_symbol = "X"
    isotopes_distribution = range(10_000)
    __slots__ = ()

    def __init__(self, isotope: int | None = None, *args, mark=None, **kwargs):
        super().__init__(0 if isotope is None else isotope, *args, **kwargs)
        self._mark = None
        if mark is not None:
            self.mark = mark

    @property
    def mark(self):
        return self._mark

    @mark.setter
    def mark(self, mark):
        self._mark = mark

    @property
    def atomic_symbol(self) -> str:
        return self.marker_symbol

    @property
    def symbol(self) -> str:
        return self.marker_symbol

    def __repr__(self):
        return f"{self.symbol}({self.isotope or 0})"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if not isinstance(other, Marked):
            return NotImplemented
        return (
            self.marker_symbol,
            self.mark,
            self.isotope,
            getattr(self, "atomic_number", 0),
            getattr(self, "charge", 0),
            getattr(self, "is_radical", False),
        ) == (
            other.marker_symbol,
            other.mark,
            other.isotope,
            getattr(other, "atomic_number", 0),
            getattr(other, "charge", 0),
            getattr(other, "is_radical", False),
        )

    def __hash__(self):
        return hash(
            (
                self.marker_symbol,
                self.mark,
                self.isotope,
                getattr(self, "atomic_number", 0),
                getattr(self, "charge", 0),
                getattr(self, "is_radical", False),
            )
        )

    def __len__(self):
        return super().__len__()


class MarkedAt(Marked):
    __slots__ = ("_mark",)


class MarkedY(Marked):
    """Display-only marker for route-supporting pseudo-reactants."""

    marker_symbol = "Y"
    __slots__ = ("_mark",)


class DynamicX(DynamicElement):
    __slots__ = ("_isotope", "_mark")

    atomic_number = 85
    mass = 0.0
    group = 0
    period = 0
    isotopes_distribution = list(range(20))
    atomic_radius = 0.5
    isotopes_masses = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._isotope = None
        self._mark = None

    @property
    def mark(self):
        return getattr(self, "_mark", None)

    @mark.setter
    def mark(self, value):
        self._mark = value

    @property
    def isotope(self):
        return getattr(self, "_isotope", None)

    @isotope.setter
    def isotope(self, value):
        self._isotope = value

    @property
    def symbol(self) -> str:
        return "X"

    def valence_rules(
        self, charge: int = 0, is_radical: bool = False, valence: int = 0
    ) -> tuple:
        return tuple()

    def __repr__(self):
        return f"Dynamic{self.symbol}()"

    @property
    def p_charge(self) -> int:
        return self.charge

    @property
    def p_is_radical(self) -> bool:
        return self.is_radical

    @property
    def p_hybridization(self) -> int | None:
        return self.hybridization
