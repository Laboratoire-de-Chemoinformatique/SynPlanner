from CGRtools.periodictable import Core, At, DynamicElement
from typing import Optional


class Marked(Core):
    __slots__ = "__mark", "_isotope"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__mark = None
        self._isotope = 0  # Make sure this exists

    @property
    def mark(self):
        return self.__mark

    @mark.setter
    def mark(self, mark):
        self.__mark = mark

    @property
    def isotope(self):
        return getattr(self, "_isotope", 0)  # Always returns int

    @isotope.setter
    def isotope(self, value):
        self._isotope = int(value)

    def __repr__(self):
        return f"{self.symbol}({self.isotope})"

    @property
    def atomic_symbol(self) -> str:
        return self.__class__.__name__[6:]

    @property
    def symbol(self) -> str:
        return "X"  # For human-readable representation

    def __len__(self):
        return super().__len__()


class MarkedAt(Marked, At):
    atomic_number = At.atomic_number

    @property
    def atomic_symbol(self):
        return "At"

    @property
    def symbol(self):
        return "X"

    def __repr__(self):
        return f"X({self.isotope})"

    def __str__(self):
        return f"X({self.isotope})"

    def __hash__(self):
        return hash(
            (
                self.isotope,
                getattr(self, "atomic_number", 0),
                getattr(self, "charge", 0),
                getattr(self, "is_radical", False),
            )
        )


class DynamicX(DynamicElement):
    __slots__ = ("_mark", "_isotope")

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
        if charge == 0 and not is_radical and (valence == 1):
            return tuple()
        elif charge == 0 and not is_radical and valence == 0:
            return tuple()
        else:
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
    def p_hybridization(self) -> Optional[int]:
        return self.hybridization
