from chython.periodictable import At, DynamicElement


class Marked:
    """Mixin that adds a mark property and overrides isotope.

    Must be used together with an Element-based class (e.g. At) via
    multiple inheritance so the real atom behavior comes from Element.
    Uses __slots__ = () to avoid layout conflict with Element's slots.
    Concrete subclasses (MarkedAt) define the actual storage slot.
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mark = None
        self._isotope = 0

    @property
    def mark(self):
        return self._mark

    @mark.setter
    def mark(self, mark):
        self._mark = mark

    @property
    def isotope(self):
        return getattr(self, "_isotope", 0)

    @isotope.setter
    def isotope(self, value):
        self._isotope = int(value) if value is not None else 0

    def __repr__(self):
        return f"{self.symbol}({self.isotope})"

    @property
    def atomic_symbol(self) -> str:
        return self.__class__.__name__[6:]

    @property
    def symbol(self) -> str:
        return "X"

    def __len__(self):
        return super().__len__()


class MarkedAt(Marked, At):
    __slots__ = ("_mark",)
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
    def p_hybridization(self) -> int | None:
        return self.hybridization
