"""Recombination: one join keyed by the 26-pair table, plus the two enumeration modes."""

from collections.abc import Iterable, Iterator
from itertools import product as cartesian
from math import inf
from time import perf_counter

from chython import synthon_smiles
from chython.containers import MoleculeContainer, SynthonContainer
from chython.exceptions import InvalidAromaticRing
from chython.periodictable.base.synthon import BIVALENT_LABELS

from synplan.chem.synthon.config import SynthonConfig, load_data
from synplan.chem.synthon.stock import label_keys

Key = tuple[str, bool, str]


def load_pairs(
    config: SynthonConfig | None = None, key: str = "pairs"
) -> dict[Key, set[Key]]:
    """The compatibility matrix, expanded symmetric. Keyed on (symbol, aromatic, token) — NEVER on
    the token alone: C:elec has six partners, S:elec has exactly one.

    `key="ring_pairs"` reads the ring-only rows instead; they are never consulted by `join`.
    """
    rows = load_data((config or SynthonConfig()).rules_path)[key]
    pairs: dict[Key, set[Key]] = {}
    for symbol_a, aromatic_a, token_a, symbol_b, aromatic_b, token_b in rows:
        a, b = (symbol_a, aromatic_a, token_a), (symbol_b, aromatic_b, token_b)
        pairs.setdefault(a, set()).add(b)
        pairs.setdefault(b, set()).add(a)
    return pairs


def _fix_aromatic_marks(molecule: SynthonContainer, new_ring: bool = False) -> None:
    """Rebuilding a molecule atom by atom unsets every aromatic hydrogen mark it touches, so a
    pyrrole nitrogen anywhere in the partner silently loses its hydrogen.

    `add_bond` documents kekule/thiele as the cure. The retry covers a labelled ring nitrogen
    whose hydrogen tautomer standardisation moved elsewhere; a ring that cannot exist raises.

    `join` bonds two disconnected fragments and so can never build a ring: with nothing aromatic
    already there is nothing to redo. `close_ring` can, and a ring that has just become aromatic
    carries no aromatic atom until `thiele` finds one - hence `new_ring`.
    """
    # ponytail: rederives the whole product's aromatic marks per join; scope it to the ring
    # system the new bond touches if join() ever shows up in a profile
    if not new_ring and not any(
        atom.hybridization == 4 for _, atom in molecule.atoms()
    ):
        return
    try:
        molecule.kekule()
    except InvalidAromaticRing:
        molecule.kekule(ignore_pyrrole_hydrogen=True)
    molecule.thiele()


def join(
    a: SynthonContainer, atom_a: int, b: SynthonContainer, atom_b: int
) -> SynthonContainer:
    """Draw the bond the two labels stand for and consume both attachment points.

    The 38 reconstruction SMIRKS collapse to this: every one of them is 2 reactants -> 1 product
    drawing exactly one new bond, with the marker elements only there to find the ends.

    Raises `InvalidAromaticRing` when the two labels ask for a ring that cannot exist.
    """
    label_a, label_b = a.atom(atom_a).label, b.atom(atom_b).label
    order = 2 if label_a in BIVALENT_LABELS and label_b in BIVALENT_LABELS else 1
    merged = a.copy()
    mapping = {}
    for n, atom in b.atoms():
        # add_atom re-assigns charge/is_radical from its kwargs, so they travel as kwargs
        mapping[n] = merged.add_atom(
            atom.copy(full=True), charge=atom.charge, is_radical=atom.is_radical
        )
    for n, m, bond in b.bonds():
        merged.add_bond(mapping[n], mapping[m], int(bond))
    merged.atom(atom_a)._label = None
    merged.atom(mapping[atom_b])._label = None
    merged.add_bond(atom_a, mapping[atom_b], order)
    _fix_aromatic_marks(merged)
    merged.flush_cache()
    return merged


def close_ring(
    molecule: SynthonContainer, atom_a: int, atom_b: int
) -> SynthonContainer:
    """`join`, minus the merge: the two labels are already in the same molecule.

    This is the whole of heterocyclisation. A ring synthon is an ordinary H-capped fragment of the
    product carrying two labels; the first bond is drawn by `join` and merges the partner in, which
    leaves the second one INTRAMOLECULAR and so beyond anything `join` can express.
    """
    label_a, label_b = molecule.atom(atom_a).label, molecule.atom(atom_b).label
    order = 2 if label_a in BIVALENT_LABELS and label_b in BIVALENT_LABELS else 1
    closed = molecule.copy()
    closed.atom(atom_a)._label = None
    closed.atom(atom_b)._label = None
    closed.add_bond(atom_a, atom_b, order)
    _fix_aromatic_marks(closed, new_ring=True)
    closed.flush_cache()
    return closed


def ring_size(
    molecule: SynthonContainer, start: int, end: int, limit: int
) -> int | None:
    """Size of the ring the bond start-end would close, or None beyond `limit` atoms."""
    seen, frontier = {start}, [start]
    bonds = molecule._bonds
    for distance in range(1, limit):
        nxt = []
        for n in frontier:
            for m in bonds[n]:
                if m == end:
                    return distance + 1
                if m not in seen:
                    seen.add(m)
                    nxt.append(m)
        frontier = nxt
    return None


def open_points(synthon: SynthonContainer) -> list[tuple[int, Key]]:
    return [
        (n, (a.atomic_symbol, a.hybridization == 4, a.label))
        for n, a in synthon.atoms()
        if getattr(a, "_label", None) is not None
    ]


class Enumerator:
    """Streams products. Upstream accumulates a list and rebuilds `list(set(...))` inside the loop,
    then caps at a hard-coded million because it cannot prioritise."""

    def __init__(
        self,
        config: SynthonConfig | None = None,
        pairs: dict[Key, set[Key]] | None = None,
    ) -> None:
        self.config = config or SynthonConfig()
        self.pairs = pairs if pairs is not None else load_pairs(self.config)
        # a separate table: aliphatic C:nuc + N:elec is the C5-N1 bond of every 1,2,3-triazole, but
        # in `pairs` it would claim an alkyl nucleophile aminates, so it is legal ONLY in a ring
        self.ring_pairs = load_pairs(self.config, "ring_pairs")

    def compatible(self, a: Key, b: Key) -> bool:
        return b in self.pairs.get(a, ())

    def closable(self, a: Key, b: Key) -> bool:
        """A ring closes on anything `pairs` allows, plus the ring-only rows."""
        return self.compatible(a, b) or b in self.ring_pairs.get(a, ())

    def _fuse(self, molecule: SynthonContainer) -> Iterator[SynthonContainer]:
        """Every legal ring closure between two open points of ONE molecule.

        Terminates: each fuse consumes two labels. `ring_closure_sizes=()` disables it entirely.
        """
        sizes = self.config.ring_closure_sizes
        if not sizes:
            return
        points = open_points(molecule)
        limit = max(sizes)
        for index, (atom_a, key_a) in enumerate(points):
            for atom_b, key_b in points[index + 1 :]:
                if not self.closable(key_a, key_b):
                    continue
                if ring_size(molecule, atom_a, atom_b, limit) not in sizes:
                    continue
                try:
                    yield close_ring(molecule, atom_a, atom_b)
                except InvalidAromaticRing:
                    continue  # the two labels ask for a ring that cannot exist

    def _index(self, synthons: Iterable[str]) -> dict[Key, list[str]]:
        index: dict[Key, list[str]] = {}
        for smi in synthons:
            for key in set(label_keys(synthon_smiles(smi))):
                index.setdefault(key, []).append(smi)
        return index

    def enumerate_library(self, synthons: Iterable[str]) -> Iterator[MoleculeContainer]:
        """Unconstrained forward synthesis: grow a molecule until it has no open labels left."""
        pool = list(synthons)
        index = self._index(pool)
        deadline = self._deadline()
        emitted = 0
        seen: set[str] = set()
        for seed in pool:
            if perf_counter() > deadline:
                return
            for molecule in self._grow(
                synthon_smiles(seed), index, frozenset({seed}), deadline
            ):
                key = str(molecule)
                if key in seen:
                    continue
                seen.add(key)
                if (
                    not self.config.mw_lower
                    <= molecule.molecular_mass
                    <= self.config.mw_upper
                ):
                    continue
                yield molecule
                emitted += 1
                if emitted >= self.config.max_products:
                    return

    def _deadline(self) -> float:
        return perf_counter() + (self.config.time_budget_s or inf)

    def _grow(
        self,
        molecule: SynthonContainer,
        index: dict[Key, list[str]],
        used: frozenset[str],
        deadline: float,
    ) -> Iterator[SynthonContainer]:
        if perf_counter() > deadline:
            return
        for fused in self._fuse(molecule):
            yield from self._grow(fused, index, used, deadline)
        points = open_points(molecule)
        if not points:
            yield molecule
            return
        if len(used) >= self.config.max_reacted_synthons:
            return  # a product that hits the cap with open labels is discarded
        atom, key = points[0]
        # sorted: partner order decides which products a capped run keeps, and set order is
        # PYTHONHASHSEED-randomised
        for partner_key in sorted(self.pairs.get(key, ())):
            for candidate in index.get(partner_key, ()):
                if candidate in used:  # each branch gets its OWN used set
                    continue
                partner = synthon_smiles(candidate)
                for partner_atom, other in open_points(partner):
                    if other != partner_key:
                        continue
                    try:
                        grown = join(molecule, atom, partner, partner_atom)
                    except InvalidAromaticRing:
                        continue  # the two labels ask for a ring that cannot exist
                    yield from self._grow(grown, index, used | {candidate}, deadline)

    def enumerate_analogues(
        self, pathway_synthons: Iterable[str], slots: dict[str, list[str]]
    ) -> Iterator[MoleculeContainer]:
        """Every slot of a fragmentation pathway used exactly once, each from its own candidates.

        `strict_availability` is an all-or-nothing veto: an empty slot kills the pathway.
        """
        order = list(pathway_synthons)
        candidates = [
            slots.get(s, [])
            if self.config.strict_availability
            else (slots.get(s) or [s])
            for s in order
        ]
        if any(not c for c in candidates):
            return
        deadline = self._deadline()
        emitted = 0
        seen: set[str] = set()
        for combination in cartesian(*candidates):
            if perf_counter() > deadline:
                return
            for molecule in self._assemble(
                [synthon_smiles(s) for s in combination], deadline
            ):
                # _close reaches the same product once per ordering of the slots
                key = str(molecule)
                if key in seen:
                    continue
                seen.add(key)
                yield molecule
                emitted += 1
                if emitted >= self.config.max_products:
                    return

    def _assemble(
        self, synthons: list[SynthonContainer], deadline: float
    ) -> Iterator[MoleculeContainer]:
        if not synthons:
            return
        yield from self._close(synthons[0], synthons[1:], deadline)

    def _close(
        self,
        molecule: SynthonContainer,
        remaining: list[SynthonContainer],
        deadline: float,
    ) -> Iterator[MoleculeContainer]:
        if perf_counter() > deadline:
            return
        for fused in self._fuse(molecule):
            yield from self._close(fused, remaining, deadline)
        if not remaining:
            if not open_points(molecule):
                yield molecule
            return
        points = open_points(molecule)
        for index, partner in enumerate(remaining):
            rest = remaining[:index] + remaining[index + 1 :]
            for atom, key in points:
                for partner_atom, other in open_points(partner):
                    if not self.compatible(key, other):
                        continue
                    try:
                        closed = join(molecule, atom, partner, partner_atom)
                    except InvalidAromaticRing:
                        continue  # the two labels ask for a ring that cannot exist
                    yield from self._close(closed, rest, deadline)


__all__ = [
    "Enumerator",
    "Key",
    "close_ring",
    "join",
    "load_pairs",
    "open_points",
    "ring_size",
]
