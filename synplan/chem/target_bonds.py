"""Target-bond constraints and persistent target-atom provenance."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from operator import index

from chython.containers import MoleculeContainer

BondKey = tuple[int, int]

_ALLOWED_BOND_STATES = frozenset({0, 1, 2})


def bond_key(atom1: int, atom2: int) -> BondKey:
    """Return a direction-independent atom-pair key."""

    return tuple(sorted((atom1, atom2)))


def molecule_bond_keys(molecules: Iterable[MoleculeContainer]) -> set[BondKey]:
    """Collect local atom-number bond keys from one or more molecules."""

    return {
        bond_key(atom1, atom2)
        for molecule in molecules
        for atom1, atom2, _bond in molecule.bonds()
    }


@dataclass(frozen=True, slots=True)
class TargetAtomProvenance:
    """Immutable mapping from local atom numbers to original target atoms.

    Chython may reuse atom numbers after a target has been split into multiple
    precursors. Keeping target identity separately prevents a newly introduced
    atom from inheriting the identity of an atom in another fragment merely
    because both use the same local integer.
    """

    pairs: frozenset[tuple[int, int]] = frozenset()

    def __post_init__(self) -> None:
        local_atoms = [local for local, _target in self.pairs]
        target_atoms = [target for _local, target in self.pairs]
        if len(local_atoms) != len(set(local_atoms)):
            raise ValueError("local atom numbers must have unique target provenance")
        if len(target_atoms) != len(set(target_atoms)):
            raise ValueError("target atom identities must be unique within a precursor")

    @classmethod
    def from_mapping(cls, provenance: Mapping[int, int] | None) -> TargetAtomProvenance:
        """Create an immutable provenance value from a mapping."""

        if provenance is None:
            return cls()
        return cls(
            frozenset((int(local), int(target)) for local, target in provenance.items())
        )

    @classmethod
    def for_target(cls, target: MoleculeContainer) -> TargetAtomProvenance:
        """Seed identity provenance for a canonicalized target molecule."""

        return cls(
            frozenset(
                (atom_number, atom_number) for atom_number in target.atoms_numbers
            )
        )

    def as_dict(self) -> dict[int, int]:
        """Return a mutable copy for efficient local lookup."""

        return dict(self.pairs)

    def inherit(self, molecule: MoleculeContainer) -> TargetAtomProvenance:
        """Retain provenance only for atoms inherited from this precursor."""

        parent = self.as_dict()
        return TargetAtomProvenance(
            frozenset(
                (atom_number, parent[atom_number])
                for atom_number in molecule.atoms_numbers
                if atom_number in parent
            )
        )


@dataclass(frozen=True, slots=True)
class TargetBondConstraints:
    """Normalized immutable target-bond constraint specification."""

    neutral: frozenset[BondKey] = frozenset()
    required: frozenset[BondKey] = frozenset()
    frozen: frozenset[BondKey] = frozenset()

    def __post_init__(self) -> None:
        if self.neutral & self.required or self.neutral & self.frozen:
            raise ValueError("target-bond states must be mutually exclusive")
        if self.required & self.frozen:
            raise ValueError("a target bond cannot be both required and frozen")

    @property
    def active(self) -> bool:
        """Return whether any non-zero constraint is present."""

        return bool(self.required or self.frozen)

    @classmethod
    def from_state(
        cls,
        target: MoleculeContainer,
        bonds_state: Mapping[BondKey, int] | None,
    ) -> TargetBondConstraints:
        """Validate and normalize the public ``Tree(bonds_state=...)`` mapping."""

        if bonds_state is None:
            return cls()
        if not isinstance(bonds_state, Mapping):
            raise ValueError("bonds_state must be a mapping of atom pairs to states")

        normalized: dict[BondKey, int] = {}
        for bond, state in bonds_state.items():
            if not isinstance(bond, tuple) or len(bond) != 2:
                raise ValueError(
                    "each bonds_state key must be a two-item tuple of atom numbers"
                )
            if isinstance(state, bool) or any(isinstance(atom, bool) for atom in bond):
                raise ValueError("bond atom numbers and states must be integers")
            try:
                atom1, atom2 = (index(atom) for atom in bond)
                state_value = index(state)
            except TypeError as error:
                raise ValueError(
                    "bond atom numbers and states must be integers"
                ) from error

            if atom1 == atom2:
                raise ValueError("a bond must connect two different atoms")
            if state_value not in _ALLOWED_BOND_STATES:
                raise ValueError("bond states must be 0, 1, or 2")

            key = bond_key(atom1, atom2)
            previous = normalized.get(key)
            if previous is not None and previous != state_value:
                raise ValueError(f"conflicting states supplied for target bond {key}")
            normalized[key] = state_value

        target_bonds = molecule_bond_keys((target,))
        missing = sorted(bond for bond in normalized if bond not in target_bonds)
        if missing:
            raise ValueError(f"selected bonds are not present in the target: {missing}")

        return cls(
            neutral=frozenset(bond for bond, state in normalized.items() if state == 0),
            required=frozenset(
                bond for bond, state in normalized.items() if state == 1
            ),
            frozen=frozenset(bond for bond, state in normalized.items() if state == 2),
        )

    def as_dict(self) -> dict[BondKey, int]:
        """Return a defensive normalized mapping snapshot."""

        return {
            **{bond: 0 for bond in sorted(self.neutral)},
            **{bond: 1 for bond in sorted(self.required)},
            **{bond: 2 for bond in sorted(self.frozen)},
        }


ProvenancedMolecule = tuple[MoleculeContainer, TargetAtomProvenance]


def target_bond_keys(states: Iterable[ProvenancedMolecule]) -> set[BondKey]:
    """Translate local molecule bonds into stable target-atom bond keys."""

    result: set[BondKey] = set()
    for molecule, provenance in states:
        local_to_target = provenance.as_dict()
        for atom1, atom2, _bond in molecule.bonds():
            target1 = local_to_target.get(atom1)
            target2 = local_to_target.get(atom2)
            if target1 is not None and target2 is not None:
                result.add(bond_key(target1, target2))
    return result


def removed_target_bonds(
    parent: ProvenancedMolecule,
    products: Iterable[ProvenancedMolecule],
) -> set[BondKey]:
    """Return target-derived adjacencies removed by one reaction application."""

    return target_bond_keys((parent,)) - target_bond_keys(products)


def selected_bonds_svg(
    target: MoleculeContainer,
    bonds_state: Mapping[BondKey, int] | None,
    width: str = "900px",
    height: str = "650px",
) -> str:
    """Depict required target bonds in red and frozen target bonds in blue.

    The supplied mapping is normalized and validated with the same rules used by
    :class:`~synplan.mcts.tree.Tree`. Neutral bonds are not highlighted.

    :param target: Target molecule with 2D coordinates.
    :param bonds_state: Target-bond state mapping accepted by ``Tree``.
    :param width: SVG canvas width.
    :param height: SVG canvas height.
    :return: Target depiction with non-zero bond constraints overlaid.
    """

    constraints = TargetBondConstraints.from_state(target, bonds_state)
    highlighted = {
        **{bond: 1 for bond in constraints.required},
        **{bond: 2 for bond in constraints.frozen},
    }
    base_svg = target.depict(format="svg", width=width, height=height)
    atoms = dict(target.atoms())
    lines = ['<g fill="none" stroke-linecap="round" pointer-events="none">']

    for (atom1, atom2), state in sorted(highlighted.items()):
        color = "red" if state == 1 else "blue"
        x1, y1 = float(atoms[atom1].x), -float(atoms[atom1].y)
        x2, y2 = float(atoms[atom2].x), -float(atoms[atom2].y)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        shorten = min(0.18, length * 0.25) if length else 0
        ux, uy = (dx / length, dy / length) if length else (0, 0)
        lines.append(
            f'<line x1="{x1 + ux * shorten:.3f}" '
            f'y1="{y1 + uy * shorten:.3f}" '
            f'x2="{x2 - ux * shorten:.3f}" '
            f'y2="{y2 - uy * shorten:.3f}" '
            f'stroke="{color}" stroke-width="0.14"/>'
        )

    lines.append("</g>")
    return base_svg.rsplit("</svg>", 1)[0] + "\n" + "\n".join(lines) + "\n</svg>"
