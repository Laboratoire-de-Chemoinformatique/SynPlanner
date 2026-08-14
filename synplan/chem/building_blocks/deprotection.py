"""Bounded building-block deprotection transformations."""

from __future__ import annotations

from collections.abc import Mapping

from chython.containers import MoleculeContainer

from .config import DeprotectionPolicy
from .rules import ProtectiveRule, protective_rules

MAX_DEPROTECTION_PASSES = 64


def _rules_for_policy(
    policy: DeprotectionPolicy,
    rules: Mapping[str, ProtectiveRule] | None = None,
) -> dict[str, ProtectiveRule]:
    if policy not in {"conservative", "aggressive"}:
        raise ValueError("policy must be 'conservative' or 'aggressive'")
    source = protective_rules if rules is None else rules
    if policy == "aggressive":
        return dict(source)
    return {name: rule for name, rule in source.items() if rule.policy == policy}


def remove_protective_groups(
    molecule: MoleculeContainer,
    *,
    policy: DeprotectionPolicy = "conservative",
    rules: Mapping[str, ProtectiveRule] | None = None,
    max_passes: int = MAX_DEPROTECTION_PASSES,
) -> bool:
    """Remove selected protecting groups in place until a bounded fixed point.

    A visited-state guard prevents a malformed/custom transformation set from cycling.
    Exceeding ``max_passes`` is a hard failure rather than returning a partially
    transformed building block.
    """
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")
    selected = _rules_for_policy(policy, rules)
    visited = {str(molecule)}
    changed = False
    for _ in range(max_passes):
        if not _remove_protective_groups_once(molecule, selected):
            return changed
        changed = True
        state = str(molecule)
        if state in visited:
            raise RuntimeError("deprotection transformations entered a cycle")
        visited.add(state)
    raise RuntimeError(f"deprotection did not converge within {max_passes} passes")


def deprotect_molecule(
    molecule: MoleculeContainer,
    *,
    policy: DeprotectionPolicy = "conservative",
    max_passes: int = MAX_DEPROTECTION_PASSES,
) -> MoleculeContainer:
    """Return a deprotected copy without mutating the source molecule."""
    result = molecule.copy()
    remove_protective_groups(result, policy=policy, max_passes=max_passes)
    return result


def _remove_protective_groups_once(
    molecule: MoleculeContainer, rules: Mapping[str, ProtectiveRule]
) -> bool:
    to_delete: set[int] = set()
    to_add: list[tuple[int, str, int]] = []
    kept_atoms: set[int] = set()
    claimed_atoms: set[int] = set()

    for rule in rules.values():
        for mapping in rule.query.get_mapping(molecule, automorphism_filter=False):
            mapped_atoms = set(mapping.values())
            if not claimed_atoms.isdisjoint(mapped_atoms):
                continue
            deletable_atoms = {
                molecule_atom
                for query_atom, molecule_atom in mapping.items()
                if query_atom not in rule.keep_atoms
            }
            if not to_delete.isdisjoint(deletable_atoms):
                continue
            claimed_atoms.update(mapped_atoms)
            to_delete.update(deletable_atoms)
            kept_atoms.update(mapping[number] for number in rule.keep_atoms)
            to_add.extend(
                (mapping[number], atom_type, bond_type)
                for number, atom_type, bond_type in rule.add_atoms
            )

    if not to_delete and not to_add:
        return False
    for atom_number, atom_type, bond_type in to_add:
        new_atom = molecule.add_atom(atom_type, _skip_calculation=True)
        molecule.add_bond(new_atom, atom_number, bond_type, _skip_calculation=True)
    for atom_number in to_delete:
        molecule.delete_atom(atom_number, _skip_calculation=True)

    if molecule._changed is not None:
        molecule._changed.intersection_update(molecule._atoms)
    molecule.fix_structure()
    for atom_number in kept_atoms:
        atom = molecule.atom(atom_number)
        if (
            atom.atomic_symbol == "N"
            and atom.hybridization == 4
            and atom.implicit_hydrogens is None
        ):
            atom._implicit_hydrogens = 1
    molecule.fix_stereo()
    return True


__all__ = [
    "MAX_DEPROTECTION_PASSES",
    "deprotect_molecule",
    "remove_protective_groups",
]
