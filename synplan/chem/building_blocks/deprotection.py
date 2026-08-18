"""Bounded building-block deprotection transformations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from chython.containers import MoleculeContainer

from .config import DeprotectionPolicy
from .rules import ProtectiveRule, protective_rules

MAX_DEPROTECTION_PASSES = 64
DeprotectionSequenceMode = Literal["enumerate", "deterministic"]


class DeprotectionSequenceLimitError(ValueError):
    """Raised before sequence enumeration exceeds its configured bound."""

    def __init__(self, *, limit: int) -> None:
        self.limit = limit
        super().__init__(
            "deprotection sequence enumeration exceeds "
            f"the configured limit of {limit} variants"
        )


@dataclass(frozen=True, slots=True)
class DeprotectionStep:
    """One taxonomy match removed from a protected molecule."""

    rule_name: str
    protected: MoleculeContainer
    deprotected: MoleculeContainer


@dataclass(frozen=True, slots=True)
class DeprotectionEvent:
    """One exact taxonomy site accepted during preparation deprotection."""

    pass_index: int
    rule_name: str
    query_mapping: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON representation stored in identity artifacts."""
        return {
            "pass_index": self.pass_index,
            "rule_name": self.rule_name,
            "query_mapping": [list(pair) for pair in self.query_mapping],
        }


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
    event_collector: list[DeprotectionEvent] | None = None,
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
    for pass_index in range(max_passes):
        if not _remove_protective_groups_once(
            molecule,
            selected,
            pass_index=pass_index,
            event_collector=event_collector,
        ):
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
    molecule: MoleculeContainer,
    rules: Mapping[str, ProtectiveRule],
    *,
    pass_index: int = 0,
    event_collector: list[DeprotectionEvent] | None = None,
) -> bool:
    to_delete: set[int] = set()
    to_add: list[tuple[int, str, int]] = []
    kept_atoms: set[int] = set()
    claimed_atoms: set[int] = set()

    accepted_events: list[DeprotectionEvent] = []
    for rule_name, rule in rules.items():
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
            accepted_events.append(
                DeprotectionEvent(
                    pass_index=pass_index,
                    rule_name=rule_name,
                    query_mapping=tuple(sorted(mapping.items())),
                )
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
    if event_collector is not None:
        event_collector.extend(accepted_events)
    return True


def _remove_one_match(
    molecule: MoleculeContainer,
    rule: ProtectiveRule,
    mapping: Mapping[int, int],
) -> None:
    """Apply one mapped taxonomy transformation in place."""
    to_delete = {
        molecule_atom
        for query_atom, molecule_atom in mapping.items()
        if query_atom not in rule.keep_atoms
    }
    kept_atoms = {mapping[number] for number in rule.keep_atoms}
    to_add = [
        (mapping[number], atom_type, bond_type)
        for number, atom_type, bond_type in rule.add_atoms
    ]
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


def _state_key(molecule: MoleculeContainer) -> str:
    """Return a canonical, atom-map-free key while retaining valid stereo."""
    return str(molecule)


def _one_step_transitions(
    molecule: MoleculeContainer,
    rules: Mapping[str, ProtectiveRule],
) -> tuple[tuple[str, MoleculeContainer], ...]:
    """Return unique next molecular states in deterministic taxonomy order."""
    transitions: dict[str, tuple[str, MoleculeContainer]] = {}
    for rule_name, rule in rules.items():
        mappings = sorted(
            rule.query.get_mapping(molecule, automorphism_filter=False),
            key=lambda item: tuple(item[number] for number in sorted(item)),
        )
        for mapping in mappings:
            candidate = molecule.copy()
            _remove_one_match(candidate, rule, mapping)
            transitions.setdefault(_state_key(candidate), (rule_name, candidate))
    return tuple(transitions.values())


def deprotect_molecule_steps(
    molecule: MoleculeContainer,
    *,
    policy: DeprotectionPolicy = "conservative",
    rules: Mapping[str, ProtectiveRule] | None = None,
    max_steps: int = MAX_DEPROTECTION_PASSES,
) -> tuple[DeprotectionStep, ...]:
    """Return the single deterministic one-group-at-a-time trace.

    The first applicable taxonomy rule and its first deterministic site match
    are selected at each state. Atom numbers are retained across states so the
    reversed trace can be emitted as mapped protection reactions.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    selected = _rules_for_policy(policy, rules)
    current = molecule.copy()
    visited = {_state_key(current)}
    steps: list[DeprotectionStep] = []
    for _ in range(max_steps):
        transitions = _one_step_transitions(current, selected)
        if not transitions:
            return tuple(steps)
        rule_name, deprotected = transitions[0]
        state = _state_key(deprotected)
        if state in visited:
            raise RuntimeError("deprotection transformations entered a cycle")
        steps.append(DeprotectionStep(rule_name, current.copy(), deprotected.copy()))
        current = deprotected
        visited.add(state)
    raise RuntimeError(f"deprotection did not converge within {max_steps} steps")


def deprotect_molecule_traces(
    molecule: MoleculeContainer,
    *,
    policy: DeprotectionPolicy = "conservative",
    sequence_mode: DeprotectionSequenceMode = "enumerate",
    rules: Mapping[str, ProtectiveRule] | None = None,
    max_steps: int = MAX_DEPROTECTION_PASSES,
    max_variants: int = 100,
) -> tuple[tuple[DeprotectionStep, ...], ...]:
    """Return bounded traces for valid protecting-group removal sequences.

    Enumeration explores every unique next molecular state and retains complete
    traces that reach the same endpoint as the preparation deprotection logic.
    Trace identity is the complete sequence of canonical intermediate molecular
    structures, so symmetry-equivalent site choices are collapsed. The input
    molecule is never modified.
    """
    if sequence_mode not in {"enumerate", "deterministic"}:
        raise ValueError("sequence_mode must be 'enumerate' or 'deterministic'")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if max_variants < 1:
        raise ValueError("max_variants must be at least 1")
    if sequence_mode == "deterministic":
        return (
            deprotect_molecule_steps(
                molecule,
                policy=policy,
                rules=rules,
                max_steps=max_steps,
            ),
        )

    selected = _rules_for_policy(policy, rules)
    expected = molecule.copy()
    remove_protective_groups(
        expected,
        policy=policy,
        rules=rules,
        max_passes=max_steps + 1,
    )
    expected_key = _state_key(expected)
    traces: list[tuple[DeprotectionStep, ...]] = []
    trace_keys: set[tuple[str, ...]] = set()

    def visit(
        current: MoleculeContainer,
        steps: tuple[DeprotectionStep, ...],
        visited: frozenset[str],
    ) -> None:
        transitions = _one_step_transitions(current, selected)
        if not transitions:
            if _state_key(current) != expected_key:
                return
            trace_key = tuple(_state_key(step.deprotected) for step in steps)
            if trace_key in trace_keys:
                return
            if len(traces) >= max_variants:
                raise DeprotectionSequenceLimitError(limit=max_variants)
            trace_keys.add(trace_key)
            traces.append(steps)
            return
        if len(steps) >= max_steps:
            raise RuntimeError(
                f"deprotection did not converge within {max_steps} steps"
            )
        for rule_name, deprotected in transitions:
            state = _state_key(deprotected)
            if state in visited:
                raise RuntimeError("deprotection transformations entered a cycle")
            visit(
                deprotected,
                (
                    *steps,
                    DeprotectionStep(
                        rule_name,
                        current.copy(),
                        deprotected.copy(),
                    ),
                ),
                visited | {state},
            )

    initial = molecule.copy()
    visit(initial, (), frozenset({_state_key(initial)}))
    return tuple(traces)


__all__ = [
    "MAX_DEPROTECTION_PASSES",
    "DeprotectionEvent",
    "DeprotectionSequenceLimitError",
    "DeprotectionSequenceMode",
    "DeprotectionStep",
    "deprotect_molecule",
    "deprotect_molecule_steps",
    "deprotect_molecule_traces",
    "remove_protective_groups",
]
