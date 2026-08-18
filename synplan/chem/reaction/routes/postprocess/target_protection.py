"""Restore protection steps above a route planned for a deprotected target."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from chython import smiles as smiles_parser
from chython.containers import MoleculeContainer, ReactionContainer

from synplan.chem.building_blocks.config import DeprotectionPolicy
from synplan.chem.building_blocks.deprotection import (
    DeprotectionSequenceMode,
    deprotect_molecule,
    deprotect_molecule_traces,
)
from synplan.chem.building_blocks.provenance import (
    current_protective_rules_sha256,
    validate_deprotection_provenance,
)
from synplan.chem.reaction.routes.tree_ops import (
    max_route_atom_map,
    node_children,
    reindex_reaction_steps,
)


class TargetProtectionError(ValueError):
    """Raised when a protected target cannot be reconciled with a route root."""


def _molecule(value: object, *, context: str) -> MoleculeContainer:
    if not isinstance(value, str) or not value.strip():
        raise TargetProtectionError(f"{context}: expected non-empty molecule SMILES")
    try:
        parsed = smiles_parser(value)
    except Exception as error:
        raise TargetProtectionError(f"{context}: invalid molecule SMILES") from error
    if not isinstance(parsed, MoleculeContainer):
        raise TargetProtectionError(f"{context}: expected one molecule")
    return parsed


def _same_graph(first: MoleculeContainer, second: MoleculeContainer) -> bool:
    first_plain = first.copy()
    first_plain.clean_stereo()
    second_plain = second.copy()
    second_plain.clean_stereo()
    if len(first_plain) != len(second_plain):
        return False
    if sum(1 for _ in first_plain.bonds()) != sum(1 for _ in second_plain.bonds()):
        return False
    return (
        next(
            iter(first_plain.get_mapping(second_plain, automorphism_filter=False)), None
        )
        is not None
    )


def _mapped_route_root(route: Mapping[str, Any]) -> MoleculeContainer:
    root = _molecule(route.get("smiles"), context="route root")
    reactions = [
        child
        for child in node_children(route)
        if isinstance(child, Mapping) and child.get("type") == "reaction"
    ]
    if not reactions:
        root.remap({number: number for number in root})
        return root
    if len(reactions) != 1:
        raise TargetProtectionError("route root must have exactly one reaction child")
    try:
        reaction = smiles_parser(reactions[0]["smiles"])
    except Exception as error:
        raise TargetProtectionError("route root reaction has invalid SMILES") from error
    if not isinstance(reaction, ReactionContainer):
        raise TargetProtectionError("route root child must contain a reaction")
    matches = [
        product.copy() for product in reaction.products if _same_graph(root, product)
    ]
    if len(matches) != 1:
        raise TargetProtectionError(
            "route root must match exactly one product of its reaction child"
        )
    return matches[0]


def route_root_matches_target(route: Mapping[str, Any], target_smiles: str) -> bool:
    """Return whether target and route root differ only by stereochemistry."""
    if not isinstance(route, Mapping) or route.get("type") != "mol":
        raise TypeError("route must be a JSON-like molecule-root mapping")
    target = _molecule(target_smiles, context="target")
    return _same_graph(target, _mapped_route_root(route))


def _mapped_states(
    states: list[MoleculeContainer],
    mapped_deprotected_root: MoleculeContainer,
    *,
    first_new_atom_map: int,
) -> list[MoleculeContainer]:
    final_plain = states[-1].copy()
    final_plain.clean_stereo()
    root_plain = mapped_deprotected_root.copy()
    root_plain.clean_stereo()
    if len(final_plain) != len(root_plain) or sum(
        1 for _ in final_plain.bonds()
    ) != sum(1 for _ in root_plain.bonds()):
        raise TargetProtectionError(
            "protected target does not deprotect to the route root"
        )
    mapping = next(
        iter(final_plain.get_mapping(root_plain, automorphism_filter=False)), None
    )
    if mapping is None or len(mapping) != len(final_plain):
        raise TargetProtectionError(
            "protected target does not deprotect to the route root"
        )

    remapping = dict(mapping)
    next_atom_map = first_new_atom_map
    all_atom_numbers = dict.fromkeys(number for state in states for number in state)
    for number in all_atom_numbers:
        if number not in remapping:
            remapping[number] = next_atom_map
            next_atom_map += 1

    result = []
    for state in states:
        mapped = state.copy()
        mapped.remap({number: remapping[number] for number in state})
        result.append(mapped)
    return result


def restore_protected_target(
    route: Mapping[str, Any],
    protected_target_smiles: str,
    *,
    policy: DeprotectionPolicy = "conservative",
    sequence_mode: DeprotectionSequenceMode = "enumerate",
    max_steps: int = 64,
    max_variants: int = 100,
    preprocessing_provenance: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    """Return routes containing explicit final target-protection sequences.

    The protected target is deprotected using the preparation taxonomy one match
    at a time. By default every unique intermediate molecular sequence is
    enumerated; deterministic mode retains only the first taxonomy-ordered trace.
    Each trace is reversed into synthesis-direction protection reactions, so a
    route ending at ``T_deprotected`` becomes
    ``... -> T_protected_1 -> ... -> T``. Reintroduced atoms receive map
    numbers strictly above the current route maximum. The input route is not
    modified, and the return type is always a list.

    These are bookkeeping reactions: they encode the molecular transformation
    and taxonomy rule but do not claim reagents, conditions, or feasibility.
    """
    if not isinstance(route, Mapping) or route.get("type") != "mol":
        raise TypeError("route must be a JSON-like molecule-root mapping")
    protected_target = _molecule(protected_target_smiles, context="protected target")
    taxonomy_digest = current_protective_rules_sha256()
    recorded_endpoint: MoleculeContainer | None = None
    provenance_mode = "runtime_inference"
    if preprocessing_provenance is not None:
        try:
            recorded_reaction = validate_deprotection_provenance(
                preprocessing_provenance,
                context="protected target provenance",
                required=True,
            )
        except ValueError as error:
            raise TargetProtectionError(str(error)) from error
        if recorded_reaction is None:
            raise TargetProtectionError(
                "protected target provenance has no mapped transformation"
            )
        recorded_policy = str(preprocessing_provenance.get("deprotection_policy") or "")
        if recorded_policy != policy:
            raise TargetProtectionError(
                "protected target provenance policy "
                f"{recorded_policy!r} does not match requested policy {policy!r}"
            )
        recorded_digest = str(
            preprocessing_provenance.get("protective_rules_sha256") or ""
        )
        if recorded_digest != taxonomy_digest:
            raise TargetProtectionError(
                "protecting-group taxonomy differs from target preprocessing: "
                f"recorded {recorded_digest}, current {taxonomy_digest}; "
                "the original taxonomy is required to enumerate exact sequences"
            )
        if not _same_graph(recorded_reaction.reactants[0], protected_target):
            raise TargetProtectionError(
                "protected target does not match its preprocessing provenance"
            )
        recorded_endpoint = recorded_reaction.products[0]
        provenance_mode = "exact"
    traces = deprotect_molecule_traces(
        protected_target,
        policy=policy,
        sequence_mode=sequence_mode,
        max_steps=max_steps,
        max_variants=max_variants,
    )
    expected = deprotect_molecule(
        protected_target, policy=policy, max_passes=max_steps + 1
    )
    if recorded_endpoint is not None and not _same_graph(expected, recorded_endpoint):
        raise TargetProtectionError(
            "current deprotection disagrees with target preprocessing provenance"
        )
    mapped_root = _mapped_route_root(route)
    if not _same_graph(expected, mapped_root):
        raise TargetProtectionError(
            "protected target does not deprotect to the route root"
        )
    current_max = max(
        max_route_atom_map(route),
        max(mapped_root.atoms_numbers, default=0),
    )
    variants: list[dict[str, Any]] = []
    for variant_index, steps in enumerate(traces):
        traced_endpoint = steps[-1].deprotected if steps else protected_target
        if not _same_graph(expected, traced_endpoint):
            raise TargetProtectionError(
                "one-group deprotection trace disagrees with preparation deprotection"
            )
        if not _same_graph(traced_endpoint, mapped_root):
            raise TargetProtectionError(
                "protected target does not deprotect to the route root"
            )

        restored = deepcopy(dict(route))
        if not steps:
            variants.append(restored)
            continue

        states = [steps[0].protected, *(step.deprotected for step in steps)]
        states = _mapped_states(
            states,
            mapped_root,
            first_new_atom_map=current_max + 1,
        )
        restored["smiles"] = str(states[-1])

        subtree = restored
        protection_rule_sequence: list[str] = []
        for protection_order, index in enumerate(
            range(len(steps) - 1, -1, -1),
            start=1,
        ):
            less_protected = states[index + 1]
            more_protected = states[index]
            reaction = ReactionContainer(
                reactants=[less_protected.copy()],
                products=[more_protected.copy()],
            )
            protection_rule_sequence.append(steps[index].rule_name)
            subtree = {
                "type": "mol",
                "smiles": str(more_protected),
                "in_stock": False,
                "children": [
                    {
                        "type": "reaction",
                        "smiles": format(reaction, "m"),
                        "children": [subtree],
                        "meta": {
                            "reaction_class": "protection",
                            "source": "protective_group_taxonomy",
                            "protective_rule": steps[index].rule_name,
                            "protection_order": protection_order,
                            "bookkeeping": True,
                            "preprocessing_provenance": provenance_mode,
                            "deprotection_policy": policy,
                            "protective_rules_sha256": taxonomy_digest,
                        },
                    }
                ],
            }

        subtree["target_protection_restored"] = True
        subtree["target_protection_sequence_mode"] = sequence_mode
        subtree["target_protection_variant_index"] = variant_index
        subtree["target_protection_rule_sequence"] = protection_rule_sequence
        subtree["target_protection_steps"] = len(steps)
        subtree["target_preprocessing_provenance"] = provenance_mode
        subtree["target_deprotection_policy"] = policy
        subtree["target_protective_rules_sha256"] = taxonomy_digest
        reindex_reaction_steps(subtree)
        variants.append(subtree)
    if not variants:
        raise TargetProtectionError(
            "no valid deprotection sequence reaches the route root"
        )
    return variants


__all__ = [
    "TargetProtectionError",
    "restore_protected_target",
    "route_root_matches_target",
]
