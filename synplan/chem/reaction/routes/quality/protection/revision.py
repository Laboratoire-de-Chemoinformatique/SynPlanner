"""Protection/deprotection revision of detached, identity-linked routes."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import count
from typing import Any

from chython import smiles
from chython.containers import MoleculeContainer, ReactionContainer
from pydantic import BaseModel, ConfigDict

from synplan.chem.reaction import apply_reaction_rule
from synplan.chem.reaction.routes.quality.protection.config import (
    ProtectionRevisionConfig,
)
from synplan.chem.reaction.routes.quality.protection.functional_groups import (
    FunctionalGroupDetector,
    HalogenDetector,
)
from synplan.chem.reaction.routes.quality.protection.reaction_classifier import (
    get_reaction_center_atoms,
)
from synplan.chem.reaction.routes.quality.protection.scanner import (
    CompetingInteraction,
    IncompatibilityMatrix,
    RouteScanner,
)
from synplan.chem.reaction.routes.quality.protection.scorer import CompetingSitesScore
from synplan.chem.reaction.routes.quality.protection.strategy import (
    FirstAllowedClassifier,
    ProtectingGroup,
    ProtectingGroupClassifier,
    ProtectionPlanner,
    build_protection_rule,
    load_allowed_labels,
    load_protecting_groups,
)
from synplan.chem.reaction.routes.representation import build_route_cgr, compose_sb_cgr
from synplan.chem.reaction.routes.route import Route, Step
from synplan.chem.utils import molecule_key, validate_and_canonicalize

_PENALTY = {"incompatible": 1.0, "competing": 0.5, "compatible": 0.0, "unknown": 0.0}


class ProtectionRevisionAction(BaseModel):
    """One insertion; all step references are zero-based final Route positions."""

    model_config = ConfigDict(frozen=True)
    original_step_id: int
    new_step_ids: tuple[int, int, int]
    protected_step_ids: tuple[int, ...]
    fg_name: str
    fg_atoms: tuple[int, ...]
    anchor_atom: int
    severity: str
    group_label: int
    protection_class: str
    deprotection_class: str
    added_atoms: tuple[int, ...]


class ProtectionRevisionDiagnostic(BaseModel):
    """A skipped/rejected candidate. step_id refers to the input route."""

    model_config = ConfigDict(frozen=True)
    route_id: int | None = None
    step_id: int | None = None
    fg_name: str | None = None
    group_label: int | None = None
    reason: str
    detail: str | None = None


@dataclass
class RevisedRoute:
    """One detached result, including unresolved findings on its final route."""

    route: Route
    accepted: bool
    original_score: float
    revised_score: float
    actions: list[ProtectionRevisionAction]
    diagnostics: list[ProtectionRevisionDiagnostic]
    remaining_interactions: list[CompetingInteraction]
    original_to_final: dict[int, int]


@dataclass(frozen=True)
class _ProtectionGraft:
    fragment: MoleculeContainer
    attachment_atom: int
    attachment_bond: Any


def _molecule_state(molecule):
    """Mapped chemistry, independent of depiction and route history labels."""
    atoms = tuple(
        sorted(
            (n, a.atomic_number, a.isotope, a.charge, a.is_radical)
            for n, a in molecule.atoms()
        )
    )
    bonds = tuple(
        sorted((min(a, b), max(a, b), bond.order) for a, b, bond in molecule.bonds())
    )
    return atoms, bonds


def mapped_cgr_state(cgr) -> tuple:
    """Chemical SB-CGR equality including atom IDs, excluding step metadata."""
    atoms = tuple(
        sorted(
            (
                n,
                a.atomic_number,
                a.isotope,
                a.charge,
                a.p_charge,
                a.is_radical,
                a.p_is_radical,
            )
            for n, a in cgr.atoms()
        )
    )
    bonds = tuple(
        sorted(
            (min(a, b), max(a, b), bond.order, bond.p_order)
            for a, b, bond in cgr.bonds()
        )
    )
    return atoms, bonds


class ProtectionRouteReviser:
    """Insert supported protection pairs, validating every accepted Route.

    This uses graph templates, not a reaction-condition or feasibility model.
    Unknown interactions and unavailable chemistry remain explicit diagnostics.
    """

    def __init__(
        self,
        config: ProtectionRevisionConfig,
        scanner: RouteScanner,
        scorer: CompetingSitesScore,
        planner: ProtectionPlanner,
    ) -> None:
        self.config = config
        self.scanner = scanner
        self.scorer = scorer
        self.planner = planner

    @classmethod
    def from_config(
        cls,
        config: ProtectionRevisionConfig | None = None,
        classifier: ProtectingGroupClassifier | None = None,
    ) -> ProtectionRouteReviser:
        config = config or ProtectionRevisionConfig()
        scanner = RouteScanner(
            FunctionalGroupDetector(config.competing_groups_path),
            IncompatibilityMatrix(config.incompatibility_path),
            HalogenDetector(config.halogen_groups_path),
        )
        planner = ProtectionPlanner(
            load_protecting_groups(config.protecting_groups_path),
            load_allowed_labels(config.allowed_labels_path),
            classifier=classifier or FirstAllowedClassifier(),
            protect_severities=config.candidate_severities,
        )
        return cls(config, scanner, CompetingSitesScore(scanner), planner)

    def revise_routes(self, routes: dict[int, Route]) -> dict[int, RevisedRoute]:
        """Revise independently, preserving caller route IDs."""
        return {
            key: self.revise_route(route, route_id=key) for key, route in routes.items()
        }

    def revise_route(self, route: Route, route_id: int | None = None) -> RevisedRoute:
        """Return the best successive safe improvements without editing input."""
        # Chython lazily caches chemistry: even scanning runs on detached copies.
        working = deepcopy(route)
        original_score, _ = self.scorer.score_route(working)
        score = original_score
        actions = []
        diagnostics = []
        original_to_final = {i: i for i in range(len(route.steps))}
        baseline_error = None
        try:
            baseline = build_route_cgr(working, include_reactions=True)
            if baseline.mapped_route is None:
                baseline_error = (
                    baseline.diagnostic.message
                    if baseline.diagnostic is not None
                    else "Route composition returned no mapped route"
                )
            else:
                baseline_cgr = baseline.cgr
                working = baseline.mapped_route
                baseline_sb = mapped_cgr_state(compose_sb_cgr(baseline_cgr))
                original_score, _ = self.scorer.score_route(working)
                score = original_score
        except Exception as exc:
            baseline_error = str(exc)
        if baseline_error is not None:
            diagnostics.append(
                ProtectionRevisionDiagnostic(
                    route_id=route_id,
                    reason="baseline_invalid",
                    detail=baseline_error,
                )
            )
            return self._result(
                route,
                original_score,
                original_score,
                [],
                diagnostics,
                original_to_final,
            )

        for iteration in range(self.config.max_revisions_per_route):
            scan = self.scanner.scan_route(working, detailed=True)
            if scan.skipped_steps:
                diagnostics.extend(
                    ProtectionRevisionDiagnostic(
                        route_id=route_id,
                        step_id=self._original_index(i, original_to_final),
                        reason="scan_step_skipped",
                    )
                    for i in scan.skipped_steps
                )
                break
            best = None
            findings = sorted(
                scan.interactions,
                key=lambda x: (
                    -_PENALTY.get(x.severity, 0),
                    x.step_id,
                    x.site_key or "",
                ),
            )
            for interaction in findings:
                original_index = self._original_index(
                    interaction.step_id, original_to_final
                )

                def diagnose(
                    reason,
                    group=None,
                    detail=None,
                    original_index=original_index,
                    interaction=interaction,
                ):
                    diagnostics.append(
                        ProtectionRevisionDiagnostic(
                            route_id=route_id,
                            step_id=original_index,
                            fg_name=interaction.fg_name,
                            group_label=None if group is None else group.label,
                            reason=reason,
                            detail=detail,
                        )
                    )

                if interaction.severity == "unknown":
                    diagnose("interaction_unknown")
                if interaction.severity not in self.config.candidate_severities:
                    continue
                if original_index is None:
                    # Added chemistry is scanned and scored, not recursively repaired.
                    diagnose("inserted_step_interaction")
                    continue
                if (
                    interaction.anchor_atom is None
                    or interaction.template_smarts is None
                ):
                    diagnose("site_anchor_not_found")
                    continue
                source = working.steps[interaction.step_id].reaction.products[
                    interaction.product_index
                ]
                groups = self.planner.ranked_candidates(interaction, source)
                if not groups:
                    diagnose("no_supported_protecting_group")
                for group in groups:
                    try:
                        built = self._build_candidate(
                            working, interaction, group, original_index, iteration
                        )
                        if isinstance(built, str):
                            diagnose(built, group)
                            continue
                        candidate, action, step_map = built
                        reason = self._validate(
                            candidate, baseline_cgr, baseline_sb, deepcopy(route.target)
                        )
                        if reason is not None:
                            diagnose(reason, group)
                            continue
                        candidate_score = self._candidate_score(
                            candidate, action, interaction, score
                        )
                        if isinstance(candidate_score, str):
                            diagnose(candidate_score, group)
                            continue
                    except Exception as exc:
                        # Unexpected chemistry-library failures are reported once,
                        # at the candidate boundary, without stopping other repairs.
                        diagnose("candidate_invalid", group, str(exc))
                        continue
                    if best is None or candidate_score > best[0]:
                        best = candidate_score, candidate, action, step_map
            if best is None:
                break
            score, working, action, step_map = best
            actions = [
                old.model_copy(
                    update={
                        "new_step_ids": tuple(step_map[i] for i in old.new_step_ids),
                        "protected_step_ids": tuple(
                            step_map[i] for i in old.protected_step_ids
                        ),
                    }
                )
                for old in actions
            ]
            actions.append(action)
            original_to_final = {i: step_map[j] for i, j in original_to_final.items()}
        else:
            diagnostics.append(
                ProtectionRevisionDiagnostic(
                    route_id=route_id, reason="revision_limit_reached"
                )
            )
        if not actions:
            working = route
        return self._result(
            working, original_score, score, actions, diagnostics, original_to_final
        )

    def _result(self, route, original_score, score, actions, diagnostics, mapping):
        scan = self.scanner.scan_route(deepcopy(route), detailed=True)
        return RevisedRoute(
            route=route,
            accepted=bool(actions),
            original_score=original_score,
            revised_score=score,
            actions=actions,
            diagnostics=diagnostics,
            remaining_interactions=[
                x for x in scan.interactions if x.severity != "compatible"
            ],
            original_to_final=mapping,
        )

    @staticmethod
    def _original_index(index, mapping):
        return next((old for old, current in mapping.items() if current == index), None)

    def _candidate_score(self, candidate, action, interaction, score) -> float | str:
        """Return an improved score, or the reason this candidate is unsuitable."""
        scan = self.scanner.scan_route(candidate, detailed=True)
        if scan.skipped_steps:
            return "candidate_scan_incomplete"
        targeted = [
            x
            for x in scan.interactions
            if x.step_id == action.new_step_ids[1]
            and set(x.fg_atoms) & set(interaction.fg_atoms)
        ]
        if any(x.severity == "unknown" for x in targeted):
            return "target_interaction_unknown"
        if (
            max((_PENALTY[x.severity] for x in targeted), default=0)
            >= _PENALTY[interaction.severity]
        ):
            return "target_interaction_not_reduced"
        candidate_score, _ = self.scorer.score_route(candidate)
        if candidate_score - score < self.config.min_score_improvement:
            return "score_not_improved"
        return candidate_score

    def _build_candidate(
        self, route, interaction, group, original_index, action_id
    ) -> tuple[Route, ProtectionRevisionAction, dict[int, int]] | str:
        """Build a detached candidate, or return an unsupported-chemistry reason."""
        working = deepcopy(route)
        index, anchor = interaction.step_id, interaction.anchor_atom
        step = working.steps[index]
        source = step.reaction.products[interaction.product_index]
        if source is not step.product:
            return "site_on_byproduct"
        substrates = [m for m in step.reaction.reactants if m.has_atom(anchor)]
        if len(substrates) != 1:
            return "protection_substrate_not_unique"
        substrate = substrates[0]
        maximum = max(n for s in working for m in s.reaction.molecules() for n in m)
        allocator = count(maximum + 1)
        graft = self._derive_graft(
            source, interaction.template_smarts, group, allocator
        )
        if graft is None:
            return "protecting_group_graft_failed"
        reagents = self._protection_reagents(group, graft, allocator)
        if reagents is None:
            return "protection_reagent_mapping_failed"
        protected_substrate = self._apply_graft(substrate, anchor, graft)
        if protected_substrate is None:
            return "protection_substrate_invalid"

        metadata = dict(
            action_id=action_id,
            original_step_id=original_index,
            fg_name=interaction.fg_name,
            anchor_atom=anchor,
            fg_atoms=list(interaction.fg_atoms),
            group_label=group.label,
            protection_class=group.reaction_class,
            deprotection_class=group.deprotection_class,
        )

        def tagged(meta, role):
            result = deepcopy(meta)
            previous = result.setdefault("protection_revision", [])
            previous.append({**metadata, "role": role})
            return result

        protection = Step(
            ReactionContainer(
                [substrate, *reagents],
                [protected_substrate],
                meta=tagged({}, "protection"),
            ),
            protected_substrate,
        )
        # The affected path is discovered from molecule identity. A shared
        # intermediate consumed twice needs separate strategies; reject it.
        consumers = {}
        for j, current in enumerate(working):
            for mol in current.reaction.reactants:
                consumers.setdefault(id(mol), []).append(j)
        replacements = {id(substrate): protected_substrate}
        changed = {}
        carried = []
        current_index = index
        endpoint = None
        while True:
            current = working.steps[current_index]
            if not current.product.has_atom(anchor):
                return "protected_site_leaves_route"
            product = self._apply_graft(current.product, anchor, graft)
            if product is None:
                return "downstream_protection_failed"
            reactants = [replacements.get(id(m), m) for m in current.reaction.reactants]
            # Only the selected product carries the tracked site; duplicate
            # atom ownership in another product would make the graft ambiguous.
            if any(
                m.has_atom(anchor)
                for m in current.reaction.products
                if m is not current.product
            ):
                return "protected_site_product_ambiguous"
            products = [
                product if m is current.product else m
                for m in current.reaction.products
            ]
            changed[current_index] = replace(
                current,
                reaction=ReactionContainer(
                    reactants,
                    products,
                    reagents=list(current.reaction.reagents),
                    meta=tagged(current.reaction.meta, "protected_transformation"),
                ),
                product=product,
            )
            carried.append(current_index)
            following = consumers.get(id(current.product), [])
            if len(following) > 1:
                return "protected_intermediate_has_multiple_consumers"
            endpoint = current.product
            if not following:
                break
            next_index = following[0]
            next_center = get_reaction_center_atoms(working.steps[next_index].reaction)
            if set(interaction.fg_atoms) & next_center:
                break
            replacements[id(current.product)] = product
            current_index = next_index

        # The original endpoint object reconnects deprotection to every original
        # downstream consumer, or remains the final target object.
        deprotection = Step(
            ReactionContainer(
                [changed[current_index].product],
                [endpoint],
                meta=tagged({}, "deprotection"),
            ),
            endpoint,
        )
        steps = [changed.get(j, old) for j, old in enumerate(working)]
        steps.extend([protection, deprotection])
        known = {molecule_key(leaf) for leaf in route.leaves()} - route.unresolved
        unresolved = working.unresolved | frozenset(
            molecule_key(m) for m in reagents if molecule_key(m) not in known
        )
        revised = replace(working, steps=tuple(steps), unresolved=unresolved)
        positions = {id(s): j for j, s in enumerate(revised)}
        step_map = {
            j: positions[id(changed.get(j, old))] for j, old in enumerate(working)
        }
        action = ProtectionRevisionAction(
            original_step_id=original_index,
            new_step_ids=(
                positions[id(protection)],
                step_map[index],
                positions[id(deprotection)],
            ),
            protected_step_ids=tuple(step_map[j] for j in carried),
            fg_name=interaction.fg_name,
            fg_atoms=interaction.fg_atoms,
            anchor_atom=anchor,
            severity=interaction.severity,
            group_label=group.label,
            protection_class=group.reaction_class,
            deprotection_class=group.deprotection_class,
            added_atoms=tuple(range(maximum + 1, next(allocator))),
        )
        for old in carried:
            new = step_map[old]
            before = ~working.steps[old].reaction
            after = ~revised.steps[new].reaction
            if not set(before) <= set(after) or mapped_cgr_state(
                after.substructure(set(before))
            ) != mapped_cgr_state(before):
                return "original_transformation_changed"
        return revised, action, step_map

    @staticmethod
    def _validate(candidate, baseline_cgr, baseline_sb, target) -> str | None:
        """Return a violated representation contract, or None when all hold."""
        if (
            _molecule_state(candidate.target) != _molecule_state(target)
            or candidate.target != target
        ):
            return "target_changed"
        cgr = candidate.route_cgr()
        if not set(baseline_cgr) <= set(cgr):
            return "original_atoms_lost"
        if mapped_cgr_state(cgr.substructure(set(baseline_cgr))) != mapped_cgr_state(
            baseline_cgr
        ):
            return "original_cgr_chemistry_changed"
        if any(n <= max(baseline_cgr) for n in set(cgr) - set(baseline_cgr)):
            return "new_atom_mapping_invalid"
        if mapped_cgr_state(compose_sb_cgr(cgr)) != baseline_sb:
            return "sb_cgr_changed"
        return None

    @staticmethod
    def _derive_graft(
        source: MoleculeContainer,
        template_smarts: str,
        group: ProtectingGroup,
        allocator: Iterator[int],
    ) -> _ProtectionGraft | None:
        reactor = build_protection_rule(template_smarts, group)
        applications = apply_reaction_rule(
            source,
            reactor,
            top_reactions_num=max(len(source), 5),
        )
        source_atoms = set(source)
        for products in applications:
            for protected in products:
                if not source_atoms.issubset(protected):
                    continue
                fragment_atoms = set(protected) - source_atoms
                if not fragment_atoms:
                    continue
                crossing = []
                for atom1, atom2, bond in protected.bonds():
                    if (atom1 in source_atoms) == (atom2 in source_atoms):
                        continue
                    fragment_atom = atom2 if atom1 in source_atoms else atom1
                    crossing.append((fragment_atom, bond))
                if len(crossing) != 1:
                    continue
                old_attachment, attachment_bond = crossing[0]
                fragment = protected.substructure(fragment_atoms)
                mapping = {
                    atom_number: next(allocator) for atom_number in sorted(fragment)
                }
                fragment.remap(mapping)
                for _, atom in fragment.atoms():
                    atom._parsed_mapping = None
                return _ProtectionGraft(
                    fragment=fragment,
                    attachment_atom=mapping[old_attachment],
                    attachment_bond=attachment_bond.copy(full=True),
                )
        return None

    @staticmethod
    def _apply_graft(
        molecule: MoleculeContainer,
        anchor_atom: int,
        graft: _ProtectionGraft,
    ) -> MoleculeContainer | None:
        updated = molecule.union(graft.fragment)
        updated.add_bond(
            anchor_atom, graft.attachment_atom, graft.attachment_bond.copy(full=True)
        )
        canonical = validate_and_canonicalize(updated)
        if canonical is None or not set(graft.fragment).issubset(canonical):
            return None
        return canonical

    @staticmethod
    def _protection_reagents(
        group: ProtectingGroup,
        graft: _ProtectionGraft,
        allocator: Iterator[int],
    ) -> list[MoleculeContainer] | None:
        if not group.example_reagent:
            return []
        reagent = smiles(group.example_reagent)
        fragment_to_reagent = next(iter(graft.fragment.get_mapping(reagent)), None)
        if fragment_to_reagent is None:
            return None
        reagent_to_route = {
            reagent_atom: fragment_atom
            for fragment_atom, reagent_atom in fragment_to_reagent.items()
        }
        for reagent_atom in sorted(reagent):
            if reagent_atom not in reagent_to_route:
                reagent_to_route[reagent_atom] = next(allocator)
        reagent.remap(reagent_to_route)
        reactants = list(reagent.split())
        for reactant in reactants:
            reactant.meta["added_protecting_group"] = True
        return reactants
