"""Deterministic protection/deprotection route revision.

This module implements a conservative, chython-native post-search revision pass.
Protecting-group rows are treated as fragment descriptors, not as executable
reaction templates.

Revision support is intentionally limited to amine, hydroxyl/phenol, and
carbonyl-acetal protection in this pass.  Carboxyl, thiol, and diol rows remain
loaded as metadata-only protecting-group data for future implementations.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, replace
from typing import Any

from chython import smarts, smiles
from chython.containers import MoleculeContainer, ReactionContainer
from pydantic import BaseModel, ConfigDict

from synplan.chem.utils import validate_and_canonicalize
from synplan.routes.quality.protection.config import (
    ProtectionConfig,
    ProtectionRevisionConfig,
)
from synplan.routes.quality.protection.functional_groups import (
    FunctionalGroupDetector,
    HalogenDetector,
)
from synplan.routes.quality.protection.scanner import (
    CompetingInteraction,
    IncompatibilityMatrix,
    RouteScanner,
)
from synplan.routes.quality.protection.reaction_classifier import (
    get_reaction_center_atoms,
)
from synplan.routes.quality.protection.scorer import CompetingSitesScore
from synplan.routes.route_cgr import compose_route_cgr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProtectionFragment:
    """A chython-parsed protecting group fragment."""

    rule_name: str
    reaction_class: str
    smarts: str
    atoms_to_keep: tuple[int, ...]
    atoms_to_add: tuple[tuple[int, str, int], ...]
    protected_example: str
    cleaved_example: str
    decoys: tuple[str, ...]
    h2o: str
    bases: str
    nucleophiles: str
    electrophiles: str
    reduction: str
    oxidation: str
    p_mol: str
    molecule: MoleculeContainer | None
    attachment_atoms: tuple[int, ...]
    attachment_bonds: tuple[tuple[int, int], ...]
    strategy: str

    @property
    def attachment_atom(self) -> int | None:
        """Compatibility accessor for single-anchor fragments."""

        if len(self.attachment_atoms) != 1:
            return None
        return self.attachment_atoms[0]

    @property
    def attachment_bond_order(self) -> int | None:
        """Compatibility accessor for single-anchor attachment bond order."""

        if len(self.attachment_bonds) != 1:
            return None
        return self.attachment_bonds[0][1]


class ProtectionAction(BaseModel):
    """Accepted protection/deprotection insertion."""

    model_config = ConfigDict(frozen=True)

    original_step_id: int
    new_step_ids: tuple[int, int, int]
    fg_name: str
    fg_atoms: tuple[int, ...]
    anchor_atom: int
    severity: str
    rule_name: str
    strategy: str
    p_mol: str
    protection_class: str


class ProtectionRevisionDiagnostic(BaseModel):
    """Reason a candidate site or protecting group was skipped."""

    model_config = ConfigDict(frozen=True)

    route_id: int | None = None
    step_id: int | None = None
    fg_name: str | None = None
    site_key: str | None = None
    rule_name: str | None = None
    strategy: str | None = None
    candidate_score: float | None = None
    baseline_score: float | None = None
    reason: str
    detail: str | None = None


class RevisedRoute(BaseModel):
    """Route revision result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    route: dict[int, ReactionContainer]
    accepted: bool
    original_score: float
    revised_score: float
    actions: list[ProtectionAction]
    diagnostics: list[ProtectionRevisionDiagnostic]
    route_metadata: dict[int, dict[str, Any]]


class ProtectionFragmentCatalog:
    """Load and filter bundled protecting-group fragment data."""

    _SCHEMA = (
        "rule_name",
        "reaction_class",
        "smarts",
        "atoms_to_keep",
        "atoms_to_add",
        "protected_example",
        "cleaved_example",
        "decoys",
        "h2o",
        "bases",
        "nucleophiles",
        "electrophiles",
        "reduction",
        "oxidation",
        "p_mol",
    )
    _METADATA_ONLY_FAMILIES = {"CO2H", "thiol", "diol"}

    def __init__(
        self,
        templates_path: str,
        label_mapping_path: str,
    ) -> None:
        del label_mapping_path
        self._fragments: list[ProtectionFragment] = []
        self._chython_fragments_by_family: dict[str, list[ProtectionFragment]] = {}
        self.diagnostics: list[ProtectionRevisionDiagnostic] = []
        self._load_templates(templates_path)

    @classmethod
    def from_config(cls, config: ProtectionConfig) -> "ProtectionFragmentCatalog":
        return cls(
            templates_path=config.protection_group_templates_path,
            label_mapping_path=config.reactive_function_label_mapping_path,
        )

    def candidates_for_fg(self, fg_name: str) -> list[ProtectionFragment]:
        family = self._fg_family(fg_name)
        if family is None or family in self._METADATA_ONLY_FAMILIES:
            return []

        candidates: list[ProtectionFragment] = []
        for fragment in self._chython_fragments_by_family.get(family, ()):
            if self._fragment_matches_fg(fragment, fg_name, family):
                candidates.append(fragment)
        return candidates

    @property
    def fragments(self) -> tuple[ProtectionFragment, ...]:
        """All loaded fragment rows, including metadata-only unsupported rows."""

        return tuple(self._fragments)

    def _load_templates(self, path: str) -> None:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            self._validate_schema(reader.fieldnames, path)
            for row in reader:
                fragment = self._fragment_from_row(row)
                if fragment is None:
                    continue
                self._fragments.append(fragment)
                family = self._family_from_rule_name(fragment.rule_name)
                if family is None:
                    continue
                if family in self._METADATA_ONLY_FAMILIES:
                    self.diagnostics.append(
                        ProtectionRevisionDiagnostic(
                            rule_name=fragment.rule_name,
                            strategy=fragment.strategy,
                            reason="unsupported_protection_family",
                            detail=family,
                        )
                    )
                    continue
                self._chython_fragments_by_family.setdefault(family, []).append(
                    fragment
                )

    @classmethod
    def _validate_schema(cls, fieldnames: list[str] | None, path: str) -> None:
        if not fieldnames:
            raise ValueError(f"{path} is empty or missing a header row")
        if len(fieldnames) == 1 and "," in fieldnames[0]:
            raise ValueError(
                f"{path} must be tab-delimited with the protection rule schema; "
                "it looks comma-delimited"
            )
        expected = list(cls._SCHEMA)
        if fieldnames != expected:
            missing = [name for name in expected if name not in fieldnames]
            extra = [name for name in fieldnames if name not in expected]
            details: list[str] = []
            if missing:
                details.append(f"missing columns: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected columns: {', '.join(extra)}")
            if not details:
                details.append("columns are not in the required order")
            raise ValueError(
                f"{path} has an invalid protection template schema "
                f"({'; '.join(details)})"
            )

    def _fragment_from_row(
        self,
        row: dict[str, str | None],
    ) -> ProtectionFragment | None:
        rule_name = (row.get("rule_name") or "").strip()
        reaction_class = (row.get("reaction_class") or "").strip()
        molecule: MoleculeContainer | None = None
        attachment_atoms: tuple[int, ...] = ()
        attachment_bonds: tuple[tuple[int, int], ...] = ()

        if rule_name and row.get("smarts") and row.get("protected_example"):
            derived = self._derive_chython_fragment(
                rule_name=rule_name,
                smarts_value=row["smarts"] or "",
                protected_example=row["protected_example"] or "",
                atoms_to_keep=self._parse_int_tuple(row.get("atoms_to_keep")),
            )
            if derived is not None:
                molecule, attachment_atoms, attachment_bonds = derived

        fragment = ProtectionFragment(
            rule_name=rule_name,
            reaction_class=reaction_class,
            smarts=(row.get("smarts") or "").strip(),
            atoms_to_keep=self._parse_int_tuple(row.get("atoms_to_keep")),
            atoms_to_add=self._parse_atoms_to_add(row.get("atoms_to_add")),
            protected_example=(row.get("protected_example") or "").strip(),
            cleaved_example=(row.get("cleaved_example") or "").strip(),
            decoys=self._parse_string_tuple(row.get("decoys")),
            h2o=self._metadata_value(row.get("h2o")),
            bases=self._metadata_value(row.get("bases")),
            nucleophiles=self._metadata_value(row.get("nucleophiles")),
            electrophiles=self._metadata_value(row.get("electrophiles")),
            reduction=self._metadata_value(row.get("reduction")),
            oxidation=self._metadata_value(row.get("oxidation")),
            p_mol=self._metadata_value(row.get("p_mol")),
            molecule=molecule,
            attachment_atoms=attachment_atoms,
            attachment_bonds=attachment_bonds,
            strategy="unsupported",
        )
        strategy = self._infer_strategy(fragment)
        return replace(fragment, strategy=strategy)

    def _derive_chython_fragment(
        self,
        *,
        rule_name: str,
        smarts_value: str,
        protected_example: str,
        atoms_to_keep: tuple[int, ...],
    ) -> tuple[MoleculeContainer, tuple[int, ...], tuple[tuple[int, int], ...]] | None:
        if not atoms_to_keep:
            return None
        try:
            query = smarts(smarts_value)
            protected = smiles(protected_example)
            mapping = next(iter(query.get_mapping(protected)), None)
        except Exception as exc:
            self.diagnostics.append(
                ProtectionRevisionDiagnostic(
                    reason="chython_rule_fragment_derivation_failed",
                    detail=f"{rule_name}: {exc}",
                )
            )
            return None
        if not mapping:
            return None

        try:
            keep_atoms = {mapping[atom_id] for atom_id in atoms_to_keep}
        except KeyError:
            self.diagnostics.append(
                ProtectionRevisionDiagnostic(
                    reason="chython_rule_keep_atom_missing",
                    detail=rule_name,
                )
            )
            return None

        matched_atoms = set(mapping.values())
        fragment_atoms = matched_atoms - keep_atoms
        if not fragment_atoms:
            return None

        attachment_bonds: set[tuple[int, int]] = set()
        for atom1, atom2, bond in protected.bonds():
            bond_order = int(getattr(bond, "order", bond))
            if atom1 in keep_atoms and atom2 in fragment_atoms:
                attachment_bonds.add((atom2, bond_order))
            elif atom2 in keep_atoms and atom1 in fragment_atoms:
                attachment_bonds.add((atom1, bond_order))
        if not attachment_bonds:
            return None

        try:
            fragment = protected.substructure(fragment_atoms)
        except Exception as exc:
            self.diagnostics.append(
                ProtectionRevisionDiagnostic(
                    reason="chython_rule_fragment_derivation_failed",
                    detail=f"{rule_name}: {exc}",
                )
            )
            return None
        attachment_atoms = tuple(sorted({atom for atom, _order in attachment_bonds}))
        return fragment, attachment_atoms, tuple(sorted(attachment_bonds))

    @staticmethod
    def _parse_int_tuple(raw: str | None) -> tuple[int, ...]:
        if raw is None or not str(raw).strip():
            return ()
        value = json.loads(raw)
        return tuple(int(item) for item in value)

    @staticmethod
    def _parse_atoms_to_add(raw: str | None) -> tuple[tuple[int, str, int], ...]:
        if raw is None or not str(raw).strip():
            return ()
        value = json.loads(raw)
        return tuple(
            (int(anchor), str(symbol), int(order))
            for anchor, symbol, order in value
        )

    @staticmethod
    def _parse_string_tuple(raw: str | None) -> tuple[str, ...]:
        if raw is None or not str(raw).strip():
            return ()
        value = json.loads(raw)
        return tuple(str(item) for item in value)

    @staticmethod
    def _metadata_value(raw: str | None) -> str:
        value = (raw or "").strip()
        return value if value else "None"

    @classmethod
    def _infer_strategy(cls, fragment: ProtectionFragment) -> str:
        family = cls._family_from_rule_name(fragment.rule_name)
        if family in cls._METADATA_ONLY_FAMILIES:
            return "unsupported"

        if family in {"N", "O"}:
            if (
                fragment.atoms_to_keep == (1,)
                and not fragment.atoms_to_add
                and fragment.molecule is not None
                and len(fragment.attachment_bonds) == 1
                and fragment.attachment_bond_order in {1, 2, 3}
            ):
                return "single_anchor"
        if family == "carbonyl" or fragment.reaction_class == "carbonyl":
            if (
                fragment.atoms_to_add == ((1, "O", 2),)
                and fragment.molecule is not None
                and len(fragment.attachment_atoms) == 2
            ):
                return "carbonyl_acetal"
        return "unsupported"

    @staticmethod
    def _fg_family(fg_name: str) -> str | None:
        if "AminoAcid" in fg_name:
            return "N"
        if "Alcohol" in fg_name or "Phenol" in fg_name:
            return "O"
        if "Acid" in fg_name:
            return "CO2H"
        if "Aldehyde" in fg_name or "Ketone" in fg_name:
            return "carbonyl"
        if "Thiol" in fg_name:
            return "thiol"
        if "Diol" in fg_name:
            return "diol"
        if (
            "Amine" in fg_name
            or "Amino" in fg_name
            or "Benzylamine" in fg_name
            or "Hydrazine" in fg_name
            or "HeterocycleNH" in fg_name
            or "Hydroxylamine" in fg_name
            or "Sulfinylamine" in fg_name
        ):
            return "N"
        return None

    @staticmethod
    def _family_from_rule_name(rule_name: str) -> str | None:
        if rule_name.startswith("hydroxyl_"):
            return "O"
        if rule_name.startswith("amine_"):
            return "N"
        if rule_name.startswith("carbonyl_"):
            return "carbonyl"
        if rule_name.startswith("carboxyl_"):
            return "CO2H"
        if rule_name.startswith("thiol_"):
            return "thiol"
        if rule_name.startswith("diol_"):
            return "diol"
        return None

    @classmethod
    def _fragment_matches_fg(
        cls,
        fragment: ProtectionFragment,
        fg_name: str,
        family: str,
    ) -> bool:
        if fragment.strategy == "unsupported":
            return False
        fragment_family = (
            cls._family_from_rule_name(fragment.rule_name)
            or fragment.reaction_class
        )
        return fragment_family == family


class ProtectionRouteReviser:
    """Insert deterministic protection/deprotection steps into route dicts."""

    _RULE_PRIORITY = {
        "amine_boc": 0,
        "amine_cbz": 1,
        "amine_fmoc": 2,
        "amine_benzyl": 3,
        "hydroxyl_tbs": 0,
        "hydroxyl_tbdps": 1,
        "hydroxyl_benzyl": 2,
        "hydroxyl_acyl": 3,
        "hydroxyl_mom": 4,
        "carbonyl_dioxolane": 0,
        "carbonyl_dioxane": 1,
        "carbonyl_dithiane": 2,
        "carbonyl_dithiolane": 3,
        "carbonyl_dimethoxy": 4,
    }
    _SEVERITY_PENALTY = {
        "incompatible": 1.0,
        "competing": 0.5,
        "compatible": 0.0,
    }

    def __init__(
        self,
        scanner: RouteScanner,
        scorer: CompetingSitesScore,
        catalog: ProtectionFragmentCatalog,
        config: ProtectionRevisionConfig | None = None,
    ) -> None:
        self.scanner = scanner
        self.scorer = scorer
        self.catalog = catalog
        self.config = config or ProtectionRevisionConfig()

    @classmethod
    def from_config(
        cls,
        config: ProtectionRevisionConfig | ProtectionConfig | None = None,
    ) -> "ProtectionRouteReviser":
        if config is None:
            revision_config = ProtectionRevisionConfig()
        elif isinstance(config, ProtectionRevisionConfig):
            revision_config = config
        else:
            revision_config = ProtectionRevisionConfig(**config.model_dump())

        detector = FunctionalGroupDetector(revision_config.competing_groups_path)
        matrix = IncompatibilityMatrix(revision_config.incompatibility_path)
        halogen = HalogenDetector(revision_config.halogen_groups_path)
        scanner = RouteScanner(detector, matrix, halogen_detector=halogen)
        scorer = CompetingSitesScore(scanner)
        catalog = ProtectionFragmentCatalog.from_config(revision_config)
        return cls(scanner, scorer, catalog, revision_config)

    def revise_routes(
        self,
        routes: dict[int, dict[int, ReactionContainer]],
    ) -> dict[int, RevisedRoute]:
        return {
            route_id: self.revise_route(route, route_id=route_id)
            for route_id, route in routes.items()
        }

    def revise_route(
        self,
        route: dict[int, ReactionContainer],
        route_id: int | None = None,
    ) -> RevisedRoute:
        current_route = dict(route)
        original_score, _ = self.scorer.score_route(current_route)
        current_score = original_score
        actions: list[ProtectionAction] = []
        diagnostics: list[ProtectionRevisionDiagnostic] = []
        route_metadata: dict[int, dict[str, Any]] = {}

        for _ in range(self.config.max_revisions_per_route):
            candidate = self._best_revision_candidate(
                current_route,
                current_score,
                route_id,
                route_metadata,
            )
            diagnostics.extend(candidate["diagnostics"])
            if candidate["route"] is None:
                break

            current_route = candidate["route"]
            current_score = candidate["score"]
            actions.append(candidate["action"])
            route_metadata = candidate["route_metadata"]

        return RevisedRoute(
            route=current_route,
            accepted=bool(actions),
            original_score=original_score,
            revised_score=current_score,
            actions=actions,
            diagnostics=diagnostics,
            route_metadata=route_metadata,
        )

    def _best_revision_candidate(
        self,
        route: dict[int, ReactionContainer],
        current_score: float,
        route_id: int | None,
        existing_metadata: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        scan_result = self.scanner.scan_route(route, detailed=True)
        interactions = self._actionable_interactions(scan_result.interactions)
        diagnostics: list[ProtectionRevisionDiagnostic] = []
        best: dict[str, Any] = {
            "route": None,
            "score": current_score,
            "action": None,
            "route_metadata": {},
            "fragment_priority": (10_000, ""),
            "diagnostics": diagnostics,
        }

        for interaction in interactions:
            site_diagnostics = self._site_diagnostics(interaction, route_id)
            if site_diagnostics:
                diagnostics.extend(site_diagnostics)
                continue

            candidates = sorted(
                self.catalog.candidates_for_fg(interaction.fg_name),
                key=self._fragment_sort_key,
            )
            if not candidates:
                diagnostics.append(
                    ProtectionRevisionDiagnostic(
                        route_id=route_id,
                        step_id=interaction.step_id,
                        fg_name=interaction.fg_name,
                        site_key=interaction.site_key,
                        reason="no_chython_fragment_candidates",
                    )
                )
                continue

            for fragment in candidates:
                candidate = self._build_candidate_route(
                    route,
                    interaction,
                    fragment,
                    route_id,
                    existing_metadata,
                )
                diagnostics.extend(candidate["diagnostics"])
                if candidate["route"] is None:
                    continue

                score, candidate_interactions = self.scorer.score_route(
                    candidate["route"]
                )
                if score <= current_score + self.config.min_score_improvement:
                    diagnostics.append(
                        ProtectionRevisionDiagnostic(
                            route_id=route_id,
                            step_id=interaction.step_id,
                            fg_name=interaction.fg_name,
                            site_key=interaction.site_key,
                            rule_name=fragment.rule_name,
                            strategy=fragment.strategy,
                            candidate_score=score,
                            baseline_score=current_score,
                            reason="score_not_improved",
                            detail=(
                                f"candidate {score:.6f} <= "
                                f"baseline {current_score:.6f}"
                            ),
                        )
                    )
                    continue

                if not self._target_lowered(
                    interaction,
                    candidate_interactions,
                    candidate["transformed_step_id"],
                ):
                    diagnostics.append(
                        ProtectionRevisionDiagnostic(
                            route_id=route_id,
                            step_id=interaction.step_id,
                            fg_name=interaction.fg_name,
                            site_key=interaction.site_key,
                            rule_name=fragment.rule_name,
                            strategy=fragment.strategy,
                            reason="target_site_not_resolved",
                        )
                    )
                    continue

                if self.config.validate_route_cgr and not self._route_cgr_valid(
                    candidate["route"], route_id
                ):
                    diagnostics.append(
                        ProtectionRevisionDiagnostic(
                            route_id=route_id,
                            step_id=interaction.step_id,
                            fg_name=interaction.fg_name,
                            site_key=interaction.site_key,
                            rule_name=fragment.rule_name,
                            strategy=fragment.strategy,
                            reason="route_cgr_validation_failed",
                        )
                    )
                    continue

                fragment_priority = self._fragment_sort_key(fragment)
                if (
                    score > best["score"] + 1e-12
                    or (
                        abs(score - best["score"]) <= 1e-12
                        and fragment_priority < best["fragment_priority"]
                    )
                ):
                    best.update(
                        {
                            "route": candidate["route"],
                            "score": score,
                            "action": candidate["action"],
                            "route_metadata": candidate["route_metadata"],
                            "fragment_priority": fragment_priority,
                        }
                    )

        return best

    def _actionable_interactions(
        self,
        interactions: list[CompetingInteraction],
    ) -> list[CompetingInteraction]:
        severities = set(self.config.candidate_severities)
        return sorted(
            [
                interaction
                for interaction in interactions
                if interaction.severity in severities
            ],
            key=lambda item: (
                -self._SEVERITY_PENALTY.get(item.severity, 0.0),
                item.step_id,
                item.fg_name,
                item.fg_atoms,
            ),
        )

    def _fragment_sort_key(
        self,
        fragment: ProtectionFragment,
    ) -> tuple[int, str]:
        return (
            self._RULE_PRIORITY.get(fragment.rule_name, 10_000),
            fragment.rule_name,
        )

    def _site_diagnostics(
        self,
        interaction: CompetingInteraction,
        route_id: int | None,
    ) -> list[ProtectionRevisionDiagnostic]:
        diagnostics: list[ProtectionRevisionDiagnostic] = []
        if interaction.anchor_atom is None:
            diagnostics.append(
                ProtectionRevisionDiagnostic(
                    route_id=route_id,
                    step_id=interaction.step_id,
                    fg_name=interaction.fg_name,
                    site_key=interaction.site_key,
                    reason="missing_anchor_atom",
                )
            )
        if (
            interaction.modification_type
            not in self.config.supported_modification_types
        ):
            diagnostics.append(
                ProtectionRevisionDiagnostic(
                    route_id=route_id,
                    step_id=interaction.step_id,
                    fg_name=interaction.fg_name,
                    site_key=interaction.site_key,
                    reason="unsupported_modification_type",
                    detail=str(interaction.modification_type),
                )
            )
        return diagnostics

    def _build_candidate_route(
        self,
        route: dict[int, ReactionContainer],
        interaction: CompetingInteraction,
        fragment: ProtectionFragment,
        route_id: int | None,
        existing_metadata: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        diagnostics: list[ProtectionRevisionDiagnostic] = []
        reaction = route[interaction.step_id]
        reactants = list(reaction.reactants)
        products = list(reaction.products)
        anchor_atom = interaction.anchor_atom
        if anchor_atom is None:
            return {"route": None, "diagnostics": diagnostics}

        reactant_index = self._molecule_index_with_atom(reactants, anchor_atom)
        product_index = (
            interaction.product_index
            if interaction.product_index < len(products)
            and anchor_atom in products[interaction.product_index]._atoms
            else self._molecule_index_with_atom(products, anchor_atom)
        )
        if reactant_index is None or product_index is None:
            diagnostics.append(
                ProtectionRevisionDiagnostic(
                    route_id=route_id,
                    step_id=interaction.step_id,
                    fg_name=interaction.fg_name,
                    site_key=interaction.site_key,
                    rule_name=fragment.rule_name,
                    strategy=fragment.strategy,
                    reason="anchor_not_present_in_reactant_and_product",
                )
            )
            return {"route": None, "diagnostics": diagnostics}

        atom_mapping = self._fresh_fragment_atom_mapping(route, fragment)
        protected_reactant = self._attach_fragment(
            reactants[reactant_index],
            anchor_atom,
            fragment,
            atom_mapping,
        )
        protected_product = self._attach_fragment(
            products[product_index],
            anchor_atom,
            fragment,
            atom_mapping,
        )
        if protected_reactant is None or protected_product is None:
            diagnostics.append(
                ProtectionRevisionDiagnostic(
                    route_id=route_id,
                    step_id=interaction.step_id,
                    fg_name=interaction.fg_name,
                    site_key=interaction.site_key,
                    rule_name=fragment.rule_name,
                    strategy=fragment.strategy,
                    reason="chython_fragment_attachment_failed",
                    detail=fragment.rule_name,
                )
            )
            return {"route": None, "diagnostics": diagnostics}

        transformed_reactants = list(reactants)
        transformed_products = list(products)
        transformed_reactants[reactant_index] = protected_reactant
        transformed_products[product_index] = protected_product

        original_product = products[product_index]
        original_reactant = reactants[reactant_index]
        p_mol_reactants = self._p_mol_reactants(fragment, atom_mapping)
        protection_step = ReactionContainer(
            [original_reactant, *p_mol_reactants],
            [protected_reactant],
            meta={"protection_revision_role": "protection"},
        )
        transformed_step = ReactionContainer(
            transformed_reactants,
            transformed_products,
            list(reaction.reagents),
            dict(reaction.meta),
        )
        protected_steps: dict[int, ReactionContainer] = {
            interaction.step_id: transformed_step
        }
        deprotection_before_step = self._deprotection_before_step(route, interaction)
        for step_id in sorted(route):
            if step_id <= interaction.step_id:
                continue
            if (
                deprotection_before_step is not None
                and step_id >= deprotection_before_step
            ):
                break
            downstream_step = self._protect_downstream_step(
                route[step_id],
                anchor_atom,
                fragment,
                atom_mapping,
            )
            if downstream_step is None:
                diagnostics.append(
                    ProtectionRevisionDiagnostic(
                        route_id=route_id,
                        step_id=step_id,
                        fg_name=interaction.fg_name,
                        site_key=interaction.site_key,
                        rule_name=fragment.rule_name,
                        strategy=fragment.strategy,
                        reason="downstream_protection_failed",
                        detail=fragment.rule_name,
                    )
                )
                return {"route": None, "diagnostics": diagnostics}
            protected_steps[step_id] = downstream_step

        deprotection_pair = self._deprotection_pair(
            route,
            protected_steps,
            interaction.step_id,
            deprotection_before_step,
            anchor_atom,
            fragment,
            atom_mapping,
        )
        if deprotection_pair is None:
            diagnostics.append(
                ProtectionRevisionDiagnostic(
                    route_id=route_id,
                    step_id=interaction.step_id,
                    fg_name=interaction.fg_name,
                    site_key=interaction.site_key,
                    rule_name=fragment.rule_name,
                    strategy=fragment.strategy,
                    reason="deprotection_target_not_found",
                    detail=fragment.rule_name,
                )
            )
            return {"route": None, "diagnostics": diagnostics}
        protected_deprotection_reactant, deprotected_product = deprotection_pair
        deprotection_step = ReactionContainer(
            [protected_deprotection_reactant],
            [deprotected_product],
            meta={"protection_revision_role": "deprotection"},
        )

        new_route: dict[int, ReactionContainer] = {}
        route_metadata: dict[int, dict[str, Any]] = {}
        protection_step_id = -1
        transformed_step_id = -1
        deprotection_step_id = -1
        step_cursor = 0
        for step_id in sorted(route):
            if step_id == deprotection_before_step:
                deprotection_step_id = step_cursor
                new_route[deprotection_step_id] = deprotection_step
                route_metadata[deprotection_step_id] = self._metadata(
                    "deprotection",
                    interaction,
                    fragment,
                )
                step_cursor += 1

            if step_id != interaction.step_id:
                step = protected_steps.get(step_id, route[step_id])
                new_route[step_cursor] = step
                if step_id in protected_steps:
                    route_metadata[step_cursor] = self._with_revision_metadata(
                        existing_metadata.get(step_id, {}),
                        "protected_downstream_transformation",
                        interaction,
                        fragment,
                    )
                elif step_id in existing_metadata:
                    route_metadata[step_cursor] = existing_metadata[step_id]
                step_cursor += 1
                continue

            protection_step_id = step_cursor
            new_route[protection_step_id] = protection_step
            route_metadata[protection_step_id] = self._metadata(
                "protection",
                interaction,
                fragment,
            )
            step_cursor += 1

            transformed_step_id = step_cursor
            new_route[transformed_step_id] = transformed_step
            route_metadata[transformed_step_id] = self._with_revision_metadata(
                existing_metadata.get(step_id, {}),
                "protected_transformation",
                interaction,
                fragment,
            )
            step_cursor += 1

        if deprotection_before_step is None:
            deprotection_step_id = step_cursor
            new_route[deprotection_step_id] = deprotection_step
            route_metadata[deprotection_step_id] = self._metadata(
                "deprotection",
                interaction,
                fragment,
            )

        action = ProtectionAction(
            original_step_id=interaction.step_id,
            new_step_ids=(
                protection_step_id,
                transformed_step_id,
                deprotection_step_id,
            ),
            fg_name=interaction.fg_name,
            fg_atoms=interaction.fg_atoms,
            anchor_atom=anchor_atom,
            severity=interaction.severity,
            rule_name=fragment.rule_name,
            strategy=fragment.strategy,
            p_mol=fragment.p_mol,
            protection_class=fragment.reaction_class,
        )
        return {
            "route": new_route,
            "action": action,
            "route_metadata": route_metadata,
            "transformed_step_id": transformed_step_id,
            "diagnostics": diagnostics,
        }

    def _deprotection_before_step(
        self,
        route: dict[int, ReactionContainer],
        interaction: CompetingInteraction,
    ) -> int | None:
        site_atoms = set(interaction.fg_atoms)
        if interaction.anchor_atom is not None:
            site_atoms.add(interaction.anchor_atom)

        for step_id in sorted(route):
            if step_id <= interaction.step_id:
                continue
            try:
                center_atoms = get_reaction_center_atoms(route[step_id])
            except Exception:
                logger.exception("Could not identify downstream reaction center")
                return step_id
            if site_atoms & center_atoms:
                return step_id
        return None

    def _protect_downstream_step(
        self,
        reaction: ReactionContainer,
        anchor_atom: int,
        fragment: ProtectionFragment,
        atom_mapping: dict[int, int],
    ) -> ReactionContainer | None:
        changed = False
        reactants = list(reaction.reactants)
        products = list(reaction.products)

        for idx, molecule in enumerate(reactants):
            if anchor_atom not in molecule._atoms:
                continue
            protected = self._attach_fragment(
                molecule,
                anchor_atom,
                fragment,
                atom_mapping,
            )
            if protected is None:
                return None
            reactants[idx] = protected
            changed = True

        for idx, molecule in enumerate(products):
            if anchor_atom not in molecule._atoms:
                continue
            protected = self._attach_fragment(
                molecule,
                anchor_atom,
                fragment,
                atom_mapping,
            )
            if protected is None:
                return None
            products[idx] = protected
            changed = True

        if not changed:
            return reaction
        return ReactionContainer(
            reactants,
            products,
            list(reaction.reagents),
            dict(reaction.meta),
        )

    def _deprotection_pair(
        self,
        route: dict[int, ReactionContainer],
        protected_steps: dict[int, ReactionContainer],
        protected_from_step: int,
        deprotection_before_step: int | None,
        anchor_atom: int,
        fragment: ProtectionFragment,
        atom_mapping: dict[int, int],
    ) -> tuple[MoleculeContainer, MoleculeContainer] | None:
        if deprotection_before_step is not None:
            original_reactants = list(route[deprotection_before_step].reactants)
            reactant_index = self._molecule_index_with_atom(
                original_reactants,
                anchor_atom,
            )
            if reactant_index is None:
                return None
            deprotected_product = original_reactants[reactant_index]
            protected_reactant = self._attach_fragment(
                deprotected_product,
                anchor_atom,
                fragment,
                atom_mapping,
            )
            if protected_reactant is None:
                return None
            return protected_reactant, deprotected_product

        for step_id in sorted(route, reverse=True):
            if step_id < protected_from_step:
                continue
            original_products = list(route[step_id].products)
            product_index = self._molecule_index_with_atom(
                original_products,
                anchor_atom,
            )
            if product_index is None:
                continue
            protected_products = list(
                protected_steps.get(step_id, route[step_id]).products
            )
            if product_index >= len(protected_products):
                return None
            protected_reactant = protected_products[product_index]
            if anchor_atom not in protected_reactant._atoms:
                return None
            return protected_reactant, original_products[product_index]
        return None

    @staticmethod
    def _p_mol_reactants(
        fragment: ProtectionFragment,
        atom_mapping: dict[int, int],
    ) -> list[MoleculeContainer]:
        if fragment.p_mol == "None" or fragment.molecule is None:
            return []

        try:
            p_mol = smiles(fragment.p_mol)
            remap = ProtectionRouteReviser._p_mol_atom_mapping(
                p_mol,
                fragment,
                atom_mapping,
            )
            p_mol.remap(remap)
            reactants = list(p_mol.split())
            for reactant in reactants:
                reactant.meta["added_protecting_group"] = True
            return reactants
        except Exception:
            logger.exception("Failed to prepare p_mol protection reactants")
            return []

    @staticmethod
    def _p_mol_atom_mapping(
        p_mol: MoleculeContainer,
        fragment: ProtectionFragment,
        atom_mapping: dict[int, int],
    ) -> dict[int, int]:
        if fragment.molecule is None:
            return {}

        fragment_to_p_mol = next(iter(fragment.molecule.get_mapping(p_mol)), {})
        p_mol_to_route = {
            p_mol_atom: atom_mapping[fragment_atom]
            for fragment_atom, p_mol_atom in fragment_to_p_mol.items()
            if fragment_atom in atom_mapping
        }

        next_atom = max(atom_mapping.values(), default=0) + 1
        used_atoms = set(p_mol_to_route.values())
        for p_mol_atom in sorted(p_mol._atoms):
            if p_mol_atom in p_mol_to_route:
                continue
            while next_atom in used_atoms:
                next_atom += 1
            p_mol_to_route[p_mol_atom] = next_atom
            used_atoms.add(next_atom)
            next_atom += 1
        return p_mol_to_route

    def _attach_fragment(
        self,
        molecule: MoleculeContainer,
        anchor_atom: int,
        fragment: ProtectionFragment,
        atom_mapping: dict[int, int],
    ) -> MoleculeContainer | None:
        if anchor_atom not in molecule._atoms:
            return None
        if fragment.molecule is None:
            return None
        if fragment.strategy == "carbonyl_acetal":
            return self._attach_carbonyl_fragment(
                molecule,
                anchor_atom,
                fragment,
                atom_mapping,
            )
        if fragment.strategy != "single_anchor":
            return None
        if fragment.attachment_atom is None:
            return None
        if fragment.attachment_bond_order is None:
            return None

        updated = molecule.copy()
        try:
            for atom_id, atom in fragment.molecule.atoms():
                self._add_atom_copy(updated, atom, atom_mapping[atom_id])
            for atom1, atom2, bond in fragment.molecule.bonds():
                new_bond = bond.copy(full=True) if hasattr(bond, "copy") else bond
                updated.add_bond(
                    atom_mapping[atom1],
                    atom_mapping[atom2],
                    new_bond,
                )
            updated.add_bond(
                anchor_atom,
                atom_mapping[fragment.attachment_atom],
                fragment.attachment_bond_order,
            )
        except Exception:
            logger.exception("Failed to attach protection fragment")
            return None

        return validate_and_canonicalize(updated)

    def _attach_carbonyl_fragment(
        self,
        molecule: MoleculeContainer,
        anchor_atom: int,
        fragment: ProtectionFragment,
        atom_mapping: dict[int, int],
    ) -> MoleculeContainer | None:
        if fragment.molecule is None or len(fragment.attachment_atoms) != 2:
            return None
        carbonyl_oxygen = self._carbonyl_oxygen(molecule, anchor_atom)
        if carbonyl_oxygen is None:
            return None

        updated = molecule.copy()
        try:
            updated.delete_atom(carbonyl_oxygen)
            for atom_id, atom in fragment.molecule.atoms():
                self._add_atom_copy(updated, atom, atom_mapping[atom_id])
            for atom1, atom2, bond in fragment.molecule.bonds():
                new_bond = bond.copy(full=True) if hasattr(bond, "copy") else bond
                updated.add_bond(
                    atom_mapping[atom1],
                    atom_mapping[atom2],
                    new_bond,
                )
            for attachment_atom in fragment.attachment_atoms:
                updated.add_bond(
                    anchor_atom,
                    atom_mapping[attachment_atom],
                    1,
                )
        except Exception:
            logger.exception("Failed to attach carbonyl protection fragment")
            return None

        return validate_and_canonicalize(updated)

    @staticmethod
    def _add_atom_copy(
        molecule: MoleculeContainer,
        atom: Any,
        atom_id: int,
    ) -> None:
        new_atom = atom.copy(full=True)
        if hasattr(new_atom, "_parsed_mapping"):
            new_atom._parsed_mapping = None
        molecule.add_atom(
            new_atom,
            atom_id,
            charge=getattr(atom, "charge", getattr(atom, "_charge", 0)),
            is_radical=getattr(
                atom,
                "is_radical",
                getattr(atom, "_is_radical", False),
            ),
            xy=getattr(atom, "xy", (0.0, 0.0)),
            _skip_calculation=True,
        )

    @classmethod
    def _is_carboxyl_anchor(
        cls,
        molecule: MoleculeContainer,
        anchor_atom: int,
    ) -> bool:
        atom = molecule._atoms.get(anchor_atom)
        if atom is None or getattr(atom, "atomic_symbol", None) != "O":
            return False
        for neighbor, bond in molecule._bonds.get(anchor_atom, {}).items():
            if getattr(bond, "order", None) != 1:
                continue
            if (
                cls._carbonyl_oxygen(molecule, neighbor, exclude=anchor_atom)
                is not None
            ):
                return True
        return False

    @staticmethod
    def _carbonyl_oxygen(
        molecule: MoleculeContainer,
        anchor_atom: int,
        exclude: int | None = None,
    ) -> int | None:
        if anchor_atom not in molecule._atoms:
            return None
        for neighbor, bond in molecule._bonds.get(anchor_atom, {}).items():
            if neighbor == exclude:
                continue
            atom = molecule._atoms.get(neighbor)
            if (
                atom is not None
                and getattr(atom, "atomic_symbol", None) == "O"
                and getattr(bond, "order", None) == 2
            ):
                return neighbor
        return None

    def _fresh_fragment_atom_mapping(
        self,
        route: dict[int, ReactionContainer],
        fragment: ProtectionFragment,
    ) -> dict[int, int]:
        if fragment.molecule is None:
            return {}
        next_atom = self._max_atom_number(route) + 1
        mapping: dict[int, int] = {}
        for atom_id in sorted(fragment.molecule._atoms):
            mapping[atom_id] = next_atom
            next_atom += 1
        return mapping

    @staticmethod
    def _max_atom_number(route: dict[int, ReactionContainer]) -> int:
        max_atom = 0
        for reaction in route.values():
            for molecule in (
                list(reaction.reactants)
                + list(reaction.products)
                + list(reaction.reagents)
            ):
                if molecule._atoms:
                    max_atom = max(max_atom, max(molecule._atoms))
        return max_atom

    @staticmethod
    def _molecule_index_with_atom(
        molecules: list[MoleculeContainer],
        atom_id: int,
    ) -> int | None:
        matches = [
            idx for idx, molecule in enumerate(molecules) if atom_id in molecule._atoms
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _target_lowered(
        self,
        original: CompetingInteraction,
        candidate_interactions: list[CompetingInteraction],
        transformed_step_id: int,
    ) -> bool:
        original_penalty = self._SEVERITY_PENALTY.get(original.severity, 0.0)
        candidate_penalty = 0.0
        original_atoms = set(original.fg_atoms)
        for interaction in candidate_interactions:
            if interaction.step_id != transformed_step_id:
                continue
            same_anchor = (
                original.anchor_atom is not None
                and interaction.anchor_atom == original.anchor_atom
            )
            same_atoms = bool(original_atoms & set(interaction.fg_atoms))
            if same_anchor or same_atoms:
                candidate_penalty = max(
                    candidate_penalty,
                    self._SEVERITY_PENALTY.get(interaction.severity, 0.0),
                )
        return candidate_penalty < original_penalty

    def _route_cgr_valid(
        self,
        route: dict[int, ReactionContainer],
        route_id: int | None,
    ) -> bool:
        validation_route_id = 0 if route_id is None else route_id
        try:
            result = compose_route_cgr(
                {validation_route_id: route},
                validation_route_id,
                preserve_transient_bonds=self.config.preserve_transient_bonds,
            )
        except Exception:
            logger.exception("Route-CGR validation failed")
            return False
        return result is not None

    @classmethod
    def _metadata(
        cls,
        role: str,
        interaction: CompetingInteraction,
        fragment: ProtectionFragment,
    ) -> dict[str, Any]:
        return cls._with_revision_metadata({}, role, interaction, fragment)

    @staticmethod
    def _with_revision_metadata(
        base: dict[str, Any],
        role: str,
        interaction: CompetingInteraction,
        fragment: ProtectionFragment,
    ) -> dict[str, Any]:
        merged = dict(base)
        merged["protection_revision"] = {
            "role": role,
            "original_step_id": interaction.step_id,
            "fg_name": interaction.fg_name,
            "fg_atoms": list(interaction.fg_atoms),
            "anchor_atom": interaction.anchor_atom,
            "severity": interaction.severity,
            "rule_name": fragment.rule_name,
            "strategy": fragment.strategy,
            "p_mol": fragment.p_mol,
            "protection_class": fragment.reaction_class,
            "conditions": {
                "h2o": fragment.h2o,
                "bases": fragment.bases,
                "nucleophiles": fragment.nucleophiles,
                "electrophiles": fragment.electrophiles,
                "reduction": fragment.reduction,
                "oxidation": fragment.oxidation,
            },
        }
        return merged


__all__ = [
    "ProtectionAction",
    "ProtectionFragment",
    "ProtectionFragmentCatalog",
    "ProtectionRouteReviser",
    "ProtectionRevisionConfig",
    "ProtectionRevisionDiagnostic",
    "RevisedRoute",
]
