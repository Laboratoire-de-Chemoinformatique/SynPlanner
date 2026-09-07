"""Conservative reconstruction of stereo inherited from purchased leaves.

Unchanged mapped carbon tetrahedra, ordinary double bonds and allenes are supported.
The result establishes a structural contract, never experimental selectivity.
Search invokes this audit before accepting a stereo-bearing terminal route.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, ClassVar

from chython.containers import MoleculeContainer, ReactionContainer

from synplan.chem.building_blocks import BuildingBlockCatalogue, molecule_to_inchikey
from synplan.chem.building_blocks.stereo import record_molecule
from synplan.chem.mapping import MappingBudgetExceeded, bounded_mappings
from synplan.chem.reaction.routes.route import Route, Step
from synplan.chem.stereo import (
    REVIEWABLE_STEREO_REASONS,
    StereoRequirement,
    UnresolvedStereo,
    assign_stereo,
    atom_owners,
    has_stereo_groups,
    local_environment,
    stereo_requirements,
    stereo_sign,
    transfer_stereo,
)
from synplan.chem.stereo_evidence import chemistry_assessment


@dataclass
class StereoAudit:
    """The original prediction is untouched; ``route`` is inferred reconstruction.

    ``ledger`` and ``issues`` contain JSON-compatible dictionaries. Step numbers
    are zero-based indices into the input Route's forward-ordered steps.
    """

    original_target: str
    route: Route | None = None
    ledger: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    assigned_stereo: ClassVar[str] = "inferred_during_reconstruction"
    chemistry_scope: ClassVar[str] = (
        "graph inheritance only; conditions and epimerization unverified"
    )

    @property
    def supported(self) -> bool:
        return self.route is not None and not self.issues

    def add_issue(self, exc, step=None, leaf=None, req=None):
        self.issues.append(
            {
                "reason": exc.reason,
                "detail": str(exc),
                "step": step,
                "leaf": leaf,
                "target_atoms": list(req.target_atoms) if req else None,
                **exc.context,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_target": self.original_target,
            "supported": self.supported,
            "status": "inherited_from_stock"
            if self.supported
            else "unresolved_proposal",
            "assigned_stereo": self.assigned_stereo,
            "chemistry_scope": self.chemistry_scope,
            "first_responsible": self.issues[0] if self.issues else None,
            "issues": self.issues,
            "ledger": self.ledger,
            # The ordinary exporter normalises molecules; this sidecar explicitly
            # preserves every assigned molecule and full mapped reaction instead.
            "reconstructed_steps": [format(s.reaction, "m") for s in self.route.steps]
            if self.route is not None
            else None,
        }


def connectivity_key(mol: MoleculeContainer) -> str:
    copy = mol.copy()
    copy.clean_stereo()
    return str(copy)


def molecule_mappings(
    source: MoleculeContainer, dest: MoleculeContainer, cap: int
) -> list[dict[int, int]]:
    a, b = source.copy(), dest.copy()
    a.clean_stereo()
    b.clean_stereo()
    if str(a) != str(b):
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved",
            "inconsistent adjacent molecule structures",
        )
    try:
        mappings = list(islice(bounded_mappings(a, b), cap + 1))
    except MappingBudgetExceeded as error:
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved", str(error)
        ) from error
    if len(mappings) > cap:
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved",
            f"isomorphism enumeration exceeded {cap}; no arbitrary mapping selected",
        )
    if not mappings:
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved", "no complete atom correspondence"
        )
    return mappings


def orientation_key(
    mol: MoleculeContainer,
    reqs: tuple[StereoRequirement, ...],
    *,
    by_requirement: bool = False,
) -> tuple:
    """Keep stereo-distinct atom correspondences even for symmetric molecules."""
    out = []
    for req in reqs:
        if req.kind == "tetrahedron":
            n = req.atoms[0]
            native = mol._translate_tetrahedron_sign(n, req.environment, req.sign)
            sign = mol._translate_tetrahedron_sign(
                n, tuple(sorted(mol.stereogenic_tetrahedrons[n])), native
            )
            atoms = req.atoms
        elif req.kind == "allene":
            center = req.atoms[0]
            n1, n2 = req.environment
            native = mol._translate_allene_sign(center, n1, n2, req.sign)
            env = mol.stereogenic_allenes[center][:2]
            sign = mol._translate_allene_sign(center, *env, native)
            atoms = (center,)
        else:
            n, m = atoms = tuple(sorted(req.atoms))
            start, end = req.atoms
            n1, n2 = req.environment
            native = mol._translate_cis_trans_sign(start, end, n1, n2, req.sign)
            env = (
                min(k for k in mol.neighbor_numbers(n) if k != m),
                min(k for k in mol.neighbor_numbers(m) if k != n),
            )
            sign = mol._translate_cis_trans_sign(n, m, *env, native)
        out.append((req.target_atoms if by_requirement else (), req.kind, atoms, sign))
    return frozenset(out)


def align_stereo(
    source: MoleculeContainer,
    dest: MoleculeContainer,
    reqs: tuple[StereoRequirement, ...],
    cap: int,
) -> tuple[tuple[StereoRequirement, ...], dict[int, int], list[dict[int, int]]]:
    choices, equivalent = {}, []
    for mapping in molecule_mappings(source, dest, cap):
        translated = tuple(r.remap(mapping) for r in reqs)
        if any(stereo_sign(dest, r) not in (None, r.sign) for r in translated):
            continue
        key = orientation_key(dest, translated)
        choices.setdefault(key, (translated, mapping))
        equivalent.append(mapping)
    if not choices:
        raise UnresolvedStereo(
            "configuration_contradicted",
            "opposite mapped configuration in adjacent molecule records",
        )
    if len(choices) != 1:
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved",
            f"{len(choices)} stereo-distinct atom correspondences",
        )
    required, mapping = next(iter(choices.values()))
    return required, mapping, equivalent


def validate_mapping(reaction: ReactionContainer) -> None:
    sides = []
    for side in (reaction.reactants, reaction.products):
        atoms = {}
        for mol in side:
            for n, atom in mol.atoms():
                if n in atoms:
                    raise UnresolvedStereo(
                        "mapping_or_representation_unresolved",
                        f"duplicate mapped atom {n} on reaction side",
                    )
                atoms[n] = atom
        sides.append(atoms)
    for n in sides[0].keys() & sides[1].keys():
        if (sides[0][n].atomic_number, sides[0][n].isotope) != (
            sides[1][n].atomic_number,
            sides[1][n].isotope,
        ):
            raise UnresolvedStereo(
                "mapping_or_representation_unresolved",
                f"element/isotope changes at mapped atom {n}",
            )


def match_stereo_stock(
    mol: MoleculeContainer,
    reqs: tuple[StereoRequirement, ...],
    catalogue: BuildingBlockCatalogue,
    cap: int,
) -> tuple[MoleculeContainer, dict[str, Any]]:
    key = molecule_to_inchikey(mol)
    constraints = (*reqs, *stereo_requirements(mol))
    rejected = []
    bucket = catalogue.get(key[:14], ())
    if selected := mol.meta.get("selected_stock"):
        bucket = tuple(
            r
            for r in bucket
            if r.inchikey == selected["inchikey"] and r.smiles == selected["smiles"]
        )
    if len(bucket) > cap:
        raise UnresolvedStereo(
            "stock_assessment_incomplete",
            f"stock bucket exceeds {cap} explicit records",
        )
    if has_stereo_groups(mol):
        raise UnresolvedStereo(
            "relative_or_mixture_stereo", "material/group semantics require review"
        )
    connectivity = connectivity_key(mol)
    for record in bucket:
        try:
            candidate = record_molecule(record.smiles, record.inchikey)
        except ValueError as error:
            rejected.append(
                {
                    "inchikey": record.inchikey,
                    "reason": "invalid_stock_record",
                    "detail": str(error),
                }
            )
            continue
        if has_stereo_groups(candidate):
            rejected.append(
                {"inchikey": record.inchikey, "reason": "relative_or_mixture_stock"}
            )
            continue
        if connectivity_key(candidate) != connectivity:
            rejected.append(
                {"inchikey": record.inchikey, "reason": "connectivity_bucket_only"}
            )
            continue
        compatible = []
        for mapping in molecule_mappings(mol, candidate, cap):
            mapped = tuple(r.remap(mapping) for r in constraints)
            if all(stereo_sign(candidate, r) == r.sign for r in mapped):
                compatible.append(mapping)
        if compatible:
            # Extra configurations are retained from the selected actual record.
            candidate_reqs = stereo_requirements(candidate)
            orientations = set()
            for mapping in compatible:
                inverse = {v: k for k, v in mapping.items()}
                orientations.add(
                    orientation_key(
                        mol, tuple(r.remap(inverse) for r in candidate_reqs)
                    )
                )
            if len(orientations) > 1:
                raise UnresolvedStereo(
                    "mapping_or_representation_unresolved",
                    "stock record has stereo-distinct alignments",
                )
            selected = candidate.copy()
            selected.remap({v: k for k, v in compatible[0].items()})
            record_data = {
                "inchikey": record.inchikey,
                "smiles": record.smiles,
                "vendors": dict(record.vendors),
                "price": min(record.vendors.values()) if record.vendors else None,
                "record_to_route_atom_map": {v: k for k, v in compatible[0].items()},
                "compatible_mapping_count": len(compatible),
                "rejected_candidates": rejected,
            }
            selected.meta["selected_stock"] = record_data
            return selected, record_data
        rejected.append(
            {
                "inchikey": record.inchikey,
                "reason": "wrong_or_unspecified_configuration",
            }
        )
    raise UnresolvedStereo(
        "required_stock_unavailable",
        f"no explicit compatible record in {key[:14]}",
        connectivity_prefix=key[:14],
        rejected_candidates=rejected,
    )


def chemistry_flags(
    product: MoleculeContainer, reqs: tuple[StereoRequirement, ...]
) -> list[dict[str, Any]]:
    """Structural review flags, with no claim about the unreported conditions."""
    flags = []
    for req in reqs:
        if req.kind != "tetrahedron":
            continue
        n = req.atoms[0]
        if not product.atom(n).implicit_hydrogens:
            continue
        if any(
            product.atom(k).atomic_number == 6
            and any(
                int(b) == 2 and product.atom(j).atomic_number == 8
                for j, b in product.bond_items(k)
            )
            for k in product.neighbor_numbers(n)
        ):
            flags.append(
                {
                    "target_atoms": req.target_atoms,
                    "atom": n,
                    "flag": "alpha_carbonyl_stereocentre",
                    "assessment": "review conditions for epimerization; graph inheritance alone is insufficient",
                }
            )
    return flags


def audit_stereo_inheritance(
    route: Route,
    original_target: MoleculeContainer,
    catalogue: BuildingBlockCatalogue,
    *,
    mapping_sources: Mapping[int, str],
    max_mappings: int = 256,
    allow_unconstrained_target: bool = False,
) -> StereoAudit:
    """Reconstruct a detached Route without modifying input molecules or provenance.

    ``mapping_sources`` supplies the provenance of *the current reaction maps*
    for every zero-based step (e.g. patent ID, or planner export path/rule ID).
    It must not be supplied for atom numbers invented while parsing unmapped
    SMILES. Missing evidence abstains. Maps are checked for conservation and
    local environments; source provenance is not proof of experimental mapping.

    All leaves require catalogue records, including achiral/small molecules.
    No incomplete reconstruction is returned as a supported route. Callers must
    preserve the original source serialization before using lossy route readers.
    """
    if max_mappings < 1:
        raise ValueError("max_mappings must be positive")
    original_target = original_target.copy()
    working = deepcopy(route)
    result = StereoAudit(format(original_target, "m"))
    requirements: dict[int, tuple[StereoRequirement, ...]] = {}
    assignments = {}
    reviewed_steps = set()

    try:
        required = stereo_requirements(original_target)
        if has_stereo_groups(original_target):
            raise UnresolvedStereo(
                "relative_or_mixture_stereo",
                "target material/group semantics require review",
            )
        if not required and not allow_unconstrained_target:
            raise UnresolvedStereo(
                "mapping_or_representation_unresolved",
                "original target has no supported explicit stereo requirement",
            )
        required, mapping, equivalent_mappings = align_stereo(
            original_target, working.target, required, max_mappings
        )
        requirements[id(working.target)] = required
        result.ledger.append(
            {
                "stage": "target",
                "mapping": mapping,
                "equivalent_mappings": equivalent_mappings,
                "requirements": [r.__dict__ for r in required],
            }
        )
    except UnresolvedStereo as exc:
        result.add_issue(exc)
        return result

    for index in reversed(range(len(working.steps))):
        step = working.steps[index]
        if chemistry_assessment(step.reaction) == "accepted":
            reviewed_steps.add(index)
        reqs = requirements.get(id(step.product), ())
        try:
            reqs = tuple(dict.fromkeys((*reqs, *stereo_requirements(step.product))))
            requirements[id(step.product)] = reqs
        except UnresolvedStereo as exc:
            result.add_issue(exc, index)
        entry = {
            "stage": "step",
            "step": index,
            "mapping_source": mapping_sources.get(index),
            "source_reaction": step.reaction.meta.get("stereo_source_reaction"),
            "source_atom_rebase": step.reaction.meta.get("stereo_atom_rebase"),
            "reaction_before": format(step.reaction, "m"),
            "requirements": [r.__dict__ for r in reqs],
            "transfers": [],
            "conditions": "not_evaluated",
            "chemistry_flags": chemistry_flags(step.product, reqs),
        }
        result.ledger.append(entry)
        owners = atom_owners(step.reaction.reactants)
        try:
            if not mapping_sources.get(index):
                raise UnresolvedStereo(
                    "mapping_or_representation_unresolved",
                    "no provenance for reaction atom correspondence",
                )
            validate_mapping(step.reaction)
            precursor_atoms = {n for mol in step.reaction.reactants for n in mol}
            if missing := set(step.product) - precursor_atoms:
                raise UnresolvedStereo(
                    "mapping_or_representation_unresolved",
                    f"product atoms {sorted(missing)} have no mapped reactant source",
                )
            for req in reqs:
                try:
                    assign_stereo(step.product, req)
                    slot = transfer_stereo(
                        step.product, step.reaction.reactants, req, owners=owners
                    )
                    child = step.reaction.reactants[slot]
                    requirements[id(child)] = (*requirements.get(id(child), ()), req)
                    entry["transfers"].append(
                        {
                            "target_atoms": req.target_atoms,
                            "reactant_slot": slot,
                            "atoms": req.atoms,
                            "configuration": "mapped_orientation_retained",
                        }
                    )
                except UnresolvedStereo as exc:
                    if (
                        index in reviewed_steps
                        and exc.reason in REVIEWABLE_STEREO_REASONS
                    ):
                        entry["transfers"].append(
                            {
                                "target_atoms": req.target_atoms,
                                "configuration": "scoped_chemist_assessment",
                            }
                        )
                    else:
                        result.add_issue(exc, index, req=req)
        except UnresolvedStereo as exc:
            result.add_issue(exc, index)

    for index, leaf in enumerate(working.leaves()):
        reqs = requirements.get(id(leaf), ())
        entry = {
            "stage": "leaf",
            "leaf": index,
            "required_smiles": format(leaf, "m"),
            "requirements": [r.__dict__ for r in reqs],
        }
        result.ledger.append(entry)
        try:
            selected, record = match_stereo_stock(leaf, reqs, catalogue, max_mappings)
            assignments[id(leaf)] = selected
            entry["selected_stock"] = record
        except UnresolvedStereo as exc:
            result.add_issue(exc, leaf=index)
            result.issues[-1]["target_atoms"] = sorted(
                {n for r in reqs for n in r.target_atoms}
            )

    if result.issues:
        return result

    # Forward reconstruction uses the very same assigned intermediate objects.
    # Reapply unchanged stereo from actual stock, then check every required sign.
    rebuilt = []
    for index, step in enumerate(working.steps):
        reactants = tuple(assignments[id(m)] for m in step.reaction.reactants)
        product = step.product.copy()
        product.clean_stereo()
        for precursor in reactants:
            for req in stereo_requirements(precursor):
                if all(product.has_atom(n) for n in req.atoms) and all(
                    local_environment(product, n) == local_environment(precursor, n)
                    for n in req.atoms
                ):
                    # Non-required stock stereo can disappear by symmetry;
                    # target-linked obligations are checked immediately below.
                    with suppress(UnresolvedStereo):
                        assign_stereo(product, req)
        for req in requirements.get(id(step.product), ()):
            if index in reviewed_steps:
                assign_stereo(product, req)
            if stereo_sign(product, req) != req.sign:
                result.add_issue(
                    UnresolvedStereo(
                        "configuration_contradicted",
                        "forward replay did not reproduce assigned configuration",
                    ),
                    index,
                    req=req,
                )
        assignments[id(step.product)] = product
        products = tuple(
            product if m is step.product else m.copy() for m in step.reaction.products
        )
        reaction = ReactionContainer(
            reactants,
            products,
            [m.copy() for m in step.reaction.reagents],
            meta=deepcopy(step.reaction.meta),
            name=step.reaction.name,
        )
        rebuilt.append(Step(reaction, product, step.origin, step.conditions))
        result.ledger.append(
            {
                "stage": "forward",
                "step": index,
                "reaction_after": format(reaction, "m"),
                "required_configurations_checked": len(
                    requirements.get(id(step.product), ())
                ),
            }
        )
    if not result.issues:
        result.route = Route(tuple(rebuilt), provenance=route.provenance)
    return result


__all__ = ["StereoAudit", "StereoRequirement", "audit_stereo_inheritance"]
