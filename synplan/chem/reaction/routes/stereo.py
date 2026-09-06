"""Conservative, opt-in reconstruction of stereo inherited from purchased leaves.

Only unchanged mapped carbon tetrahedra and ordinary double bonds are supported.
The result establishes a structural contract, never experimental selectivity.
Search, canonicalisation and the catalogue's connectivity lookup are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

from chython import smiles
from chython.containers import MoleculeContainer, ReactionContainer

from synplan.chem.building_blocks import BuildingBlockCatalogue, molecule_to_inchikey
from synplan.chem.reaction.routes.route import Route, Step


@dataclass(frozen=True)
class StereoRequirement:
    """A mapped local orientation, linked to its original target atoms."""

    target_atoms: tuple[int, ...]
    kind: str
    atoms: tuple[int, ...]
    environment: tuple[int, ...]
    sign: bool

    def remap(self, mapping: Mapping[int, int]) -> StereoRequirement:
        return StereoRequirement(
            self.target_atoms,
            self.kind,
            tuple(mapping[n] for n in self.atoms),
            tuple(mapping[n] for n in self.environment),
            self.sign,
        )


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
    assigned_stereo: str = "inferred_during_reconstruction"
    chemistry_scope: str = (
        "graph inheritance only; conditions and epimerization unverified"
    )

    @property
    def supported(self) -> bool:
        return self.route is not None and not self.issues

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


class _Unresolved(ValueError):
    def __init__(self, reason: str, detail: str, **context):
        self.reason = reason
        self.context = context
        super().__init__(detail)


def _requirements(mol: MoleculeContainer) -> tuple[StereoRequirement, ...]:
    out = []
    for n, atom in mol.atoms():
        if atom.stereo is None:
            continue
        if n not in mol.stereogenic_tetrahedrons or atom.atomic_number != 6:
            raise _Unresolved(
                "unsupported_stereo_type", f"atom {n}: only carbon tetrahedra supported"
            )
        env = tuple(sorted(mol.stereogenic_tetrahedrons[n]))
        out.append(
            StereoRequirement(
                (n,), "tetrahedron", (n,), env, mol._translate_tetrahedron_sign(n, env)
            )
        )
    for n, m, bond in mol.bonds():
        if bond.stereo is None:
            continue
        if int(bond) != 2 or (
            (n, m) not in mol.stereogenic_cis_trans
            and (m, n) not in mol.stereogenic_cis_trans
        ):
            raise _Unresolved(
                "unsupported_stereo_type",
                f"bond {n}-{m}: cumulene or unsupported stereo",
            )
        env = (
            next(k for k in mol._bonds[n] if k != m),
            next(k for k in mol._bonds[m] if k != n),
        )
        out.append(
            StereoRequirement(
                (n, m),
                "double_bond",
                (n, m),
                env,
                mol._translate_cis_trans_sign(n, m, *env),
            )
        )
    return tuple(out)


def _sign(mol: MoleculeContainer, req: StereoRequirement) -> bool | None:
    if req.kind == "tetrahedron":
        if mol.atom(req.atoms[0]).stereo is None:
            return None
        return mol._translate_tetrahedron_sign(req.atoms[0], req.environment)
    n, m = req.atoms
    if mol._bonds[n][m].stereo is None:
        return None
    return mol._translate_cis_trans_sign(n, m, *req.environment)


def _assign(mol: MoleculeContainer, req: StereoRequirement) -> None:
    existing = _sign(mol, req)
    if existing is not None:
        if existing != req.sign:
            raise _Unresolved(
                "configuration_contradicted",
                f"opposite mapped configuration at {req.atoms}",
            )
        return
    try:
        if req.kind == "tetrahedron":
            mol.add_atom_stereo(req.atoms[0], req.environment, req.sign)
        else:
            n, m = req.atoms
            n1, n2 = req.environment
            mol.add_cis_trans_stereo(n, m, n1, n2, req.sign)
    except (KeyError, ValueError) as exc:
        raise _Unresolved(
            "requires_stereo_forming_step",
            f"configuration not stereogenic at {req.atoms}: {exc}",
        ) from exc


def _connectivity(mol: MoleculeContainer) -> str:
    copy = mol.copy()
    copy.clean_stereo()
    return str(copy)


def _mappings(
    source: MoleculeContainer, dest: MoleculeContainer, cap: int
) -> list[dict[int, int]]:
    if _connectivity(source) != _connectivity(dest):
        raise _Unresolved(
            "mapping_or_representation_unresolved",
            "inconsistent adjacent molecule structures",
        )
    a, b = source.copy(), dest.copy()
    a.clean_stereo()
    b.clean_stereo()
    mappings = list(islice(a.get_mapping(b, automorphism_filter=False), cap + 1))
    if len(mappings) > cap:
        raise _Unresolved(
            "mapping_or_representation_unresolved",
            f"isomorphism enumeration exceeded {cap}; no arbitrary mapping selected",
        )
    if not mappings:
        raise _Unresolved(
            "mapping_or_representation_unresolved", "no complete atom correspondence"
        )
    return mappings


def _orientation_key(
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
        else:
            n, m = atoms = tuple(sorted(req.atoms))
            start, end = req.atoms
            n1, n2 = req.environment
            native = mol._translate_cis_trans_sign(start, end, n1, n2, req.sign)
            env = (
                min(k for k in mol._bonds[n] if k != m),
                min(k for k in mol._bonds[m] if k != n),
            )
            sign = mol._translate_cis_trans_sign(n, m, *env, native)
        out.append((req.target_atoms if by_requirement else (), req.kind, atoms, sign))
    return tuple(sorted(set(out)))


def _align(
    source: MoleculeContainer,
    dest: MoleculeContainer,
    reqs: tuple[StereoRequirement, ...],
    cap: int,
) -> tuple[tuple[StereoRequirement, ...], dict[int, int], list[dict[int, int]]]:
    choices = {}
    mappings = _mappings(source, dest, cap)
    for mapping in mappings:
        translated = tuple(r.remap(mapping) for r in reqs)
        key = _orientation_key(dest, translated)
        choices.setdefault(key, (translated, mapping))
    if len(choices) != 1:
        raise _Unresolved(
            "mapping_or_representation_unresolved",
            f"{len(choices)} stereo-distinct atom correspondences",
        )
    required, mapping = next(iter(choices.values()))
    return required, mapping, mappings


def _atom_identity(atom) -> tuple:
    return atom.atomic_number, atom.isotope, atom.charge, atom.is_radical


def _local_environment(mol: MoleculeContainer, n: int) -> tuple:
    atom = mol.atom(n)
    return (
        _atom_identity(atom),
        atom.implicit_hydrogens,
        tuple(sorted((k, int(b)) for k, b in mol._bonds[n].items())),
    )


def _transfer(
    product: MoleculeContainer,
    reactants: tuple[MoleculeContainer, ...],
    req: StereoRequirement,
) -> int:
    candidates = [
        i for i, mol in enumerate(reactants) if all(n in mol._atoms for n in req.atoms)
    ]
    if len(candidates) != 1:
        if req.kind == "double_bond" and all(
            any(n in mol._atoms for mol in reactants) for n in req.atoms
        ):
            raise _Unresolved(
                "requires_stereo_forming_step",
                f"double bond {req.atoms} is assembled from separate reactants",
            )
        raise _Unresolved(
            "mapping_or_representation_unresolved",
            f"required atoms {req.atoms} have no unique precursor mapping",
        )
    index = candidates[0]
    precursor = reactants[index]
    if any(
        _local_environment(product, n) != _local_environment(precursor, n)
        for n in req.atoms
    ):
        reason = "requires_explicit_resolution_or_inversion_strategy"
        if (
            req.kind == "tetrahedron"
            and req.atoms[0] not in precursor.stereogenic_tetrahedrons
        ) or (
            req.kind == "double_bond"
            and int(precursor._bonds.get(req.atoms[0], {}).get(req.atoms[1], 0)) != 2
        ):
            reason = "requires_stereo_forming_step"
        raise _Unresolved(
            reason,
            f"mapped local environment changes at {req.atoms}; retention is unsupported",
        )
    _assign(precursor, req)
    return index


def _validate_mapping(reaction: ReactionContainer) -> None:
    sides = []
    for side in (reaction.reactants, reaction.products):
        atoms = {}
        for mol in side:
            for n, atom in mol.atoms():
                if n in atoms:
                    raise _Unresolved(
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
            raise _Unresolved(
                "mapping_or_representation_unresolved",
                f"element/isotope changes at mapped atom {n}",
            )


def _stock_match(
    mol: MoleculeContainer,
    reqs: tuple[StereoRequirement, ...],
    catalogue: BuildingBlockCatalogue,
    cap: int,
) -> tuple[MoleculeContainer, dict[str, Any]]:
    key = molecule_to_inchikey(mol)
    constraints = (*reqs, *_requirements(mol))
    rejected = []
    for record in sorted(
        catalogue.get(key[:14], ()), key=lambda r: (r.inchikey, r.smiles)
    ):
        candidate = smiles(record.smiles)
        if molecule_to_inchikey(candidate) != record.inchikey:
            rejected.append(
                {"inchikey": record.inchikey, "reason": "record_identity_mismatch"}
            )
            continue
        if _connectivity(candidate) != _connectivity(mol):
            rejected.append(
                {"inchikey": record.inchikey, "reason": "connectivity_bucket_only"}
            )
            continue
        compatible = []
        for mapping in _mappings(mol, candidate, cap):
            mapped = tuple(r.remap(mapping) for r in constraints)
            if all(_sign(candidate, r) == r.sign for r in mapped):
                compatible.append(mapping)
        if compatible:
            # Extra configurations are retained from the selected actual record.
            versions = {}
            for mapping in compatible:
                selected = candidate.copy()
                selected.remap({v: k for k, v in mapping.items()})
                versions.setdefault(format(selected, "m"), selected)
            if (
                len(
                    {_orientation_key(mol, _requirements(v)) for v in versions.values()}
                )
                > 1
            ):
                raise _Unresolved(
                    "mapping_or_representation_unresolved",
                    "stock record has stereo-distinct alignments",
                )
            selected = next(iter(versions.values()))
            return selected, {
                "inchikey": record.inchikey,
                "smiles": record.smiles,
                "vendors": dict(record.vendors),
                "price": min(record.vendors.values()) if record.vendors else None,
                "record_to_route_atom_map": {v: k for k, v in compatible[0].items()},
                "compatible_mapping_count": len(compatible),
                "rejected_candidates": rejected,
            }
        rejected.append(
            {
                "inchikey": record.inchikey,
                "reason": "wrong_or_unspecified_configuration",
            }
        )
    raise _Unresolved(
        "required_stock_unavailable",
        f"no explicit compatible record in {key[:14]}",
        connectivity_prefix=key[:14],
        rejected_candidates=rejected,
    )


def _chemistry_flags(
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
                for j, b in product._bonds[k].items()
            )
            for k in product._bonds[n]
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

    def issue(exc, step=None, leaf=None, req=None):
        result.issues.append(
            {
                "reason": exc.reason,
                "detail": str(exc),
                "step": step,
                "leaf": leaf,
                "target_atoms": list(req.target_atoms) if req else None,
                **exc.context,
            }
        )

    try:
        required = _requirements(original_target)
        if not required:
            raise _Unresolved(
                "mapping_or_representation_unresolved",
                "original target has no supported explicit stereo requirement",
            )
        required, mapping, equivalent_mappings = _align(
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
    except _Unresolved as exc:
        issue(exc)
        return result

    for index in reversed(range(len(working.steps))):
        step = working.steps[index]
        reqs = requirements.get(id(step.product), ())
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
            "chemistry_flags": _chemistry_flags(step.product, reqs),
        }
        result.ledger.append(entry)
        try:
            if not mapping_sources.get(index):
                raise _Unresolved(
                    "mapping_or_representation_unresolved",
                    "no provenance for reaction atom correspondence",
                )
            _validate_mapping(step.reaction)
            precursor_atoms = {n for mol in step.reaction.reactants for n in mol}
            if missing := set(step.product) - precursor_atoms:
                raise _Unresolved(
                    "mapping_or_representation_unresolved",
                    f"product atoms {sorted(missing)} have no mapped reactant source",
                )
            for req in reqs:
                try:
                    _assign(step.product, req)
                    slot = _transfer(step.product, step.reaction.reactants, req)
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
                except _Unresolved as exc:
                    issue(exc, index, req=req)
        except _Unresolved as exc:
            issue(exc, index)

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
            selected, record = _stock_match(leaf, reqs, catalogue, max_mappings)
            assignments[id(leaf)] = selected
            entry["selected_stock"] = record
        except _Unresolved as exc:
            issue(exc, leaf=index)
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
            for req in _requirements(precursor):
                if all(n in product._atoms for n in req.atoms) and all(
                    _local_environment(product, n) == _local_environment(precursor, n)
                    for n in req.atoms
                ):
                    # Non-required stock stereo can disappear by symmetry;
                    # target-linked obligations are checked immediately below.
                    with suppress(_Unresolved):
                        _assign(product, req)
        for req in requirements.get(id(step.product), ()):
            if _sign(product, req) != req.sign:
                issue(
                    _Unresolved(
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
