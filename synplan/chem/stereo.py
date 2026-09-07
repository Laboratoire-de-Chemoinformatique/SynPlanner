"""Chython orientation requirements shared by extraction, stock, search and routes.

Parity is relative to mapped neighbors, never a comparison of printed CIP labels.
No stereoisomer enumeration or experimental selectivity inference is performed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256

from chython.containers import MoleculeContainer

UNASSESSED_STEREO_REASONS = frozenset(
    {
        "unsupported_stereo_type",
        "mapping_budget_exceeded",
        "relative_or_mixture_stereo",
        "mapping_or_representation_unresolved",
        "stock_assessment_incomplete",
        "forward_stereo_not_assessed",
        "conflicting_chemistry_assessments",
    }
)

REVIEWABLE_STEREO_REASONS = frozenset(
    {
        "requires_stereo_forming_step",
        "requires_explicit_resolution_or_inversion_strategy",
    }
)


@dataclass(frozen=True)
class StereoObligations:
    """Persistent path ledger: append local records without copying ancestors."""

    local: tuple = ()
    parent: StereoObligations | None = None
    count: int = 0
    key: str = ""
    unassessed: bool = False

    def extend(self, records):
        records = tuple(records)
        if not records:
            return self
        payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
        return StereoObligations(
            records,
            self,
            self.count + len(records),
            sha256((self.key + payload).encode()).hexdigest(),
            self.unassessed
            or any(o["reason"] in UNASSESSED_STEREO_REASONS for o in records),
        )

    def __iter__(self):
        chain, current = [], self
        while current is not None:
            chain.append(current.local)
            current = current.parent
        for records in reversed(chain):
            yield from records

    def __len__(self):
        return self.count


@dataclass(frozen=True)
class StereoRequirement:
    """A mapped local orientation, linked to its original target atoms."""

    target_atoms: tuple[int, ...]
    kind: str
    atoms: tuple[int, ...]
    environment: tuple[int, ...]
    sign: bool
    group: int | None = None

    def remap(self, mapping: Mapping[int, int]) -> StereoRequirement:
        return StereoRequirement(
            self.target_atoms,
            self.kind,
            tuple(mapping[n] for n in self.atoms),
            tuple(mapping[n] for n in self.environment),
            self.sign,
            self.group,
        )


class UnresolvedStereo(ValueError):
    def __init__(self, reason: str, detail: str | None = None, **context):
        self.reason = reason
        self.context = context
        super().__init__(detail or reason)


def stereo_requirements(mol: MoleculeContainer) -> tuple[StereoRequirement, ...]:
    out = []
    for n, atom in mol.atoms():
        if atom.stereo is None:
            continue
        if n in mol.stereogenic_allenes:
            terminals = mol._stereo_allenes_terminals[n]
            if any(int(mol.bond(n, t)) != 2 for t in terminals):
                raise UnresolvedStereo(
                    "unsupported_stereo_type", "general cumulene axis"
                )
            env = mol.stereogenic_allenes[n][:2]
            out.append(
                StereoRequirement(
                    (n,),
                    "allene",
                    (n, *terminals),
                    env,
                    mol._translate_allene_sign(n, *env),
                    atom.extended_stereo,
                )
            )
            continue
        if n not in mol.stereogenic_tetrahedrons or atom.atomic_number != 6:
            raise UnresolvedStereo(
                "unsupported_stereo_type", f"atom {n}: only carbon tetrahedra supported"
            )
        env = tuple(sorted(mol.stereogenic_tetrahedrons[n]))
        out.append(
            StereoRequirement(
                (n,),
                "tetrahedron",
                (n,),
                env,
                mol._translate_tetrahedron_sign(n, env),
                atom.extended_stereo,
            )
        )
    for n, m, bond in mol.bonds():
        if bond.stereo is None:
            continue
        if int(bond) != 2 or (
            (n, m) not in mol.stereogenic_cis_trans
            and (m, n) not in mol.stereogenic_cis_trans
        ):
            raise UnresolvedStereo(
                "unsupported_stereo_type",
                f"bond {n}-{m}: cumulene or unsupported stereo",
            )
        env = (
            next(k for k in mol.neighbor_numbers(n) if k != m),
            next(k for k in mol.neighbor_numbers(m) if k != n),
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


def stereo_sign(mol: MoleculeContainer, req: StereoRequirement) -> bool | None:
    if req.kind == "allene":
        if mol.atom(req.atoms[0]).stereo is None:
            return None
        return mol._translate_allene_sign(req.atoms[0], *req.environment)
    if req.kind == "tetrahedron":
        if mol.atom(req.atoms[0]).stereo is None:
            return None
        return mol._translate_tetrahedron_sign(req.atoms[0], req.environment)
    n, m = req.atoms
    if mol.bond(n, m).stereo is None:
        return None
    return mol._translate_cis_trans_sign(n, m, *req.environment)


def assign_stereo(mol: MoleculeContainer, req: StereoRequirement) -> None:
    existing = stereo_sign(mol, req)
    if existing is not None:
        if existing != req.sign:
            raise UnresolvedStereo(
                "configuration_contradicted",
                f"opposite mapped configuration at {req.atoms}",
            )
        return
    try:
        if req.kind in ("tetrahedron", "allene"):
            mol.add_atom_stereo(req.atoms[0], req.environment, req.sign)
            mol.atom(req.atoms[0])._extended_stereo = req.group
        else:
            n, m = req.atoms
            n1, n2 = req.environment
            mol.add_cis_trans_stereo(n, m, n1, n2, req.sign)
    except (KeyError, ValueError) as exc:
        raise UnresolvedStereo(
            "requires_stereo_forming_step",
            f"configuration not stereogenic at {req.atoms}: {exc}",
        ) from exc


def local_environment(mol: MoleculeContainer, n: int) -> tuple:
    atom = mol.atom(n)
    return (
        (atom.atomic_number, atom.isotope, atom.charge, atom.is_radical),
        atom.implicit_hydrogens,
        tuple(sorted((k, int(b)) for k, b in mol.bond_items(n))),
    )


def transfer_stereo(
    product: MoleculeContainer,
    reactants: tuple[MoleculeContainer, ...],
    req: StereoRequirement,
    *,
    owners=None,
) -> int:
    if owners is None:
        owners = atom_owners(reactants)
    candidates = owners.get(req.atoms[0], set()).copy()
    for number in req.atoms[1:]:
        candidates.intersection_update(owners.get(number, ()))
    if len(candidates) != 1:
        if req.kind == "double_bond" and all(owners.get(n) for n in req.atoms):
            raise UnresolvedStereo(
                "requires_stereo_forming_step",
                f"double bond {req.atoms} is assembled from separate reactants",
            )
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved",
            f"required atoms {req.atoms} have no unique precursor mapping",
        )
    index = next(iter(candidates))
    precursor = reactants[index]
    if any(
        local_environment(product, n) != local_environment(precursor, n)
        for n in req.atoms
    ):
        reason = "requires_explicit_resolution_or_inversion_strategy"
        if (
            (
                req.kind == "tetrahedron"
                and req.atoms[0] not in precursor.stereogenic_tetrahedrons
            )
            or (
                req.kind == "allene"
                and req.atoms[0] not in precursor.stereogenic_allenes
            )
            or (
                req.kind == "double_bond"
                and int(precursor.get_bond(req.atoms[0], req.atoms[1], 0)) != 2
            )
        ):
            reason = "requires_stereo_forming_step"
        raise UnresolvedStereo(
            reason,
            f"mapped local environment changes at {req.atoms}; retention is unsupported",
        )
    assign_stereo(precursor, req)
    return index


def atom_owners(molecules):
    owners = {}
    for index, molecule in enumerate(molecules):
        for number in molecule:
            owners.setdefault(number, set()).add(index)
    return owners


def has_stereo(mol: MoleculeContainer) -> bool:
    return any(getattr(a, "stereo", None) is not None for _, a in mol.atoms()) or any(
        b.stereo is not None for *_, b in mol.bonds()
    )


def has_stereo_groups(mol: MoleculeContainer) -> bool:
    return any(getattr(a, "extended_stereo", None) for _, a in mol.atoms())


def validate_stereo_input(text: str) -> None:
    """Reject known lossy encodings before Chython can discard their annotation.

    The exception retains the original representation for API error ledgers.
    Extended tetrahedral and allene syntax supported by Chython stays accepted.
    """
    if re.search(r"@(?:SP|TB|OH|TH[3-9]|AL[3-9])|\bw[UD]:", text):
        raise UnresolvedStereo(
            "unsupported_stereo_type",
            "atropisomer or unsupported non-tetrahedral annotation",
            original_input=text,
        )


def parse_smiles_preserving_stereo(text, *, ignore_stereo=False):
    """Allow valence repairs while Chython rejects discarded stereo annotations."""
    from chython import smiles
    from chython.containers import ReactionContainer

    validate_stereo_input(text)
    result = smiles(
        text,
        ignore=True,
        ignore_stereo=ignore_stereo,
        strict_stereo=not ignore_stereo,
    )
    if ignore_stereo:
        return result
    if isinstance(result, ReactionContainer):
        from synplan.chem.utils import reaction_string_mapping_status

        result.meta["stereo_mapping_status"] = reaction_string_mapping_status(
            text.split()[0]
        )
    return result


def assert_stereo_preserved(
    before: MoleculeContainer, after: MoleculeContainer
) -> None:
    """Validate mapped normalization; fail explicitly if it changes a requirement."""
    for req in stereo_requirements(before):
        try:
            valid = stereo_sign(after, req) == req.sign
            if req.kind != "double_bond":
                valid = valid and after.atom(req.atoms[0]).extended_stereo == req.group
        except (KeyError, ValueError):
            valid = False
        if not valid:
            raise UnresolvedStereo(
                "normalization_changed_stereo",
                f"normalization cannot preserve {req.kind} at {req.atoms}",
                original_input=format(before, "m"),
            )


def assess_inheritance(product: MoleculeContainer, reactants) -> dict:
    """Check one mapped backward step; never assign an arbitrary new configuration.

    Reactants are owned working copies. Successful assignments constrain stock
    and subsequent search; failed transfers remain obligations on the whole path.
    """
    events, obligations = [], []
    owners = atom_owners(reactants)
    try:
        reqs = stereo_requirements(product)
    except UnresolvedStereo as error:
        return {
            "events": [],
            "obligations": [{"reason": error.reason, "detail": str(error)}],
        }
    for req in reqs:
        requirement = asdict(req)
        event = {"requirement": requirement, "basis": "mapped_structure"}
        try:
            if req.group:
                raise UnresolvedStereo(
                    "relative_or_mixture_stereo",
                    "group requires a material/relative stereo assessment",
                )
            index = transfer_stereo(product, tuple(reactants), req, owners=owners)
            event.update(event="inherited", reactant=index)
        except (KeyError, ValueError) as error:
            reason = getattr(error, "reason", "mapping_or_representation_unresolved")
            event.update(event="unresolved", reason=reason)
            obligations.append(
                {
                    "reason": reason,
                    "detail": str(error),
                    "requirement": requirement.copy(),
                    "required_product": str(product),
                    "next_action": "Find a supported stereo transformation, compatible chiral precursor, or documented separation.",
                }
            )
        events.append(event)
    return {"events": events, "obligations": obligations}


def stereo_elements(molecules):
    found = {}
    for mol in molecules:
        specified = {(r.kind, r.atoms): r for r in stereo_requirements(mol)}
        keys = list(specified)
        keys += [
            ("tetrahedron", (n,))
            for n in mol.chiral_tetrahedrons
            if mol.atom(n).atomic_number == 6
        ]
        keys += [
            ("allene", (n, *mol._stereo_allenes_terminals[n]))
            for n in mol.chiral_allenes
        ]
        keys += [
            ("double_bond", (n, m))
            for n, m in mol.chiral_cis_trans
            if m in mol.neighbor_numbers(n)
        ]
        for kind, atoms in keys:
            key = (
                kind,
                tuple(sorted(atoms)) if kind == "double_bond" else atoms[:1],
            )
            req = specified.get((kind, atoms))
            if req is None and kind == "double_bond":
                req = specified.get((kind, tuple(reversed(atoms))))
            found[key] = (mol, req, atoms)
    return found


def stereo_events(reaction) -> list[dict]:
    """Analyze mapped source structures, separating creation from annotation gain.

    Reported selectivity is independent metadata. A drawn center has no implied
    ee/er/dr. Atom correspondence must be supplied by the source, never inferred
    from accidentally equal parser numbering.
    """

    left, right = (
        stereo_elements(reaction.reactants),
        stereo_elements(reaction.products),
    )
    if not left and not right:
        return []
    from synplan.chem.utils import reaction_mapping_status

    mapping_status = reaction.meta.get(
        "stereo_mapping_status", reaction_mapping_status(reaction)
    )
    if mapping_status != "fully_mapped":
        return [
            {
                "kind": "unassessed",
                "atoms": list(
                    {
                        n
                        for _, _, atoms in (*left.values(), *right.values())
                        for n in atoms
                    }
                ),
                "event": "mapping_unresolved",
                "mapping_status": mapping_status,
                "selectivity_evidence": "not_established_by_structure",
            }
        ]
    out = []
    for key in left.keys() | right.keys():
        before, after = left.get(key), right.get(key)
        if before is None:
            event = "created"
        elif after is None:
            event = "destroyed"
        elif before[1] is None:
            event = "annotation_added" if after[1] else "unspecified"
        elif after[1] is None:
            event = "annotation_removed"
        else:
            try:
                if any(
                    local_environment(before[0], n) != local_environment(after[0], n)
                    for n in before[1].atoms
                ):
                    event = "reference_environment_changed"
                else:
                    event = (
                        "retained"
                        if stereo_sign(after[0], before[1]) == before[1].sign
                        else "inverted"
                    )
            except (KeyError, ValueError):
                event = "reference_environment_changed"
        out.append(
            {
                "kind": key[0],
                "atoms": list((after or before)[2]),
                "event": event,
                "reactant_specified": bool(before and before[1]),
                "product_specified": bool(after and after[1]),
                "selectivity_evidence": "not_established_by_structure",
            }
        )
    return out
