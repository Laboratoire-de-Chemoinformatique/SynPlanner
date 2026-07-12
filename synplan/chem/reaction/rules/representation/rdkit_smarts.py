"""Loss-aware conversion between Chython and RDKit reaction-rule SMARTS."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from chython import smarts

_CXSMARTS_BLOCK_RE = re.compile(r"\s*\|([^|]*)\|")
_RADICAL_CXSMARTS_FIELD_RE = re.compile(r"^\^\d+:[^,|]+$")
_ATOM_MAP_SUFFIX_RE = re.compile(r":[1-9][0-9]*$")


@dataclass(frozen=True)
class RDKitSMARTSConversionResult:
    """Structured result for a Chython-to-RDKit SMARTS conversion audit."""

    original_smarts: str
    rdkit_smarts: str
    parse_status: str
    atom_map_status: str
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    strict: bool = True

    @property
    def ok(self) -> bool:
        """Return whether the conversion is acceptable under the chosen mode."""
        return self.parse_status == "ok" and not self.errors

    @property
    def is_lossless(self) -> bool:
        """Return whether no parse errors or semantic warnings were observed."""
        return self.ok and not self.warnings

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV friendly representation."""
        payload = asdict(self)
        payload["ok"] = self.ok
        payload["is_lossless"] = self.is_lossless
        return payload


@dataclass(frozen=True)
class ChythonSMARTSConversionResult:
    """Structured result for an RDKit-to-Chython SMARTS conversion audit."""

    rdkit_smarts: str
    chython_smarts: str
    rdkit_parse_status: str
    chython_parse_status: str
    atom_map_status: str
    roundtrip_equal: bool | None = None
    expected_chython_smarts: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    strict: bool = True

    @property
    def ok(self) -> bool:
        """Return whether the conversion is acceptable under the chosen mode."""
        return (
            self.rdkit_parse_status == "ok"
            and self.chython_parse_status == "ok"
            and not self.errors
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV friendly representation."""
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


@dataclass(frozen=True)
class SMARTSRoundtripResult:
    """Structured result for Chython -> RDKit -> Chython roundtrip audits."""

    original_smarts: str
    rdkit_smarts: str
    chython_smarts: str
    forward_parse_status: str
    reverse_rdkit_parse_status: str
    reverse_chython_parse_status: str
    atom_map_status: str
    roundtrip_equal: bool
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    strict: bool = True

    @property
    def ok(self) -> bool:
        """Return whether the full exact roundtrip succeeded without errors."""
        return (
            self.forward_parse_status == "ok"
            and self.reverse_rdkit_parse_status == "ok"
            and self.reverse_chython_parse_status == "ok"
            and self.roundtrip_equal
            and not self.errors
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV friendly representation."""
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def _result(
    original_smarts: str,
    *,
    rdkit_smarts: str = "",
    parse_status: str,
    atom_map_status: str = "not_checked",
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    strict: bool,
) -> RDKitSMARTSConversionResult:
    return RDKitSMARTSConversionResult(
        original_smarts=original_smarts,
        rdkit_smarts=rdkit_smarts,
        parse_status=parse_status,
        atom_map_status=atom_map_status,
        warnings=tuple(warnings or ()),
        errors=tuple(errors or ()),
        strict=strict,
    )


def _chython_result(
    rdkit_smarts: str,
    *,
    chython_smarts: str = "",
    rdkit_parse_status: str,
    chython_parse_status: str,
    atom_map_status: str = "not_checked",
    roundtrip_equal: bool | None = None,
    expected_chython_smarts: str | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    strict: bool,
) -> ChythonSMARTSConversionResult:
    return ChythonSMARTSConversionResult(
        rdkit_smarts=rdkit_smarts,
        chython_smarts=chython_smarts,
        rdkit_parse_status=rdkit_parse_status,
        chython_parse_status=chython_parse_status,
        atom_map_status=atom_map_status,
        roundtrip_equal=roundtrip_equal,
        expected_chython_smarts=expected_chython_smarts,
        warnings=tuple(warnings or ()),
        errors=tuple(errors or ()),
        strict=strict,
    )


def _side_to_smarts(molecules: list[Any]) -> str:
    return ".".join(str(molecule) for molecule in molecules)


def _reaction_to_smarts(reaction: Any) -> str:
    # Use per-side molecule SMARTS. ``str(reaction)`` drops atom-map numbers.
    return ">".join(
        [
            _side_to_smarts(list(reaction.reactants)),
            _side_to_smarts(list(reaction.reagents)),
            _side_to_smarts(list(reaction.products)),
        ]
    )


def _reaction_to_rdkit_smarts(reaction: Any) -> str:
    return _reaction_to_smarts(reaction)


def _reaction_to_chython_smarts(reaction: Any) -> str:
    return _reaction_to_smarts(reaction)


def _strip_radical_annotations(smarts_text: str) -> str:
    def replace_block(match: re.Match[str]) -> str:
        fields = [field for field in match.group(1).split(",") if field]
        kept_fields = [
            field for field in fields if not _RADICAL_CXSMARTS_FIELD_RE.fullmatch(field)
        ]
        if not kept_fields:
            return ""
        leading_space = " " if match.group(0).startswith(" ") else ""
        return f"{leading_space}|{','.join(kept_fields)}|"

    return _CXSMARTS_BLOCK_RE.sub(replace_block, smarts_text)


def _roundtrip_smarts_equal(observed: str, expected: str) -> bool:
    return observed == expected or (
        _strip_radical_annotations(observed) == _strip_radical_annotations(expected)
    )


def _split_top_level(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    start = 0
    paren_depth = 0
    for index, char in enumerate(text):
        if char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif char == separator and paren_depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _strip_full_side_parentheses(side_smarts: str) -> str:
    side_smarts = side_smarts.strip()
    while side_smarts.startswith("(") and side_smarts.endswith(")"):
        depth = 0
        in_atom = False
        wraps_full_side = True
        for index, char in enumerate(side_smarts):
            if char == "[":
                in_atom = True
            elif char == "]":
                in_atom = False
            elif not in_atom:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and index != len(side_smarts) - 1:
                        wraps_full_side = False
                        break
                    if depth < 0:
                        wraps_full_side = False
                        break
        if not wraps_full_side or depth != 0:
            break
        side_smarts = side_smarts[1:-1].strip()
    return side_smarts


def _drop_rdkit_neutral_charge(primitive: str) -> str | None:
    if primitive == "+0":
        return None
    if primitive.endswith("+0"):
        return primitive[:-2]
    return primitive


def _normalize_rdkit_atom_query_for_chython(atom_query: str) -> str:
    atom_query = ";".join(_split_top_level(atom_query, "&"))

    atom_map = ""
    if match := _ATOM_MAP_SUFFIX_RE.search(atom_query):
        atom_map = match.group(0)
        atom_query = atom_query[: match.start()]

    primitives = []
    for primitive in _split_top_level(atom_query, ";"):
        normalized = _drop_rdkit_neutral_charge(primitive)
        if normalized:
            primitives.append(normalized)

    return ";".join(primitives) + atom_map


def _normalize_rdkit_atom_queries_for_chython(reaction_smarts: str) -> str:
    normalized: list[str] = []
    index = 0
    while index < len(reaction_smarts):
        char = reaction_smarts[index]
        if char != "[":
            normalized.append(char)
            index += 1
            continue

        atom_start = index
        index += 1
        paren_depth = 0
        while index < len(reaction_smarts):
            current = reaction_smarts[index]
            if current == "(":
                paren_depth += 1
            elif current == ")" and paren_depth:
                paren_depth -= 1
            elif current == "]" and paren_depth == 0:
                atom_query = reaction_smarts[atom_start + 1 : index]
                normalized.append(
                    "[" + _normalize_rdkit_atom_query_for_chython(atom_query) + "]"
                )
                index += 1
                break
            index += 1
        else:
            normalized.append(reaction_smarts[atom_start:])
            break

    return "".join(normalized)


def normalize_rdkit_rule_smarts_for_chython(rdkit_smarts: str) -> str:
    """Normalize RDKit/rdchiral rule SMARTS into Chython-readable syntax.

    RetroChimera templates are RDKit/rdchiral-style SMARTS. They can wrap each
    reaction side in a full-side parenthesis and use ``&`` conjunctions plus
    explicit neutral-charge primitives such as ``+0`` inside atom queries.
    Chython expects the same supported query semantics with ``;`` conjunctions
    and uses neutral charge as the default atom-query state.
    """
    reaction_parts = rdkit_smarts.strip().split(">")
    normalized_parts = [_strip_full_side_parentheses(part) for part in reaction_parts]
    normalized = ">".join(normalized_parts)
    return _normalize_rdkit_atom_queries_for_chython(normalized)


def _side_fragments(side_smarts: str) -> list[str]:
    return [fragment for fragment in side_smarts.split(".") if fragment]


def _fragment_atom_maps(fragment_mol: Any) -> tuple[int, ...]:
    return tuple(atom.GetAtomMapNum() for atom in fragment_mol.GetAtoms())


def _query_text(atom: Any) -> str:
    try:
        return atom.DescribeQuery()
    except Exception:
        return ""


def _query_has_all_values(
    query_text: str, query_name: str, values: tuple[int, ...]
) -> bool:
    return all(f"{query_name} {value} = val" in query_text for value in values)


def _chython_atom_signature(atom: Any) -> dict[str, Any]:
    return {
        "atomic_number": getattr(atom, "atomic_number", None),
        "atomic_symbol": getattr(atom, "atomic_symbol", None),
        "charge": getattr(atom, "charge", 0),
        "isotope": getattr(atom, "isotope", None),
        "is_radical": getattr(atom, "is_radical", False),
        "implicit_hydrogens": tuple(getattr(atom, "implicit_hydrogens", ()) or ()),
        "neighbors": tuple(getattr(atom, "neighbors", ()) or ()),
        "hybridization": tuple(getattr(atom, "hybridization", ()) or ()),
        "ring_sizes": tuple(getattr(atom, "ring_sizes", ()) or ()),
        "rings_count": tuple(getattr(atom, "rings_count", ()) or ()),
        "heteroatoms": tuple(getattr(atom, "heteroatoms", ()) or ()),
    }


def _rdkit_bond_order(bond: Any) -> int | None:
    from rdkit import Chem

    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return 1
    if bond_type == Chem.BondType.DOUBLE:
        return 2
    if bond_type == Chem.BondType.TRIPLE:
        return 3
    if bond_type == Chem.BondType.AROMATIC:
        return 4
    return None


def _chython_bonds_by_map(molecule: Any) -> dict[tuple[int, int], int]:
    bonds: dict[tuple[int, int], int] = {}
    for atom_1, atom_2, bond in molecule.bonds():
        bonds[tuple(sorted((atom_1, atom_2)))] = int(bond)
    return bonds


def _rdkit_bonds_by_map(fragment_mol: Any) -> dict[tuple[int, int], int | None]:
    atom_maps = {
        atom.GetIdx(): atom.GetAtomMapNum() for atom in fragment_mol.GetAtoms()
    }
    bonds: dict[tuple[int, int], int | None] = {}
    for bond in fragment_mol.GetBonds():
        atom_1 = atom_maps[bond.GetBeginAtomIdx()]
        atom_2 = atom_maps[bond.GetEndAtomIdx()]
        bonds[tuple(sorted((atom_1, atom_2)))] = _rdkit_bond_order(bond)
    return bonds


def _verify_atom_query(
    *,
    side: str,
    fragment_index: int,
    atom_map: int,
    chython_atom: Any,
    rdkit_atom: Any,
    warnings: list[str],
    errors: list[str],
) -> None:
    signature = _chython_atom_signature(chython_atom)
    query_text = _query_text(rdkit_atom)
    prefix = f"{side}[{fragment_index}] atom_map={atom_map}"

    if signature["atomic_number"] != rdkit_atom.GetAtomicNum():
        errors.append(
            f"{prefix}: atomic number mismatch "
            f"chython={signature['atomic_number']} rdkit={rdkit_atom.GetAtomicNum()}"
        )
    if signature["charge"] != rdkit_atom.GetFormalCharge():
        errors.append(
            f"{prefix}: charge mismatch "
            f"chython={signature['charge']} rdkit={rdkit_atom.GetFormalCharge()}"
        )
    chython_isotope = signature["isotope"] or 0
    if chython_isotope != rdkit_atom.GetIsotope():
        errors.append(
            f"{prefix}: isotope mismatch "
            f"chython={chython_isotope} rdkit={rdkit_atom.GetIsotope()}"
        )

    implicit_hydrogens = signature["implicit_hydrogens"]
    if implicit_hydrogens and not _query_has_all_values(
        query_text, "AtomHCount", implicit_hydrogens
    ):
        errors.append(
            f"{prefix}: RDKit query does not expose Chython H-count "
            f"{implicit_hydrogens}"
        )

    neighbors = signature["neighbors"]
    if neighbors and not _query_has_all_values(
        query_text, "AtomExplicitDegree", neighbors
    ):
        errors.append(
            f"{prefix}: RDKit query does not expose Chython degree {neighbors}"
        )

    ring_sizes = signature["ring_sizes"]
    if ring_sizes and not _query_has_all_values(
        query_text, "AtomMinRingSize", ring_sizes
    ):
        errors.append(
            f"{prefix}: RDKit query does not expose Chython ring sizes {ring_sizes}"
        )

    hybridization = signature["hybridization"]
    if hybridization == (4,) and not rdkit_atom.GetIsAromatic():
        errors.append(
            f"{prefix}: Chython aromatic query parsed as non-aromatic RDKit atom"
        )
    elif hybridization == (1, 2, 3) and rdkit_atom.GetIsAromatic():
        errors.append(
            f"{prefix}: Chython aliphatic query parsed as aromatic RDKit atom"
        )
    elif hybridization and hybridization not in ((1, 2, 3), (4,)):
        warnings.append(
            f"{prefix}: unverified Chython hybridization query {hybridization}"
        )

    for field in ("rings_count", "heteroatoms"):
        value = signature[field]
        if value:
            warnings.append(f"{prefix}: unverified Chython {field} query {value}")

    if signature["is_radical"]:
        warnings.append(f"{prefix}: unverified Chython radical query")


def _verify_fragment(
    *,
    side: str,
    fragment_index: int,
    chython_molecule: Any,
    rdkit_molecule: Any,
    warnings: list[str],
    errors: list[str],
) -> None:
    chython_atoms = {atom_map: atom for atom_map, atom in chython_molecule.atoms()}
    rdkit_atoms = {atom.GetAtomMapNum(): atom for atom in rdkit_molecule.GetAtoms()}

    if tuple(chython_atoms) != tuple(rdkit_atoms):
        errors.append(
            f"{side}[{fragment_index}]: atom-map order mismatch "
            f"chython={tuple(chython_atoms)} rdkit={tuple(rdkit_atoms)}"
        )
    if set(chython_atoms) != set(rdkit_atoms):
        errors.append(
            f"{side}[{fragment_index}]: atom-map set mismatch "
            f"chython={sorted(chython_atoms)} rdkit={sorted(rdkit_atoms)}"
        )
        return

    for atom_map, chython_atom in chython_atoms.items():
        _verify_atom_query(
            side=side,
            fragment_index=fragment_index,
            atom_map=atom_map,
            chython_atom=chython_atom,
            rdkit_atom=rdkit_atoms[atom_map],
            warnings=warnings,
            errors=errors,
        )

    chython_bonds = _chython_bonds_by_map(chython_molecule)
    rdkit_bonds = _rdkit_bonds_by_map(rdkit_molecule)
    if set(chython_bonds) != set(rdkit_bonds):
        errors.append(
            f"{side}[{fragment_index}]: bond map set mismatch "
            f"chython={sorted(chython_bonds)} rdkit={sorted(rdkit_bonds)}"
        )
    for bond_key in sorted(set(chython_bonds) & set(rdkit_bonds)):
        if chython_bonds[bond_key] != rdkit_bonds[bond_key]:
            errors.append(
                f"{side}[{fragment_index}]: bond order mismatch {bond_key} "
                f"chython={chython_bonds[bond_key]} rdkit={rdkit_bonds[bond_key]}"
            )


def _parse_rdkit_side(
    side_smarts: str,
    *,
    side: str,
    errors: list[str],
) -> list[Any]:
    from rdkit import Chem

    molecules = []
    for index, fragment in enumerate(_side_fragments(side_smarts)):
        molecule = Chem.MolFromSmarts(fragment, mergeHs=False)
        if molecule is None:
            errors.append(f"{side}[{index}]: RDKit could not parse {fragment!r}")
        else:
            molecules.append(molecule)
    return molecules


def _atom_map_status(
    chython_sides: list[list[Any]], rdkit_sides: list[list[Any]]
) -> str:
    rdkit_maps = [
        tuple(_map for molecule in side for _map in _fragment_atom_maps(molecule))
        for side in rdkit_sides
    ]
    explicit_maps = {
        atom_map for side in rdkit_maps for atom_map in side if atom_map > 0
    }
    chython_maps = [
        {
            _map
            for molecule in side
            for _map, _atom in molecule.atoms()
            if _map in explicit_maps
        }
        for side in chython_sides
    ]
    rdkit_maps = [{_map for _map in side if _map > 0} for side in rdkit_maps]
    return "ok" if chython_maps == rdkit_maps else "mismatch"


def _chython_sides_from_reaction(reaction: Any) -> list[list[Any]]:
    return [list(reaction.reactants), list(reaction.reagents), list(reaction.products)]


def _rdkit_sides_from_smarts(
    reaction_smarts: str, errors: list[str]
) -> list[list[Any]]:
    parts = reaction_smarts.split(">")
    if len(parts) != 3:
        errors.append("reaction SMARTS must contain exactly three '>'-separated sides")
        parts = [*parts, "", "", ""][:3]
    side_names = ["reactants", "reagents", "products"]
    return [
        _parse_rdkit_side(text, side=side, errors=errors)
        for side, text in zip(side_names, parts, strict=True)
    ]


def _validate_rdkit_reaction_smarts(reaction_smarts: str, errors: list[str]) -> None:
    from rdkit.Chem import AllChem

    try:
        rdkit_reaction = AllChem.ReactionFromSmarts(reaction_smarts)
        if rdkit_reaction is None:
            errors.append("RDKit could not parse full reaction SMARTS")
        else:
            rdkit_reaction.Initialize()
    except Exception as exc:
        errors.append(f"RDKit full reaction parse failed: {type(exc).__name__}: {exc}")


def chython_rule_smarts_to_rdkit_smarts(
    rule_smarts: str, *, strict: bool = True
) -> RDKitSMARTSConversionResult:
    """Convert Chython reaction-rule SMARTS to RDKit-parseable SMARTS.

    The converter reconstructs the reaction from Chython's per-side molecule
    SMARTS because the full reaction string drops atom-map numbers. It then
    parses the emitted fragments with RDKit and checks the parsed query against
    the Chython query objects. In strict mode, unverified Chython query
    semantics are treated as errors so semantic loss cannot pass silently.
    """
    try:
        reaction = smarts(rule_smarts)
    except Exception as exc:
        return _result(
            rule_smarts,
            parse_status="chython_parse_failed",
            errors=[f"{type(exc).__name__}: {exc}"],
            strict=strict,
        )

    rdkit_smarts = _reaction_to_rdkit_smarts(reaction)
    chython_sides = _chython_sides_from_reaction(reaction)
    side_names = ["reactants", "reagents", "products"]

    warnings: list[str] = []
    errors: list[str] = []
    rdkit_sides = _rdkit_sides_from_smarts(rdkit_smarts, errors)
    _validate_rdkit_reaction_smarts(rdkit_smarts, errors)

    for side, chython_molecules, rdkit_molecules in zip(
        side_names, chython_sides, rdkit_sides, strict=True
    ):
        if len(chython_molecules) != len(rdkit_molecules):
            errors.append(
                f"{side}: fragment count mismatch "
                f"chython={len(chython_molecules)} rdkit={len(rdkit_molecules)}"
            )
            continue
        for index, (chython_molecule, rdkit_molecule) in enumerate(
            zip(chython_molecules, rdkit_molecules, strict=True)
        ):
            _verify_fragment(
                side=side,
                fragment_index=index,
                chython_molecule=chython_molecule,
                rdkit_molecule=rdkit_molecule,
                warnings=warnings,
                errors=errors,
            )

    atom_map_status = _atom_map_status(chython_sides, rdkit_sides)
    if atom_map_status != "ok":
        errors.append("atom-map sequence mismatch across reaction sides")

    if strict and warnings:
        errors.extend(f"strict semantic warning: {warning}" for warning in warnings)

    parse_status = "ok"
    if errors:
        parse_status = "semantic_loss" if rdkit_smarts else "rdkit_parse_failed"

    return _result(
        rule_smarts,
        rdkit_smarts=rdkit_smarts,
        parse_status=parse_status,
        atom_map_status=atom_map_status,
        warnings=warnings,
        errors=errors,
        strict=strict,
    )


def rdkit_rule_smarts_to_chython_smarts(
    rdkit_smarts: str,
    *,
    strict: bool = True,
    expected_chython_smarts: str | None = None,
) -> ChythonSMARTSConversionResult:
    """Convert RDKit reaction SMARTS to Chython-normalized rule SMARTS.

    RDKit parsing is used as the front-door validation step. Chython then parses
    the same SMARTS and re-emits per-side query molecule SMARTS so atom-map
    numbers are preserved. When expected Chython SMARTS is supplied, exact
    string equality is checked, except Chython/RDKit radical CXSMARTS
    annotations are ignored for roundtrip pass/fail.
    """
    warnings: list[str] = []
    errors: list[str] = []
    normalized_rdkit_smarts = normalize_rdkit_rule_smarts_for_chython(rdkit_smarts)
    if normalized_rdkit_smarts != rdkit_smarts:
        warnings.append(
            "RDKit SMARTS was normalized for Chython compatibility before parsing"
        )

    rdkit_sides = _rdkit_sides_from_smarts(normalized_rdkit_smarts, errors)
    _validate_rdkit_reaction_smarts(normalized_rdkit_smarts, errors)
    rdkit_parse_status = "ok" if not errors else "rdkit_parse_failed"

    try:
        reaction = smarts(normalized_rdkit_smarts)
    except Exception as exc:
        return _chython_result(
            normalized_rdkit_smarts,
            rdkit_parse_status=rdkit_parse_status,
            chython_parse_status="chython_parse_failed",
            errors=[*errors, f"{type(exc).__name__}: {exc}"],
            strict=strict,
            expected_chython_smarts=expected_chython_smarts,
        )

    chython_smarts = _reaction_to_chython_smarts(reaction)
    chython_sides = _chython_sides_from_reaction(reaction)
    atom_map_status = _atom_map_status(chython_sides, rdkit_sides)
    if atom_map_status != "ok":
        errors.append("atom-map sequence mismatch across reaction sides")

    roundtrip_equal: bool | None = None
    if expected_chython_smarts is not None:
        exact_equal = chython_smarts == expected_chython_smarts
        roundtrip_equal = exact_equal or _roundtrip_smarts_equal(
            chython_smarts, expected_chython_smarts
        )
        if roundtrip_equal and not exact_equal:
            warnings.append(
                "Chython-normalized SMARTS matches expected SMARTS after "
                "dropping radical CXSMARTS annotations"
            )
        if not roundtrip_equal:
            message = (
                "Chython-normalized SMARTS does not match expected SMARTS: "
                f"expected={expected_chython_smarts!r} observed={chython_smarts!r}"
            )
            if strict:
                errors.append(message)
            else:
                warnings.append(message)
    elif chython_smarts != normalized_rdkit_smarts:
        warnings.append(
            "Chython-normalized SMARTS differs from input RDKit SMARTS; "
            "no expected Chython SMARTS was provided"
        )

    return _chython_result(
        normalized_rdkit_smarts,
        chython_smarts=chython_smarts,
        rdkit_parse_status=rdkit_parse_status,
        chython_parse_status="ok",
        atom_map_status=atom_map_status,
        roundtrip_equal=roundtrip_equal,
        expected_chython_smarts=expected_chython_smarts,
        warnings=warnings,
        errors=errors,
        strict=strict,
    )


def roundtrip_chython_rdkit_chython(
    rule_smarts: str, *, strict: bool = True
) -> SMARTSRoundtripResult:
    """Audit a Chython -> RDKit -> Chython SMARTS roundtrip.

    Strict mode requires the forward semantic audit to pass as well as an
    accepted reverse roundtrip.
    """
    forward = chython_rule_smarts_to_rdkit_smarts(rule_smarts, strict=strict)
    if not forward.rdkit_smarts:
        return SMARTSRoundtripResult(
            original_smarts=rule_smarts,
            rdkit_smarts="",
            chython_smarts="",
            forward_parse_status=forward.parse_status,
            reverse_rdkit_parse_status="not_run",
            reverse_chython_parse_status="not_run",
            atom_map_status=forward.atom_map_status,
            roundtrip_equal=False,
            warnings=forward.warnings,
            errors=forward.errors,
            strict=strict,
        )

    reverse = rdkit_rule_smarts_to_chython_smarts(
        forward.rdkit_smarts,
        strict=strict,
        expected_chython_smarts=rule_smarts,
    )
    return SMARTSRoundtripResult(
        original_smarts=rule_smarts,
        rdkit_smarts=forward.rdkit_smarts,
        chython_smarts=reverse.chython_smarts,
        forward_parse_status=forward.parse_status,
        reverse_rdkit_parse_status=reverse.rdkit_parse_status,
        reverse_chython_parse_status=reverse.chython_parse_status,
        atom_map_status=reverse.atom_map_status,
        roundtrip_equal=reverse.roundtrip_equal is True,
        warnings=tuple((*forward.warnings, *reverse.warnings)),
        errors=tuple((*forward.errors, *reverse.errors)),
        strict=strict,
    )


__all__ = [
    "ChythonSMARTSConversionResult",
    "RDKitSMARTSConversionResult",
    "SMARTSRoundtripResult",
    "chython_rule_smarts_to_rdkit_smarts",
    "normalize_rdkit_rule_smarts_for_chython",
    "rdkit_rule_smarts_to_chython_smarts",
    "roundtrip_chython_rdkit_chython",
]
