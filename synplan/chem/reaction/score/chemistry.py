"""Chemical applicability, conservation, and exhaustive reaction stereoisomers.

No model imports or conformer generation belong in this preparation stage.
"""

from collections import Counter
from collections.abc import Iterator
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

from synplan.chem.reaction.curation.rebalancing import (
    RebalancingError,
    rebalance_reaction,
)
from synplan.chem.reaction.score.config import MolecularCalculationError
from synplan.chem.utils import mapped_smiles

SUPPORTED_ELEMENTS = frozenset(
    ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I")
)


def canonical_species(mol) -> str:
    """Isomeric identity, including isotopes and charge, excluding atom maps."""
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)


def parse_species(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise MolecularCalculationError("applicability", "invalid_structure", smiles)
    return mol


def is_proton(smiles: str) -> bool:
    """An isolated hydrogen nucleus, with isotope labels retained."""
    mol = parse_species(smiles)
    atom = mol.GetAtomWithIdx(0)
    return (
        mol.GetNumAtoms() == 1
        and atom.GetAtomicNum() == 1
        and atom.GetFormalCharge() == 1
        and not atom.GetNumRadicalElectrons()
    )


def _check_stereo(mol) -> None:
    if mol.GetStereoGroups():
        raise MolecularCalculationError(
            "stereo",
            "unsupported_stereo",
            "Enhanced stereo groups need explicit interpretation.",
        )
    explicit = Chem.AddHs(mol)
    # RDKit can describe each bond of an allene as potential E/Z even though
    # enumeration of the individual bonds cannot represent axial chirality.
    ranks = Chem.CanonicalRankAtoms(explicit, breakTies=False)
    for atom in explicit.GetAtoms():
        ends = [
            b.GetOtherAtom(atom)
            for b in atom.GetBonds()
            if b.GetBondType() == Chem.BondType.DOUBLE
        ]
        if len(ends) == 2:
            substituents = [
                [n for n in end.GetNeighbors() if n.GetIdx() != atom.GetIdx()]
                for end in ends
            ]
            if all(
                len(ns) == 2 and ranks[ns[0].GetIdx()] != ranks[ns[1].GetIdx()]
                for ns in substituents
            ):
                raise MolecularCalculationError(
                    "stereo",
                    "unsupported_stereo",
                    "Axial allene stereochemistry is not supported by LoQI conditioning.",
                )
    allowed_tags = {
        Chem.ChiralType.CHI_UNSPECIFIED,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
    if any(a.GetChiralTag() not in allowed_tags for a in mol.GetAtoms()):
        raise MolecularCalculationError(
            "stereo",
            "unsupported_stereo",
            "Only tetrahedral and ordinary E/Z stereo are supported.",
        )
    for info in Chem.FindPotentialStereo(mol):
        if info.type == Chem.StereoType.Atom_Tetrahedral:
            if explicit.GetAtomWithIdx(info.centeredOn).GetDegree() != 4:
                raise MolecularCalculationError(
                    "stereo",
                    "unsupported_stereo",
                    "LoQI tetrahedral conditioning requires four explicit neighbors.",
                )
        elif info.type != Chem.StereoType.Bond_Double:
            raise MolecularCalculationError(
                "stereo", "unsupported_stereo", str(info.type)
            )


def validate_species(mol) -> None:
    if not mol.GetNumAtoms():
        raise MolecularCalculationError(
            "applicability", "invalid_structure", "Empty molecule."
        )
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in SUPPORTED_ELEMENTS:
            raise MolecularCalculationError(
                "applicability", "unsupported_element", atom.GetSymbol()
            )
        if atom.GetNumRadicalElectrons():
            raise MolecularCalculationError(
                "applicability", "radical", "Only closed-shell species are supported."
            )
    _check_stereo(mol)


def reaction_species(reaction, *, check: bool = True) -> tuple[list[str], list[str]]:
    """Split components without dropping salts, coproducts, or multiplicities."""
    sides = []
    for molecules in (reaction.reactants, reaction.products):
        species = []
        for molecule in molecules:
            try:
                mol = molecule.to_rdkit(keep_mapping=False)
                Chem.SanitizeMol(mol)
            except (ValueError, Chem.rdchem.MolSanitizeException) as exc:
                raise MolecularCalculationError(
                    "applicability", "invalid_structure", str(exc)
                ) from exc
            for fragment in Chem.GetMolFrags(mol, asMols=True):
                if check:
                    validate_species(fragment)
                species.append(canonical_species(fragment))
        if not species:
            raise MolecularCalculationError(
                "applicability",
                "empty_side",
                "Both sides must contain molecular species.",
            )
        sides.append(sorted(species))
    return sides[0], sides[1]


def composition(species: list[str]) -> Counter:
    """Count isotope-resolved atoms (explicit H included) and formal charge."""
    counts: Counter = Counter()
    for text in species:
        mol = Chem.AddHs(parse_species(text))
        for atom in mol.GetAtoms():
            counts[(atom.GetAtomicNum(), atom.GetIsotope())] += 1
            counts["charge"] += atom.GetFormalCharge()
    return counts


def conserved(reactants: list[str], products: list[str]) -> bool:
    return composition(reactants) == composition(products)


def prepare_reaction(
    reaction, min_confidence: float
) -> tuple[dict, list[str], list[str]]:
    """Balance a copy, recording every inferred species and the original equation."""
    original_r, original_p = reaction_species(reaction)
    original = mapped_smiles(reaction)
    if conserved(original_r, original_p):
        balanced, confidence = reaction, 1.0
    else:
        try:
            balanced = rebalance_reaction(
                deepcopy(reaction),
                min_confidence=min_confidence,
                refuse_unsupported_redox=True,
                add_redox_agents=False,
                drop_competing_products=False,
            )
        except RebalancingError as exc:
            raise MolecularCalculationError(
                "balancing", exc.reason, str(exc), category=exc.category
            ) from exc
        confidence = balanced.meta.get("confidence", 1.0)
    reactants, products = reaction_species(balanced, check=False)
    additions = {}
    for side, before, after in (
        ("reactants", original_r, reactants),
        ("products", original_p, products),
    ):
        additions[side] = [
            {"smiles": text, "count": count}
            for text, count in sorted((Counter(after) - Counter(before)).items())
        ]
    info = {
        "original_reaction": original,
        "balanced_reaction": mapped_smiles(balanced),
        "balancing_confidence": confidence,
        "inferred_species": additions,
    }
    try:
        reaction_species(balanced)
        if not conserved(reactants, products):
            raise MolecularCalculationError(
                "balancing",
                "not_conserved",
                "Atoms, isotopes, or net charge are not conserved after balancing.",
            )
    except MolecularCalculationError as exc:
        exc.reaction_context = info
        raise
    return info, reactants, products


def stereoisomers(smiles: str) -> Iterator[str]:
    mol = parse_species(smiles)
    options = StereoEnumerationOptions(
        onlyUnassigned=True, unique=True, maxIsomers=0, tryEmbedding=False
    )
    for isomer in EnumerateStereoisomers(mol, options=options):
        yield canonical_species(isomer)


def reaction_variants(
    reactants: list[str], products: list[str]
) -> Iterator[tuple[list[str], list[str]]]:
    """Lazy Cartesian enumeration, deduplicating permutations of identical species.

    No maximum, random sampling, or stereo-free identity is used. Stereo on
    opposite sides is enumerated independently; these are thermodynamic
    alternatives, not a prediction of stereo retention/inversion or selectivity.
    """
    species = [*reactants, *products]
    chosen: list[str] = []
    seen = set()

    def visit(index):
        if index == len(species):
            left = tuple(sorted(chosen[: len(reactants)]))
            right = tuple(sorted(chosen[len(reactants) :]))
            key = left, right
            if key not in seen:
                seen.add(key)
                yield list(left), list(right)
            return
        for isomer in stereoisomers(species[index]):
            chosen.append(isomer)
            yield from visit(index + 1)
            chosen.pop()

    yield from visit(0)
