"""Conservative balancing, conservation, and exhaustive reaction stereoisomers."""

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
from synplan.chem.utils import mapped_smiles

from .config import MolecularCalculationError
from .species import canonical_species, parse_species, validate_species


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
    reaction, min_confidence: float, *, infer_balance: bool = True
) -> tuple[dict, list[str], list[str]]:
    """Balance a copy, recording every inferred species and the original equation."""
    original_r, original_p = reaction_species(reaction)
    original = mapped_smiles(reaction)
    if conserved(original_r, original_p):
        balanced, confidence = reaction, 1.0
    elif not infer_balance:
        raise MolecularCalculationError(
            "balancing",
            "not_conserved",
            "Supplied equation does not conserve atoms, isotopes, or charge.",
        )
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
