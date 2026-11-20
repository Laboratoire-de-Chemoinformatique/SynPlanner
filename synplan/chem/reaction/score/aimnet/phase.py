"""Stoichiometry, achiral phase conventions and solution standard states."""

from collections import Counter
from math import fsum, log

from rdkit import Chem

from .species import canonical_species, parse_species

R_KCAL = 8.31446261815324 / 4184


def net_coefficients(reactants, products):
    coefficients = Counter(products)
    coefficients.subtract(reactants)
    return {species: n for species, n in coefficients.items() if n}


def stoichiometric_sum(coefficients, values):
    """Sum a single extensive quantity in its supplied units, with spectators cancelled."""
    return fsum(n * values[s] for s, n in coefficients.items() if n)


def separates_ions(coefficients):
    return (
        sum(
            n * Chem.GetFormalCharge(parse_species(s)) ** 2
            for s, n in coefficients.items()
        )
        != 0
    )


def _has_unassigned_stereo(molecule):
    explicit = Chem.AddHs(molecule)
    for stereo in Chem.FindPotentialStereo(molecule):
        if stereo.specified == Chem.StereoSpecified.Specified:
            continue
        if stereo.type == Chem.StereoType.Bond_Double:
            # RDKit can mark an individual bond in a cumulene such as an organic
            # azide. A linear center with a bare terminal atom on the other
            # double bond supplies neither E/Z substituents nor axial chirality.
            bond = explicit.GetBondWithIdx(stereo.centeredOn)
            if any(
                other.GetIdx() != bond.GetIdx()
                and other.GetBondType() == Chem.BondType.DOUBLE
                and other.GetOtherAtom(atom).GetDegree() == 1
                for atom in (bond.GetBeginAtom(), bond.GetEndAtom())
                if atom.GetHybridization() == Chem.HybridizationType.SP
                for other in atom.GetBonds()
            ):
                continue
        return True
    return False


def enantiomer_identity(coefficients):
    if len(coefficients) != 2 or sum(coefficients.values()) != 0:
        return False
    left, right = coefficients
    molecule = parse_species(left)
    # Unassigned stereogenic elements make a whole-molecule mirror ambiguous.
    if _has_unassigned_stereo(molecule):
        return False
    for atom in molecule.GetAtoms():
        atom.InvertChirality()
    return canonical_species(molecule) == right


def apply_phase_conventions(variant, config):
    """Apply corrections after molecular summation, preserving the uncorrected numbers."""
    net = net_coefficients(variant["reactants"], variant["products"])
    delta_g, delta_h = variant["delta_g"], variant["delta_h"]
    ledger = []
    if config.solvent_concentration_mol_l and net.get("O"):
        shift = (
            net["O"]
            * R_KCAL
            * config.temperature
            * log(config.solvent_concentration_mol_l / config.concentration_mol_l)
        )
        ledger.append(
            {
                "reason": "solvent_standard_state",
                "species": "O",
                "coefficient": net["O"],
                "delta_g_kcal_mol": shift,
                "delta_h_kcal_mol": 0.0,
            }
        )
        delta_g += shift
    if config.allow_mirror_image and enantiomer_identity(net):
        ledger.append(
            {
                "reason": "enantiomer_identity",
                "delta_g_kcal_mol": -delta_g,
                "delta_h_kcal_mol": -delta_h,
            }
        )
        delta_g = delta_h = 0.0
    if ledger:
        variant["raw_delta_g"] = variant["delta_g"]
        variant["raw_delta_h"] = variant["delta_h"]
        variant["corrections"] = ledger
    variant.update(
        delta_g=delta_g,
        delta_h=delta_h,
        delta_s_cal_mol_K=1000 * (delta_h - delta_g) / config.temperature,
    )
