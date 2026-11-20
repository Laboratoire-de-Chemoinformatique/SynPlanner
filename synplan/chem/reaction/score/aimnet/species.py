"""Species identities and the chemistry supported by LoQI and AIMNet2."""

from rdkit import Chem

from .config import MolecularCalculationError

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


def validate_loqi_charges(mol) -> None:
    """LoQI Flow v1.1.0 encodes six atom-charge classes with offset two."""
    if Chem.AddHs(mol).GetNumAtoms() == 1:
        return  # Atomic species never enter the conformer generator.
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if not -2 <= charge <= 3:
            raise MolecularCalculationError(
                "applicability",
                "unsupported_charge",
                f"LoQI supports atom charges -2 through +3; atom {atom.GetIdx()} "
                f"({atom.GetSymbol()}) has charge {charge}.",
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
    validate_loqi_charges(mol)
    _check_stereo(mol)
