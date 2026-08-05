"""Molecule wrapper used as a precursor state during tree search."""

from chython.containers import MoleculeContainer

from synplan.chem.molecule.standardization import safe_canonicalization


class Precursor:
    """Extend a molecule with the state needed by tree search."""

    def __init__(self, molecule: MoleculeContainer, canonicalize: bool = True):
        self.molecule = safe_canonicalization(molecule) if canonicalize else molecule
        self.prev_precursors = []

    def __len__(self) -> int:
        return len(self.molecule)

    def __hash__(self) -> int:
        return hash(self.molecule)

    def __str__(self) -> str:
        return str(self.molecule)

    def __eq__(self, other: "Precursor") -> bool:
        return self.molecule == other.molecule

    def __repr__(self) -> str:
        return str(self.molecule)

    def is_building_block(self, bb_stock: set[str], min_mol_size: int = 6) -> bool:
        """Return whether this precursor is small enough or present in stock."""
        if len(self.molecule) <= min_mol_size:
            return True
        return str(self.molecule) in bb_stock


def compose_precursors(
    precursors: list | None = None,
    exclude_small: bool = True,
    min_mol_size: int = 6,
) -> MoleculeContainer:
    """Compose precursor molecules into one disconnected molecule."""
    if len(precursors) == 1:
        return precursors[0].molecule
    if len(precursors) > 1:
        if exclude_small:
            big_precursors = [
                precursor
                for precursor in precursors
                if len(precursor.molecule) > min_mol_size
            ]
            if big_precursors:
                precursors = big_precursors
        output = precursors[0].molecule.copy()
        transition_mapping = {}
        for precursor in precursors[1:]:
            for atom_number, atom in precursor.molecule.atoms():
                transition_mapping[atom_number] = output.add_atom(atom.copy())
            for atom, neighbor, bond in precursor.molecule.bonds():
                output.add_bond(
                    transition_mapping[atom], transition_mapping[neighbor], bond
                )
            transition_mapping = {}
        return output


__all__ = ["Precursor", "compose_precursors"]
