"""Module containing a class Precursor that represents a precursor (extend molecule object) in
the search tree."""

from typing import Set

from CGRtools.containers import MoleculeContainer

from synplan.chem.utils import safe_canonicalization


class Precursor:
    """Precursor class is used to extend the molecule behavior needed for interaction with
    a tree in MCTS."""

    def __init__(self, molecule: MoleculeContainer, canonicalize: bool = True):
        """It initializes a Precursor object with a molecule container as a parameter.

        :param molecule: A molecule.
        """
        self.molecule = safe_canonicalization(molecule) if canonicalize else molecule
        self.prev_precursors = []

    def __len__(self) -> int:
        """Return the number of atoms in Precursor."""
        return len(self.molecule)

    def __hash__(self) -> hash:
        """Returns the hash value of Precursor."""
        return hash(self.molecule)

    def __str__(self) -> str:
        """Returns a SMILES of the Precursor."""
        return str(self.molecule)

    def __eq__(self, other: "Precursor") -> bool:
        """Checks if the current Precursor is equal to another Precursor."""
        return self.molecule == other.molecule

    def __repr__(self) -> str:
        """Returns a SMILES of the Precursor."""
        return str(self.molecule)

    def is_building_block(self, bb_stock: Set[str], min_mol_size: int = 6) -> bool:
        """Checks if a Precursor is a building block.

        :param bb_stock: The list of building blocks. Each building block is represented
            by a canonical SMILES.
        :param min_mol_size: If the size of the Precursor is equal or smaller than
            min_mol_size it is automatically classified as building block.
        :return: True is Precursor is a building block.
        """
        if len(self.molecule) <= min_mol_size:
            return True

        return str(self.molecule) in bb_stock


def compose_precursors(
    precursors: list = None, exclude_small: bool = True, min_mol_size: int = 6
) -> MoleculeContainer:
    """
    Takes a list of precursors, excludes small precursors if specified, and composes them
    into a single molecule. The composed molecule then is used for the prediction of
    synthesisability of the characterizing the possible success of the route including
    the nodes with the given precursor.

    :param precursors: The list of precursor to be composed.
    :param exclude_small: The parameter that determines whether small precursor should be excluded from the composition
                          process. If `exclude_small` is set to `True`,
                          only precursor with a length greater than min_mol_size will be composed.
    :param min_mol_size: The parameter used with exclude_small.

    :return: A composed precursor as a MoleculeContainer object.

    """

    if len(precursors) == 1:
        return precursors[0].molecule
    if len(precursors) > 1:
        if exclude_small:
            big_precursor = [
                precursor
                for precursor in precursors
                if len(precursor.molecule) > min_mol_size
            ]
            if big_precursor:
                precursors = big_precursor
        tmp_mol = precursors[0].molecule.copy()
        transition_mapping = {}
        for mol in precursors[1:]:
            for n, atom in mol.molecule.atoms():
                new_number = tmp_mol.add_atom(atom.atomic_symbol)
                transition_mapping[n] = new_number
            for atom, neighbor, bond in mol.molecule.bonds():
                tmp_mol.add_bond(
                    transition_mapping[atom], transition_mapping[neighbor], bond
                )
            transition_mapping = {}

        return tmp_mol
