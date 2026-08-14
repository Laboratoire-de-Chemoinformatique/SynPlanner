"""Module containing a class Precursor that represents a precursor (extend molecule object) in
the search tree."""

from __future__ import annotations

from collections.abc import Set

from chython.containers import MoleculeContainer

from synplan.chem.building_blocks.identity import molecule_to_inchi_key
from synplan.chem.building_blocks.stock import BuildingBlockStock
from synplan.chem.utils import safe_canonicalization


class Precursor:
    """Extend a molecule with state used by MCTS.

    The contained molecule is an immutable search-state value after construction;
    mutating it would invalidate both hashing and the lazy identity cache.
    """

    def __init__(self, molecule: MoleculeContainer, canonicalize: bool = True):
        """It initializes a Precursor object with a molecule container as a parameter.

        :param molecule: A molecule.
        """
        self.molecule = (
            safe_canonicalization(molecule, clean_stereo=False)
            if canonicalize
            else molecule
        )
        self.prev_precursors = []
        self._inchi_key: str | None = None

    def __len__(self) -> int:
        """Return the number of atoms in Precursor."""
        return len(self.molecule)

    def __hash__(self) -> int:
        """Returns the hash value of Precursor."""
        return hash(self.molecule)

    def __str__(self) -> str:
        """Returns a SMILES of the Precursor."""
        return str(self.molecule)

    def __eq__(self, other: Precursor) -> bool:
        """Checks if the current Precursor is equal to another Precursor."""
        return self.molecule == other.molecule

    def __repr__(self) -> str:
        """Returns a SMILES of the Precursor."""
        return str(self.molecule)

    @property
    def inchi_key(self) -> str:
        """Return and cache the full Standard InChIKey for this precursor."""
        cached = getattr(self, "_inchi_key", None)
        if cached is None:
            cached = molecule_to_inchi_key(self.molecule)
            self._inchi_key = cached
        return cached

    def is_building_block(
        self, bb_stock: BuildingBlockStock | Set[str], min_mol_size: int = 6
    ) -> bool:
        """Checks if a Precursor is a building block.

        :param bb_stock: A typed building-block stock or a legacy set of canonical
            SMILES.
        :param min_mol_size: If the size of the Precursor is equal or smaller than
            min_mol_size it is automatically classified as building block.
        :return: True is Precursor is a building block.
        """
        if len(self.molecule) <= min_mol_size:
            return True

        if (
            isinstance(bb_stock, BuildingBlockStock)
            and bb_stock.identity_format == "inchikey"
        ):
            return self.inchi_key in bb_stock

        contains_molecule = getattr(bb_stock, "contains_molecule", None)
        if contains_molecule is not None:
            return bool(contains_molecule(self.molecule))
        return str(self.molecule) in bb_stock


def compose_precursors(
    precursors: list | None = None, exclude_small: bool = True, min_mol_size: int = 6
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
                new_number = tmp_mol.add_atom(atom.copy())
                transition_mapping[n] = new_number
            for atom, neighbor, bond in mol.molecule.bonds():
                tmp_mol.add_bond(
                    transition_mapping[atom], transition_mapping[neighbor], bond
                )
            transition_mapping = {}

        return tmp_mol
