"""Module containing a class Precursor that represents a precursor (extend molecule object) in
the search tree."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Set

from chython.containers import MoleculeContainer
from chython.exceptions import InvalidAromaticRing
from frozendict import frozendict

from synplan.chem.building_blocks import (
    BuildingBlockCatalogue,
    SQLiteBuildingBlockCatalogue,
    molecule_to_inchikey,
)
from synplan.chem.building_blocks.stereo import compatible_records, selected_record
from synplan.chem.stereo import has_stereo_groups
from synplan.chem.utils import safe_canonicalization

logger = logging.getLogger(__name__)


class Precursor:
    """Precursor class is used to extend the molecule behavior needed for interaction with
    a tree in MCTS."""

    def __init__(self, molecule: MoleculeContainer, canonicalize: bool = True):
        """It initializes a Precursor object with a molecule container as a parameter.

        :param molecule: A molecule.
        """
        self.molecule = safe_canonicalization(molecule) if canonicalize else molecule
        self.prev_precursors = []
        self._inchi_key: str | None = None
        self._inchi_key_error: tuple[type[Exception], str] | None = None
        self.selected_stock = None
        self.stock_diagnostics = []
        self._stock_cache = None
        self._policy_molecule = None

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
    def policy_molecule(self) -> MoleculeContainer:
        """Named connectivity projection for existing stereo-free policy weights.

        The authoritative molecule and its constraints are never modified.
        Parsing the canonical projection gives opposite requirements the same
        model input, including atom order.
        """
        if self._policy_molecule is None:
            from chython import smiles

            projected = self.molecule.copy()
            projected.clean_stereo()
            self._policy_molecule = smiles(str(projected))
        return self._policy_molecule

    @property
    def inchi_key(self) -> str:
        """Return the full Chython Standard InChIKey, generated once."""

        if failure := getattr(self, "_inchi_key_error", None):
            error_type, message = failure
            raise error_type(message)
        if getattr(self, "_inchi_key", None) is None:
            try:
                self._inchi_key = molecule_to_inchikey(self.molecule)
            except (InvalidAromaticRing, ValueError) as error:
                self._inchi_key_error = type(error), str(error)
                logger.warning(
                    "Chython cannot generate an InChIKey for precursor %s; "
                    "treating it as not purchasable: %s",
                    self.molecule,
                    error,
                )
                raise
        return self._inchi_key

    def is_building_block(
        self,
        bb_stock: Set[str] | BuildingBlockCatalogue,
        min_mol_size: int = 6,
    ) -> bool:
        """Checks if a Precursor is a building block.

        :param bb_stock: The list of building blocks. Each building block is represented
            by a canonical SMILES in legacy mode. JSON mode uses an immutable
            prefix-bucket catalogue whose records retain their full InChIKeys.
        :param min_mol_size: Legacy size heuristic parameter; stock membership
            always requires an actual compatible record, including small leaves.
        :return: True is Precursor is a building block.
        """
        if has_stereo_groups(self.molecule):
            return False
        if isinstance(bb_stock, Mapping):
            cached = self._stock_cache
            if (
                isinstance(bb_stock, (frozendict, SQLiteBuildingBlockCatalogue))
                and cached is not None
                and cached[0] is bb_stock
            ):
                return cached[1]
            try:
                records = compatible_records(
                    self.molecule,
                    bb_stock,
                    inchikey=self.inchi_key,
                    diagnostics=self.stock_diagnostics,
                )
            except (InvalidAromaticRing, ValueError):
                return False
            if records:
                record = min(
                    records, key=lambda r: min(r.vendors.values(), default=float("inf"))
                )
                self.selected_stock = selected_record(record)
                self.molecule.meta["selected_stock"] = self.selected_stock
            else:
                self.selected_stock = None
                self.molecule.meta.pop("selected_stock", None)
            self._stock_cache = (bb_stock, bool(records))
            return bool(records)
        return is_purchasable(self.molecule, bb_stock)


def is_purchasable(
    molecule: MoleculeContainer,
    stock: Set[str] | BuildingBlockCatalogue,
    min_mol_size: int = 6,
    *,
    key: str | None = None,
    inchikey: str | None = None,
) -> bool:
    """Whether a molecule has an actual compatible catalogue record.

    A legacy stock is keyed by the molecule's canonical SMILES, so a caller
    holding that string can pass it as ``key``. A JSON catalogue uses the
    molecule's Chython Standard InChIKey instead.
    ``min_mol_size`` is retained for API compatibility and does not affect membership.
    """

    if has_stereo_groups(molecule):
        return False
    if isinstance(stock, Mapping):
        try:
            identity = inchikey or molecule_to_inchikey(molecule)
        except (InvalidAromaticRing, ValueError) as error:
            logger.warning(
                "Chython cannot generate an InChIKey for molecule %s; "
                "treating it as not purchasable: %s",
                molecule,
                error,
            )
            return False
        return bool(compatible_records(molecule, stock, inchikey=identity))
    return (key or str(molecule)) in stock


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
