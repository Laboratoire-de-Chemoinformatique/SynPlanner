"""Module containing additional functions needed in different reaction data processing
protocols."""

import logging
from typing import Iterable

from CGRtools.containers import (
    CGRContainer,
    MoleculeContainer,
    QueryContainer,
    ReactionContainer,
)
from CGRtools.exceptions import InvalidAromaticRing
from tqdm import tqdm

from synplan.chem import smiles_parser
from synplan.utils.files import MoleculeReader, MoleculeWriter

from chython import MoleculeContainer as MoleculeContainerChython


def mol_from_smiles(
    smiles: str,
    standardize: bool = True,
    clean_stereo: bool = True,
    clean2d: bool = True,
) -> MoleculeContainer:
    """Converts a SMILES string to a `MoleculeContainer` object and optionally
    standardizes, cleans stereochemistry, and cleans 2D coordinates.

    :param smiles: The SMILES string representing the molecule.
    :param standardize: Whether to standardize the molecule (default is True).
    :param clean_stereo: Whether to remove the stereo marks on atoms of the molecule (default is True).
    :param clean2d: Whether to clean the 2D coordinates of the molecule (default is True).
    :return: The processed molecule object.
    :raises ValueError: If the SMILES string could not be processed by CGRtools.
    """
    molecule = smiles_parser(smiles)

    if not isinstance(molecule, MoleculeContainer):
        raise ValueError("SMILES string was not processed by CGRtools")

    tmp = molecule.copy()
    try:
        if standardize:
            tmp.canonicalize()
        if clean_stereo:
            tmp.clean_stereo()
        if clean2d:
            tmp.clean2d()
        molecule = tmp
    except InvalidAromaticRing:
        logging.warning(
            "CGRtools was not able to standardize molecule due to invalid aromatic ring"
        )
    return molecule


def query_to_mol(query: QueryContainer) -> MoleculeContainer:
    """Converts a QueryContainer object into a MoleculeContainer object.

    :param query: A QueryContainer object representing the query structure.
    :return: A MoleculeContainer object that replicates the structure of the query.
    """
    new_mol = MoleculeContainer()
    for n, atom in query.atoms():
        new_mol.add_atom(
            atom.atomic_symbol, n, charge=atom.charge, is_radical=atom.is_radical
        )
    for i, j, bond in query.bonds():
        new_mol.add_bond(i, j, int(bond))
    return new_mol


def reaction_query_to_reaction(reaction_rule: ReactionContainer) -> ReactionContainer:
    """Converts a ReactionContainer object with query structures into a
    ReactionContainer with molecular structures.

    :param reaction_rule: A ReactionContainer object where reactants and products are
        QueryContainer objects.
    :return: A new ReactionContainer object where reactants and products are
        MoleculeContainer objects.
    """
    reactants = [query_to_mol(q) for q in reaction_rule.reactants]
    products = [query_to_mol(q) for q in reaction_rule.products]
    reagents = [
        query_to_mol(q) for q in reaction_rule.reagents
    ]  # Assuming reagents are also part of the rule
    reaction = ReactionContainer(reactants, products, reagents, reaction_rule.meta)
    reaction.name = reaction_rule.name
    return reaction


def unite_molecules(molecules: Iterable[MoleculeContainer]) -> MoleculeContainer:
    """Unites a list of MoleculeContainer objects into a single MoleculeContainer. This
    function takes multiple molecules and combines them into one larger molecule. The
    first molecule in the list is taken as the base, and subsequent molecules are united
    with it sequentially.

    :param molecules: A list of MoleculeContainer objects to be united.
    :return: A single MoleculeContainer object representing the union of all input
        molecules.
    """
    new_mol = MoleculeContainer()
    for mol in molecules:
        new_mol = new_mol.union(mol)
    return new_mol


def safe_canonicalization(molecule: MoleculeContainer) -> MoleculeContainer:
    """Attempts to canonicalize a molecule, handling any exceptions. If the
    canonicalization process fails due to an InvalidAromaticRing exception, it safely
    returns the original molecule.

    :param molecule: The given molecule to be canonicalized.
    :return: The canonicalized molecule if successful, otherwise the original molecule.
    """
    molecule._atoms = dict(sorted(molecule._atoms.items()))

    molecule_copy = molecule.copy()
    try:
        molecule_copy.canonicalize()
        molecule_copy.clean_stereo()
        return molecule_copy
    except InvalidAromaticRing:
        return molecule


def standardize_building_blocks(input_file: str, output_file: str) -> str:
    """Standardizes custom building blocks.

    :param input_file: The path to the file that stores the original building blocks.
    :param output_file: The path to the file that will store the standardized building
        blocks.
    :return: The path to the file with standardized building blocks.
    """
    if input_file == output_file:
        raise ValueError("input_file name and output_file name cannot be the same.")

    with MoleculeReader(input_file) as inp_file, MoleculeWriter(
        output_file
    ) as out_file:
        for mol in tqdm(
            inp_file,
            desc="Number of building blocks processed: ",
            bar_format="{desc}{n} [{elapsed}]",
        ):
            try:
                mol = safe_canonicalization(mol)
            except Exception as e:
                logging.debug(e)
                continue
            out_file.write(mol)

    return output_file


def cgr_from_reaction_rule(reaction_rule: ReactionContainer) -> CGRContainer:
    """Creates a CGR from the given reaction rule.

    :param reaction_rule: The reaction rule to be converted.
    :return: The resulting CGR.
    """

    reaction_rule = reaction_query_to_reaction(reaction_rule)
    cgr_rule = ~reaction_rule

    return cgr_rule


def hash_from_reaction_rule(reaction_rule: ReactionContainer) -> hash:
    """Generates hash for the given reaction rule.

    :param reaction_rule: The reaction rule to be converted.
    :return: The resulting hash.
    """

    reactants_hash = tuple(sorted(hash(r) for r in reaction_rule.reactants))
    reagents_hash = tuple(sorted(hash(r) for r in reaction_rule.reagents))
    products_hash = tuple(sorted(hash(r) for r in reaction_rule.products))

    return hash((reactants_hash, reagents_hash, products_hash))


def reverse_reaction(
    reaction: ReactionContainer,
) -> ReactionContainer:
    """Reverses the given reaction.

    :param reaction: The reaction to be reversed.
    :return: The reversed reaction.
    """
    reversed_reaction = ReactionContainer(
        reaction.products, reaction.reactants, reaction.reagents, reaction.meta
    )
    reversed_reaction.name = reaction.name

    return reversed_reaction


def cgrtools_to_chython_molecule(molecule):
    molecule_chython = MoleculeContainerChython()
    for n, atom in molecule.atoms():
        molecule_chython.add_atom(atom.atomic_symbol, n)

    for n, m, bond in molecule.bonds():
        molecule_chython.add_bond(n, m, int(bond))

    return molecule_chython


def chython_query_to_cgrtools(query):
    cgrtools_query = QueryContainer()
    for n, atom in query.atoms():
        cgrtools_query.add_atom(
            atom=atom.atomic_symbol,
            charge=atom.charge,
            neighbors=atom.neighbors,
            hybridization=atom.hybridization,
            _map=n,
        )
    for n, m, bond in query.bonds():
        cgrtools_query.add_bond(n, m, int(bond))

    return cgrtools_query
