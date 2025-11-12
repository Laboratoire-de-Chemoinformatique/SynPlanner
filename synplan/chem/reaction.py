"""Module containing classes and functions for manipulating reactions and reaction
rules."""

from typing import Any, Iterator, List, Optional

from CGRtools.containers import MoleculeContainer, ReactionContainer
from CGRtools.exceptions import InvalidAromaticRing
from CGRtools.reactor import Reactor


class Reaction(ReactionContainer):
    """Reaction class used for a general representation of reaction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def add_small_mols(
    big_mol: MoleculeContainer, small_molecules: Optional[Any] = None
) -> List[MoleculeContainer]:
    """Takes a molecule and returns a list of modified molecules where each small
    molecule has been added to the big molecule.

    :param big_mol: A molecule.
    :param small_molecules: A list of small molecules that need to be added to the
        molecule.
    :return: Returns a list of molecules.
    """
    if small_molecules:
        tmp_mol = big_mol.copy()
        transition_mapping = {}
        for small_mol in small_molecules:

            for n, atom in small_mol.atoms():
                new_number = tmp_mol.add_atom(atom.atomic_symbol)
                transition_mapping[n] = new_number

            for atom, neighbor, bond in small_mol.bonds():
                tmp_mol.add_bond(
                    transition_mapping[atom], transition_mapping[neighbor], bond
                )

            transition_mapping = {}
        return tmp_mol.split()

    return [big_mol]


def apply_reaction_rule(
    molecule: MoleculeContainer,
    reaction_rule: Reactor,
    sort_reactions: bool = False,
    top_reactions_num: int = 3,
    validate_products: bool = True,
    rebuild_with_cgr: bool = False,
) -> Iterator[List[MoleculeContainer,]]:
    """Applies a reaction rule to a given molecule.

    :param molecule: A molecule to which reaction rule will be applied.
    :param reaction_rule: A reaction rule to be applied.
    :param sort_reactions:
    :param top_reactions_num: The maximum amount of reactions after the application of
        reaction rule.
    :param validate_products: If True, validates the final products.
    :param rebuild_with_cgr: If True, the products are extracted from CGR decomposition.
    :return: An iterator yielding the products of reaction rule application.
    """

    reactants = add_small_mols(molecule, small_molecules=False)

    try:
        if sort_reactions:
            unsorted_reactions = list(reaction_rule(reactants))
            sorted_reactions = sorted(
                unsorted_reactions,
                key=lambda react: len(
                    list(filter(lambda mol: len(mol) > 6, react.products))
                ),
                reverse=True,
            )

            # take top-N reactions from reactor
            reactions = sorted_reactions[:top_reactions_num]
        else:
            reactions = []
            for reaction in reaction_rule(reactants):
                reactions.append(reaction)
                if len(reactions) == top_reactions_num:
                    break
    except IndexError:
        reactions = []

    for reaction in reactions:

        # temporary solution - incorrect leaving groups
        reactant_atom_nums = []
        for i in reaction.reactants:
            reactant_atom_nums.extend(i.atoms_numbers)
        product_atom_nums = []
        for i in reaction.products:
            product_atom_nums.extend(i.atoms_numbers)
        leaving_atom_nums = set(reactant_atom_nums) - set(product_atom_nums)
        if len(leaving_atom_nums) > len(product_atom_nums):
            continue

        # check reaction
        if rebuild_with_cgr:
            cgr = reaction.compose()
            reactants = cgr.decompose()[1].split()
        else:
            reactants = reaction.products  # reactants are products in retro reaction
        reactants = [mol for mol in reactants if len(mol) > 0]

        # validate products
        if validate_products:
            for mol in reactants:
                try:
                    mol.kekule()
                    if mol.check_valence():
                        yield None
                    mol.thiele()
                except InvalidAromaticRing:
                    yield None

        yield reactants
