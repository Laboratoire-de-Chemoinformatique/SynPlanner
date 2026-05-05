"""Module containing classes and functions for manipulating reactions and reaction
rules."""

from collections.abc import Iterator
from typing import Any

from chython.containers import MoleculeContainer, ReactionContainer
from chython.exceptions import InvalidAromaticRing
from chython.reactor import Reactor


class Reaction(ReactionContainer):
    """Reaction class used for a general representation of reaction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def add_small_mols(
    big_mol: MoleculeContainer, small_molecules: Any | None = None
) -> list[MoleculeContainer]:
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
                new_number = tmp_mol.add_atom(atom.copy())
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
    top_reactions_num: int = 5,
    validate_products: bool = True,
    rebuild_with_cgr: bool = False,
    multirule: bool = False,
    rm_dup: bool = False,
) -> Iterator[list[MoleculeContainer,]]:
    """Applies a reaction rule to a given molecule.

    :param molecule: A molecule to which reaction rule will be applied.
    :param reaction_rule: A reaction rule to be applied.
    :param sort_reactions: If True, candidate reactions are sorted by the
        number of large product fragments (length > 6) before truncation.
    :param top_reactions_num: The maximum amount of reactions after the
        application of reaction rule. **Default raised from 3 → 5 in 1.5.0**;
        callers that depended on the previous default must pass
        ``top_reactions_num=3`` explicitly.
    :param validate_products: If True, validates the final products.
    :param rebuild_with_cgr: If True, the products are extracted from CGR decomposition.
    :param multirule: If True, repeatedly applies the reaction rule to generated
        reactants in a BFS-style loop until no new reactant set is produced.
        Used for priority rules that should iterate (e.g. strip every protective
        group of a given kind from a fully-protected substrate).
    :param rm_dup: If True, removes duplicate reactant sets from yielded outputs
        using a canonical-SMILES dedup key. Recommended whenever ``multirule``
        is set.
    :return: An iterator yielding the products of reaction rule application.
    """

    def _collect_reactions(
        current_molecule: MoleculeContainer,
    ) -> list[ReactionContainer]:
        reactants = add_small_mols(current_molecule, small_molecules=False)
        try:
            if sort_reactions:
                unsorted_reactions = list(reaction_rule(*reactants))
                sorted_reactions = sorted(
                    unsorted_reactions,
                    key=lambda react: len(
                        [mol for mol in react.products if len(mol) > 6]
                    ),
                    reverse=True,
                )
                return sorted_reactions[:top_reactions_num]

            reactions = []
            for reaction in reaction_rule(*reactants):
                reactions.append(reaction)
                if len(reactions) == top_reactions_num:
                    break
            return reactions
        except (IndexError, InvalidAromaticRing):
            return []

    def _prepare_reactants(
        reaction: ReactionContainer,
    ) -> list[MoleculeContainer] | None:
        # temporary solution - incorrect leaving groups
        reactant_atom_nums = []
        for reactant in reaction.reactants:
            reactant_atom_nums.extend(reactant.atoms_numbers)
        product_atom_nums = []
        for product in reaction.products:
            product_atom_nums.extend(product.atoms_numbers)
        leaving_atom_nums = set(reactant_atom_nums) - set(product_atom_nums)
        if len(leaving_atom_nums) > len(product_atom_nums):
            return None

        if rebuild_with_cgr:
            cgr = reaction.compose()
            reactants = cgr.decompose()[1].split()
        else:
            reactants = reaction.products  # reactants are products in retro reaction
        reactants = [mol for mol in reactants if len(mol) > 0]

        if validate_products:
            for mol in reactants:
                try:
                    tmp_mol = mol.copy()
                    tmp_mol.remove_coordinate_bonds(keep_to_terminal=False)
                    tmp_mol.kekule()
                    if tmp_mol.check_valence():
                        return None
                except InvalidAromaticRing:
                    return None

        return reactants

    def _reactants_key(reactants: list[MoleculeContainer]) -> tuple[str, ...]:
        return tuple(sorted(str(reactant) for reactant in reactants))

    seen_reactants = set()
    pending_reactants: list[list[MoleculeContainer]] = [[molecule]]
    expanded_keys = {_reactants_key([molecule])}
    pending_index = 0

    while pending_index < len(pending_reactants):
        current_reactants = pending_reactants[pending_index]
        pending_index += 1

        for mol_index, current_molecule in enumerate(current_reactants):
            for reaction in _collect_reactions(current_molecule):
                new_reactants = _prepare_reactants(reaction)
                if new_reactants is None:
                    continue

                merged_reactants = [
                    reactant
                    for idx, reactant in enumerate(current_reactants)
                    if idx != mol_index
                ]
                merged_reactants.extend(new_reactants)
                merged_reactants = [
                    reactant for reactant in merged_reactants if len(reactant) > 0
                ]

                reactants_key = _reactants_key(merged_reactants)

                if rm_dup and reactants_key in seen_reactants:
                    continue

                if rm_dup:
                    seen_reactants.add(reactants_key)
                yield merged_reactants

                if multirule and reactants_key not in expanded_keys:
                    expanded_keys.add(reactants_key)
                    pending_reactants.append(merged_reactants)

        if not multirule:
            break
