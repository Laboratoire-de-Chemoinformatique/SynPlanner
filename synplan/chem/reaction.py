"""Module containing classes and functions for manipulating reactions and reaction
rules."""

from collections.abc import Iterator
from typing import Any

from chython.containers import MoleculeContainer, ReactionContainer
from chython.exceptions import InvalidAromaticRing
from chython.reactor import Reactor
from chython.reactor.base import (
    restore_aromaticity,
    snapshot_aromaticity_subset,
)

from synplan.chem.utils import validate_and_canonicalize


class Reaction(ReactionContainer):
    """Reaction class used for a general representation of reaction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CanonicalRetroReactor(Reactor):
    """Reactor subclass that emits **already-canonical** products in a
    single aromatization pass.

    Subclasses :class:`chython.reactor.Reactor` with
    ``fix_aromatic_rings=False`` so the inner ``_patcher`` skips its
    own ``kekule + thiele``; we inline the full canonicalize pipeline
    here instead. Result: ``kekule`` and ``thiele`` each run once per
    product (vs twice in the legacy wrapper + canonicalize pattern).

    Failures raise ``InvalidAromaticRing``, which chython's
    ``_single_stage`` catches and skips silently.
    """

    def __init__(self, *args, **kwargs):
        kwargs["fix_tautomers"] = True
        kwargs["fix_aromatic_rings"] = False  # we run all aromatization in _patcher
        super().__init__(*args, **kwargs)

    def _patcher(self, structure: MoleculeContainer, mapping: dict[int, int]) -> MoleculeContainer:
        new = super()._patcher(structure, mapping)

        # Bug-6 protection: snapshot pre-kekule aromatic atoms.
        pre_aromatic = {n for n, a in new.atoms() if a.hybridization == 4}
        snapshot = (
            snapshot_aromaticity_subset(new, pre_aromatic)
            if pre_aromatic
            else None
        )

        try:
            new.kekule(ignore_pyrrole_hydrogen=self._fix_broken_pyrroles)
        except InvalidAromaticRing:
            raise  # caught by chython._single_stage → rule skipped

        if new.check_valence():
            # ValenceError would escape; InvalidAromaticRing is caught.
            raise InvalidAromaticRing("patched molecule has invalid valence")

        try:
            new.standardize(_fix_stereo=False)
            new.implicify_hydrogens(_fix_stereo=False)
            if not new.thiele(fix_tautomers=self._fix_tautomers):
                new.fix_stereo()
            if pre_aromatic:
                post_aromatic = {n for n, a in new.atoms() if a.hybridization == 4}
                if not pre_aromatic.issubset(post_aromatic):
                    restore_aromaticity(new, snapshot)
            new.standardize_charges(prepare_molecule=False)
            new.standardize_tautomers(prepare_molecule=False)
            new.clean_stereo()
        except InvalidAromaticRing:
            raise  # reject half-canonicalized output

        return new


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
    reaction_rule: "CanonicalRetroReactor",
    sort_reactions: bool = False,
    top_reactions_num: int = 5,
    rebuild_with_cgr: bool = False,
    multirule: bool = False,
    rm_dup: bool = False,
) -> Iterator[list[MoleculeContainer,]]:
    """Applies a reaction rule to a given molecule.

    The yielded precursors are always in canonical form — either
    produced directly by :class:`CanonicalRetroReactor._patcher`
    (default path) or canonicalized via
    :func:`synplan.chem.utils.validate_and_canonicalize` when the CGR
    rebuild path is used. Callers can wrap them with
    ``Precursor(mol, canonicalize=False)`` without further work.

    :param molecule: A molecule to which reaction rule will be applied.
    :param reaction_rule: A :class:`CanonicalRetroReactor`. (Any chython
        ``Reactor`` instance also works mechanically but the yielded
        precursors won't be canonicalized — only ``CanonicalRetroReactor``
        is supported by SynPlanner's MCTS state-dedup contract.)
    :param sort_reactions: If True, candidate reactions are sorted by the
        number of large product fragments (length > 6) before truncation.
    :param top_reactions_num: The maximum amount of reactions after the
        application of reaction rule. **Default raised from 3 → 5 in 1.5.0**;
        callers that depended on the previous default must pass
        ``top_reactions_num=3`` explicitly.
    :param rebuild_with_cgr: If True, products are re-derived by composing
        the reaction into a CGR and decomposing it (recovery path for
        cases where the reactor's direct output has mapping or mass-
        balance issues). The CGR-rebuilt fragments are canonicalized
        explicitly via ``validate_and_canonicalize``; otherwise the
        reactor's already-canonical products are yielded directly.
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
        except (IndexError, InvalidAromaticRing, ValueError):
            # chython's stereo handling raises these on misaligned templates.
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
            # CGR recovery path bypasses _patcher; canonicalize per fragment.
            # chython.compose raises ValueError on element-substitution rules.
            try:
                cgr = reaction.compose()
                reactants = cgr.decompose()[1].split()
            except (ValueError, InvalidAromaticRing):
                return None
            reactants = [mol for mol in reactants if len(mol) > 0]
            canon = []
            for mol in reactants:
                c = validate_and_canonicalize(mol)
                if c is None:
                    return None
                c.meta.update(mol.meta)
                canon.append(c)
            return canon

        return [mol for mol in reaction.products if len(mol) > 0]

    def _reactants_key(reactants: list[MoleculeContainer]) -> tuple[str, ...]:
        return tuple(sorted(str(reactant) for reactant in reactants))

    track_keys = rm_dup or multirule
    seen_reactants: set[tuple[str, ...]] = set()
    pending_reactants: list[list[MoleculeContainer]] = [[molecule]]
    expanded_keys: set[tuple[str, ...]] = (
        {_reactants_key([molecule])} if multirule else set()
    )
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

                reactants_key = (
                    _reactants_key(merged_reactants) if track_keys else None
                )

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
