"""Module containing classes and functions for manipulating reactions and reaction
rules."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from chython.containers import MoleculeContainer, ReactionContainer, SynthonContainer
from chython.exceptions import InvalidAromaticRing, IsChiral, NotChiral, ValenceError
from chython.reactor import Reactor
from chython.reactor.base import (
    restore_aromaticity,
    snapshot_aromaticity_subset,
)

from synplan.chem.utils import validate_and_canonicalize

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _AtomStereoDescriptor:
    """Atom stereo expressed against source atom-map neighbours."""

    center: int
    environment: tuple[int, ...]
    mark: bool


@dataclass(frozen=True, slots=True)
class _CisTransStereoDescriptor:
    """Cis/trans stereo expressed by mapped terminals and substituents."""

    first_terminal: int
    second_terminal: int
    first_neighbor: int
    second_neighbor: int
    mark: bool


def _snapshot_product_stereo(
    products: tuple[MoleculeContainer, ...],
) -> tuple[tuple[_AtomStereoDescriptor, ...], tuple[_CisTransStereoDescriptor, ...]]:
    """Capture transferable stereo without modifying reactor products.

    A CGR stores atom/bond changes but not molecule stereo. Atom-map numbers survive
    compose/decompose, so Chython's public stereo APIs can translate source parity
    against the rebuilt molecule's neighbour ordering.
    """
    atom_descriptors: list[_AtomStereoDescriptor] = []
    bond_descriptors: list[_CisTransStereoDescriptor] = []
    for product in products:
        tetrahedrons = product.stereogenic_tetrahedrons
        allenes = product.stereogenic_allenes
        for center, atom in product.atoms():
            if atom.stereo is None:
                continue
            if center in tetrahedrons:
                environment = tetrahedrons[center]
            elif center in allenes:
                allene_environment = allenes[center]
                environment = allene_environment[:2]
            else:
                continue
            atom_descriptors.append(
                _AtomStereoDescriptor(center, tuple(environment), atom.stereo)
            )

        cis_trans_paths = {
            frozenset((path[0], path[-1])): path
            for path in product.cumulenes
            if not len(path) % 2
        }
        for (first, second), environment in product.stereogenic_cis_trans.items():
            path = cis_trans_paths[frozenset((first, second))]
            middle = len(path) // 2
            mark = product.bond(path[middle - 1], path[middle]).stereo
            if mark is not None:
                bond_descriptors.append(
                    _CisTransStereoDescriptor(
                        first,
                        second,
                        environment[0],
                        environment[1],
                        mark,
                    )
                )
    return tuple(atom_descriptors), tuple(bond_descriptors)


def _restore_product_stereo(
    molecule: MoleculeContainer,
    atom_descriptors: tuple[_AtomStereoDescriptor, ...],
    bond_descriptors: tuple[_CisTransStereoDescriptor, ...],
) -> None:
    """Restore still-valid stereo onto one CGR-rebuilt fragment in place.

    Descriptors spanning another fragment are ignored. ``NotChiral`` descriptors are
    retried because Chython can resolve dependent stereocentres only after another
    mark is installed. Descriptors the rebuilt graph never accepts are obsolete and
    deliberately dropped.
    """
    atom_numbers = set(molecule.atoms_numbers)
    pending: list[_AtomStereoDescriptor | _CisTransStereoDescriptor] = [
        descriptor
        for descriptor in atom_descriptors
        if descriptor.center in atom_numbers
        and set(descriptor.environment).issubset(atom_numbers)
    ]
    pending.extend(
        descriptor
        for descriptor in bond_descriptors
        if {
            descriptor.first_terminal,
            descriptor.second_terminal,
            descriptor.first_neighbor,
            descriptor.second_neighbor,
        }.issubset(atom_numbers)
    )

    while pending:
        unresolved: list[_AtomStereoDescriptor | _CisTransStereoDescriptor] = []
        applied = False
        for descriptor in pending:
            try:
                if isinstance(descriptor, _AtomStereoDescriptor):
                    molecule.add_atom_stereo(
                        descriptor.center,
                        descriptor.environment,
                        descriptor.mark,
                        clean_cache=False,
                    )
                else:
                    molecule.add_cis_trans_stereo(
                        descriptor.first_terminal,
                        descriptor.second_terminal,
                        descriptor.first_neighbor,
                        descriptor.second_neighbor,
                        descriptor.mark,
                        clean_cache=False,
                    )
            except NotChiral:
                unresolved.append(descriptor)
            except (IsChiral, KeyError, ValueError, ValenceError):
                # Changed local environments and duplicate marks are not transferable.
                continue
            applied = True

        if not applied:
            break
        molecule.flush_stereo_cache()
        pending = unresolved

    molecule.fix_stereo()


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

    def _patcher(
        self, structure: MoleculeContainer, mapping: dict[int, int]
    ) -> MoleculeContainer:
        new = super()._patcher(structure, mapping)

        # Bug-6 protection: snapshot pre-kekule aromatic atoms.
        pre_aromatic = {n for n, a in new.atoms() if a.hybridization == 4}
        snapshot = (
            snapshot_aromaticity_subset(new, pre_aromatic) if pre_aromatic else None
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
            # Retain unaffected, valid stereocentres in generated precursors while
            # removing only stereo marks invalidated by the transformation.
            new.fix_stereo()
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
    co_reactants: tuple[MoleculeContainer, ...] = (),
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
    :param co_reactants: Extra structures handed to the rule alongside
        ``molecule``. A retro rule is unimolecular on its reactant side, so the
        default empty tuple is the whole of retrosynthesis; a **forward**
        bimolecular rule needs both partners at once and matches nothing when
        handed one structure — it yields zero reactions, silently. Partner
        *selection* is the caller's problem: this only forwards what it is
        given. No caller supplies it today — in particular
        :class:`~synplan.mcts.tree.Tree` expands without it, so a
        ``direction="forward"`` search runs unimolecular rules only.
    :return: An iterator yielding the products of reaction rule application.
    :raises TypeError: if ``molecule`` carries synthon labels.
        ``QueryElement.__eq__`` never consults ``_label``, so a plain reactor
        matches a labelled synthon and emits unlabelled products — the labels
        vanish with no error. A caller holding a labelled synthon must strip
        the labels or stop, not expand it here.
    """
    if isinstance(molecule, SynthonContainer) and molecule.synthon_labels:
        raise TypeError(
            f"refusing to apply a label-blind reaction rule to the labelled "
            f"synthon {molecule}: it would silently strip the labels"
        )

    def _collect_reactions(
        current_molecule: MoleculeContainer,
    ) -> list[ReactionContainer]:
        reactants = add_small_mols(current_molecule, small_molecules=co_reactants)
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
            atom_stereo, bond_stereo = _snapshot_product_stereo(reaction.products)
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
                _restore_product_stereo(c, atom_stereo, bond_stereo)
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

                reactants_key = _reactants_key(merged_reactants) if track_keys else None

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


def reaction_rules_appliance(
    molecule: MoleculeContainer, reaction_rules: list[CanonicalRetroReactor]
) -> tuple[list[int], list[int]]:
    """Applies each reaction rule from the list of reaction rules to a given molecule
    and returns the indexes of the successfully applied regular and prioritized reaction
    rules.

    :param molecule: The input molecule.
    :param reaction_rules: The list of reaction rules.
    :return: The two lists of indexes of successfully applied regular reaction rules and
        priority reaction rules.
    """

    applied_rules, priority_rules = [], []
    for i, rule in enumerate(reaction_rules):
        rule_applied = False
        rule_prioritized = False

        try:
            for reaction in rule([molecule]):
                for prod in reaction.products:
                    tmp_prod = prod.copy()
                    tmp_prod.remove_coordinate_bonds(keep_to_terminal=False)
                    tmp_prod.kekule()
                    if tmp_prod.check_valence():
                        break
                    rule_applied = True

                    # check priority rules
                    if len(reaction.products) > 1:
                        # check coupling retro manual
                        if all(len(mol) > 6 for mol in reaction.products):
                            if (
                                sum(len(mol) for mol in reaction.products)
                                - len(reaction.reactants[0])
                                < 6
                            ):
                                rule_prioritized = True
                    else:
                        # check cyclization retro manual
                        if sum(len(mol.sssr) for mol in reaction.products) < sum(
                            len(mol.sssr) for mol in reaction.reactants
                        ):
                            rule_prioritized = True
            #
            if rule_applied:
                applied_rules.append(i)
                #
                if rule_prioritized:
                    priority_rules.append(i)
        except Exception as e:
            logger.debug(e)
            continue

    return applied_rules, priority_rules
