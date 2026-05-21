"""Symmetry detection helpers for reaction-rule SMARTS."""

from collections.abc import Iterable
from dataclasses import dataclass

from chython import smarts as smarts_parser
from chython.containers import QueryContainer, ReactionContainer

_ORGANOMETALLIC_SYMBOLS = frozenset({"B", "Mg", "Sn", "Zn", "Cu", "Li", "Al"})
_HALOGEN_COUPLING_SYMBOLS = frozenset({"Cl", "Br", "I"})


@dataclass(frozen=True)
class _ExternalProductBond:
    """Product bond connecting a mapped target atom to a precursor handle."""

    target_atom: int
    target_symbol: str
    external_atom: int
    external_symbol: str
    orders: tuple
    molecule: QueryContainer


@dataclass(frozen=True)
class _ReactionRuleFeatures:
    """Precomputed reaction-rule features used by useful symmetry detection."""

    target_atoms: frozenset[int]
    target_has_organometallic: bool
    external_product_bonds: tuple[_ExternalProductBond, ...]
    single_or_aromatic_target_c_pairs: tuple[tuple[int, int], ...]
    alkene_target_c_pairs: tuple[tuple[int, int], ...]
    azo_target_c_pairs: tuple[tuple[int, int], ...]


def parse_reaction_rule_smarts(rule_smarts: str) -> ReactionContainer:
    """Parse a reaction-rule SMARTS string with chython.

    This is a small guard around ``chython.smarts`` for loader code that expects
    a reaction rule, not a single molecule/query SMARTS.
    """
    reaction_rule = smarts_parser(rule_smarts)
    if not isinstance(reaction_rule, ReactionContainer):
        raise ValueError("SMARTS string was not processed by chython as a reaction")
    return reaction_rule


def is_symmetric_reaction_rule(rule: str | ReactionContainer) -> int:
    """Return ``1`` when a SMARTS reaction rule has a symmetric reaction center.

    The rule is parsed with chython. Symmetry is checked on each left-hand
    reactant query graph and must move at least one atom whose bonds change
    between the left and right side of the rule. External leaving/incoming
    group identities are ignored while building the change signatures, so
    equivalent target halves that become different useful precursor handles are
    reported as symmetric.
    """
    reaction_rule = parse_reaction_rule_smarts(rule) if isinstance(rule, str) else rule

    def atom_label(query: QueryContainer, atom_number: int) -> tuple:
        atom = query.atom(atom_number)
        return (
            repr(atom),
            getattr(atom, "atomic_number", None),
            getattr(atom, "atomic_symbol", None),
            getattr(atom, "isotope", None),
            getattr(atom, "charge", None),
            getattr(atom, "is_radical", None),
            tuple(getattr(atom, "neighbors", ()) or ()),
            tuple(getattr(atom, "hybridization", ()) or ()),
            tuple(getattr(atom, "implicit_hydrogens", ()) or ()),
            tuple(getattr(atom, "ring_sizes", ()) or ()),
        )

    def bond_label(bond) -> tuple:
        return (
            tuple(getattr(bond, "order", ()) or ()),
            getattr(bond, "in_ring", None),
            getattr(bond, "stereo", None),
        )

    def bond_key(atom_1: int, atom_2: int) -> tuple[int, int]:
        return (atom_1, atom_2) if atom_1 < atom_2 else (atom_2, atom_1)

    def side_bond_labels(
        molecules: Iterable[QueryContainer],
    ) -> dict[tuple[int, int], tuple]:
        labels = {}
        for molecule in molecules:
            for atom_1, atom_2, bond in molecule.bonds():
                labels[bond_key(atom_1, atom_2)] = bond_label(bond)
        return labels

    reactant_bonds = side_bond_labels(reaction_rule.reactants)
    product_bonds = side_bond_labels(reaction_rule.products)

    def component_change_signatures(component_atoms: set[int]) -> dict[int, tuple]:
        """Return local bond-change signatures for atoms in one reactant graph."""
        signatures: dict[int, list[tuple]] = {atom: [] for atom in component_atoms}
        for atom_1, atom_2 in reactant_bonds.keys() | product_bonds.keys():
            key = (atom_1, atom_2)
            reactant_label = reactant_bonds.get(key)
            product_label = product_bonds.get(key)
            if reactant_label == product_label:
                continue
            if atom_1 in component_atoms:
                atom_2_is_internal = atom_2 in component_atoms
                signatures[atom_1].append(
                    (
                        atom_2_is_internal,
                        reactant_label if atom_2_is_internal else None,
                        product_label if atom_2_is_internal else None,
                    )
                )
            if atom_2 in component_atoms:
                atom_1_is_internal = atom_1 in component_atoms
                signatures[atom_2].append(
                    (
                        atom_1_is_internal,
                        reactant_label if atom_1_is_internal else None,
                        product_label if atom_1_is_internal else None,
                    )
                )
        return {
            atom: tuple(sorted(atom_signatures, key=repr))
            for atom, atom_signatures in signatures.items()
        }

    def has_moving_automorphism(
        query: QueryContainer,
        labels_by_atom: dict[int, tuple],
        moved_atoms: set[int],
    ) -> bool:
        """Find an exact graph automorphism that moves a changed atom."""
        atoms = tuple(query._atoms)
        bond_labels = {
            (atom_1, atom_2): bond_label(bond)
            for atom_1, neighbors in query._bonds.items()
            for atom_2, bond in neighbors.items()
        }
        colors = _compress_labels(labels_by_atom)

        for _ in range(len(atoms)):
            signatures = {
                atom: (
                    colors[atom],
                    tuple(
                        sorted(
                            (
                                (bond_labels[(atom, neighbor)], colors[neighbor])
                                for neighbor in query._bonds[atom]
                            ),
                            key=repr,
                        )
                    ),
                )
                for atom in atoms
            }
            refined = _compress_labels(signatures)
            if refined == colors:
                break
            colors = refined

        candidates_by_atom = {
            atom: tuple(other for other in atoms if colors[other] == colors[atom])
            for atom in atoms
        }
        if all(len(candidates_by_atom[atom]) == 1 for atom in moved_atoms):
            return False

        search_order = tuple(
            sorted(
                atoms,
                key=lambda atom: (
                    atom not in moved_atoms,
                    len(candidates_by_atom[atom]),
                    -len(query._bonds[atom]),
                    atom,
                ),
            )
        )
        mapping: dict[int, int] = {}
        used: set[int] = set()

        def edge_label(atom_1: int, atom_2: int) -> tuple | None:
            if atom_2 not in query._bonds[atom_1]:
                return None
            return bond_labels[(atom_1, atom_2)]

        def mapping_is_consistent(atom: int, mapped_atom: int) -> bool:
            for other_atom, mapped_other_atom in mapping.items():
                if edge_label(atom, other_atom) != edge_label(
                    mapped_atom, mapped_other_atom
                ):
                    return False
            return True

        def search(index: int, moves_tracked_atom: bool) -> bool:
            if index == len(search_order):
                return moves_tracked_atom

            atom = search_order[index]
            candidates = candidates_by_atom[atom]
            if atom in moved_atoms:
                candidates = tuple(
                    sorted(candidates, key=lambda candidate: candidate == atom)
                )

            for candidate in candidates:
                if (
                    candidate in used
                    or labels_by_atom[atom] != labels_by_atom[candidate]
                    or not mapping_is_consistent(atom, candidate)
                ):
                    continue
                mapping[atom] = candidate
                used.add(candidate)
                if search(
                    index + 1,
                    moves_tracked_atom or (atom in moved_atoms and candidate != atom),
                ):
                    return True
                used.remove(candidate)
                del mapping[atom]
            return False

        return search(0, False)

    for reactant in reaction_rule.reactants:
        component_atoms = set(reactant._atoms)
        if len(component_atoms) < 2:
            continue

        change_signatures = component_change_signatures(component_atoms)
        changed_atoms = {
            atom for atom, signature in change_signatures.items() if signature
        }
        if len(changed_atoms) < 2:
            continue

        atom_labels = {
            atom: (atom_label(reactant, atom), change_signatures[atom])
            for atom in component_atoms
        }
        if has_moving_automorphism(reactant, atom_labels, changed_atoms):
            return 1
    return 0


def is_useful_symmetric_reaction_rule(rule: str | ReactionContainer) -> bool:
    """Return True for useful symmetric rules that need match de-collapsing.

    Included families are organometallic coupling handles, mixed-halogen C-C
    coupling handles, benzaldehyde/benzyl-chloride coupling, dibromoalkene/aryl
    halide coupling, Wittig/Julia-Kocienski/Peterson-like olefinations, cross
    metathesis, azo-coupling, and nitrile/decyanation-like precursors. The
    checks are based on product-side external bonds to mapped target atoms, so
    reagent/protecting group fragments do not trigger the reactor
    automorphism-filter override.
    """
    if isinstance(rule, str):
        rule = parse_reaction_rule_smarts(rule)

    features = _reaction_rule_features(rule)

    def paired_with_carbonyl(second_partner_targets: set[int]) -> bool:
        """Check olefination-style precursors on opposite atoms of target C=C."""
        carbonyl_targets = {
            bond.target_atom
            for bond in features.external_product_bonds
            if bond.target_symbol == "C"
            and bond.external_symbol == "O"
            and 2 in bond.orders
        }
        return any(
            (atom_1 in carbonyl_targets and atom_2 in second_partner_targets)
            or (atom_2 in carbonyl_targets and atom_1 in second_partner_targets)
            for atom_1, atom_2 in features.alkene_target_c_pairs
        )

    def sulfur_has_two_double_oxygens(
        molecule: QueryContainer, atom_number: int
    ) -> bool:
        """Identify Julia-Kocienski-like sulfone handles."""
        return (
            sum(
                1
                for neighbor, bond in molecule._bonds[atom_number].items()
                if molecule.atom(neighbor).atomic_symbol == "O"
                and 2 in _bond_orders(bond)
            )
            >= 2
        )

    halogens_by_target: dict[int, set[str]] = {}
    hydrogens_by_target: set[int] = set()
    nitrogens_by_target: set[int] = set()
    for bond in features.external_product_bonds:
        if (
            bond.target_symbol == "C"
            and bond.external_symbol in _HALOGEN_COUPLING_SYMBOLS
            and 1 in bond.orders
        ):
            halogens_by_target.setdefault(bond.target_atom, set()).add(
                bond.external_symbol
            )
        if (
            bond.target_symbol == "C"
            and bond.external_symbol == "H"
            and 1 in bond.orders
        ):
            hydrogens_by_target.add(bond.target_atom)
        if (
            bond.target_symbol == "C"
            and bond.external_symbol == "N"
            and 1 in bond.orders
        ):
            nitrogens_by_target.add(bond.target_atom)

    chlorine_targets = {
        target_atom
        for target_atom, halogens in halogens_by_target.items()
        if "Cl" in halogens
    }
    bromine_targets = {
        target_atom
        for target_atom, halogens in halogens_by_target.items()
        if "Br" in halogens
    }
    mixed_halogen_handle = any(
        atom_1 in halogens_by_target
        and atom_2 in halogens_by_target
        and any(
            halogen_1 != halogen_2
            for halogen_1 in halogens_by_target[atom_1]
            for halogen_2 in halogens_by_target[atom_2]
        )
        for atom_1, atom_2 in features.single_or_aromatic_target_c_pairs
    )

    organometallic_targets = {
        bond.target_atom
        for bond in features.external_product_bonds
        if bond.target_symbol == "C"
        and bond.external_symbol in _ORGANOMETALLIC_SYMBOLS
        and 1 in bond.orders
    }
    oxygen_handle_targets = {
        bond.target_atom
        for bond in features.external_product_bonds
        if bond.target_symbol == "C" and bond.external_symbol == "O"
    }
    carbonyl_targets = {
        bond.target_atom
        for bond in features.external_product_bonds
        if bond.target_symbol == "C"
        and bond.external_symbol == "O"
        and 2 in bond.orders
    }
    organometallic_partner_targets = set(halogens_by_target) | oxygen_handle_targets
    organometallic_handle = not features.target_has_organometallic and any(
        (atom_1 in organometallic_targets and atom_2 in organometallic_partner_targets)
        or (
            atom_2 in organometallic_targets
            and atom_1 in organometallic_partner_targets
        )
        for atom_1, atom_2 in features.single_or_aromatic_target_c_pairs
    )
    benzaldehyde_benzyl_chloride_handle = any(
        (atom_1 in carbonyl_targets and atom_2 in chlorine_targets)
        or (atom_2 in carbonyl_targets and atom_1 in chlorine_targets)
        for atom_1, atom_2 in features.single_or_aromatic_target_c_pairs
    )

    brominated_alkene_targets = set()
    for atom_1, atom_2 in features.alkene_target_c_pairs:
        if atom_1 in bromine_targets and atom_2 in bromine_targets:
            brominated_alkene_targets.update((atom_1, atom_2))
    dibromoalkene_halobenzene_handle = any(
        (atom_1 in brominated_alkene_targets and atom_2 in halogens_by_target)
        or (atom_2 in brominated_alkene_targets and atom_1 in halogens_by_target)
        for atom_1, atom_2 in features.single_or_aromatic_target_c_pairs
    )

    phosphorane_targets = {
        bond.target_atom
        for bond in features.external_product_bonds
        if bond.target_symbol == "C"
        and bond.external_symbol == "P"
        and 2 in bond.orders
    }
    sulfone_targets = {
        bond.target_atom
        for bond in features.external_product_bonds
        if bond.target_symbol == "C"
        and bond.external_symbol == "S"
        and 1 in bond.orders
        and sulfur_has_two_double_oxygens(bond.molecule, bond.external_atom)
    }
    silicon_targets = {
        bond.target_atom
        for bond in features.external_product_bonds
        if bond.target_symbol == "C"
        and bond.external_symbol == "Si"
        and 1 in bond.orders
    }
    product_alkene_targets = {
        bond.target_atom
        for bond in features.external_product_bonds
        if bond.target_symbol == "C"
        and bond.external_symbol == "C"
        and 2 in bond.orders
    }
    metathesis_handle = any(
        atom_1 in product_alkene_targets and atom_2 in product_alkene_targets
        for atom_1, atom_2 in features.alkene_target_c_pairs
    )
    nitrile_decyanation_handle = any(
        bond.target_symbol == "C" and bond.external_symbol == "N" and 3 in bond.orders
        for bond in features.external_product_bonds
    )
    azo_coupling_handle = any(
        (atom_1 in nitrogens_by_target and atom_2 in hydrogens_by_target)
        or (atom_2 in nitrogens_by_target and atom_1 in hydrogens_by_target)
        for atom_1, atom_2 in features.azo_target_c_pairs
    )

    has_useful_precursor_shape = (
        organometallic_handle
        or mixed_halogen_handle
        or benzaldehyde_benzyl_chloride_handle
        or dibromoalkene_halobenzene_handle
        or paired_with_carbonyl(phosphorane_targets)
        or paired_with_carbonyl(sulfone_targets)
        or paired_with_carbonyl(silicon_targets)
        or metathesis_handle
        or azo_coupling_handle
        or nitrile_decyanation_handle
    )
    if not has_useful_precursor_shape:
        return False
    return bool(is_symmetric_reaction_rule(rule))


def _bond_orders(bond) -> tuple:
    return tuple(getattr(bond, "order", ()) or ())


def _reaction_rule_features(rule: ReactionContainer) -> _ReactionRuleFeatures:
    """Collect target atoms, target C-C pairs, and external product handles once."""
    target_atoms: set[int] = set()
    target_has_organometallic = False
    for molecule in rule.reactants:
        for atom_number, atom in molecule.atoms():
            target_atoms.add(atom_number)
            if atom.atomic_symbol in _ORGANOMETALLIC_SYMBOLS:
                target_has_organometallic = True

    external_product_bonds: list[_ExternalProductBond] = []
    for molecule in rule.products:
        for atom_1, atom_2, bond in molecule.bonds():
            atom_1_is_target = atom_1 in target_atoms
            atom_2_is_target = atom_2 in target_atoms
            if atom_1_is_target == atom_2_is_target:
                continue
            if atom_1_is_target:
                target_atom, external_atom = atom_1, atom_2
            else:
                target_atom, external_atom = atom_2, atom_1
            external_product_bonds.append(
                _ExternalProductBond(
                    target_atom=target_atom,
                    target_symbol=molecule.atom(target_atom).atomic_symbol,
                    external_atom=external_atom,
                    external_symbol=molecule.atom(external_atom).atomic_symbol,
                    orders=_bond_orders(bond),
                    molecule=molecule,
                )
            )

    single_or_aromatic_target_c_pairs: list[tuple[int, int]] = []
    alkene_target_c_pairs: list[tuple[int, int]] = []
    azo_target_c_pairs: list[tuple[int, int]] = []
    for molecule in rule.reactants:
        for atom_1, atom_2, bond in molecule.bonds():
            if atom_1 not in target_atoms or atom_2 not in target_atoms:
                continue
            if (
                molecule.atom(atom_1).atomic_symbol != "C"
                or molecule.atom(atom_2).atomic_symbol != "C"
            ):
                continue
            orders = _bond_orders(bond)
            if 1 in orders or 4 in orders:
                single_or_aromatic_target_c_pairs.append((atom_1, atom_2))
            if 2 in orders:
                alkene_target_c_pairs.append((atom_1, atom_2))

        for atom_1, atom_2, bond in molecule.bonds():
            if (
                atom_1 not in target_atoms
                or atom_2 not in target_atoms
                or molecule.atom(atom_1).atomic_symbol != "N"
                or molecule.atom(atom_2).atomic_symbol != "N"
                or 2 not in _bond_orders(bond)
            ):
                continue
            atom_1_c_neighbors = [
                neighbor
                for neighbor, neighbor_bond in molecule._bonds[atom_1].items()
                if neighbor != atom_2
                and neighbor in target_atoms
                and molecule.atom(neighbor).atomic_symbol == "C"
                and 1 in _bond_orders(neighbor_bond)
            ]
            atom_2_c_neighbors = [
                neighbor
                for neighbor, neighbor_bond in molecule._bonds[atom_2].items()
                if neighbor != atom_1
                and neighbor in target_atoms
                and molecule.atom(neighbor).atomic_symbol == "C"
                and 1 in _bond_orders(neighbor_bond)
            ]
            azo_target_c_pairs.extend(
                (atom_1_c_neighbor, atom_2_c_neighbor)
                for atom_1_c_neighbor in atom_1_c_neighbors
                for atom_2_c_neighbor in atom_2_c_neighbors
            )

    return _ReactionRuleFeatures(
        target_atoms=frozenset(target_atoms),
        target_has_organometallic=target_has_organometallic,
        external_product_bonds=tuple(external_product_bonds),
        single_or_aromatic_target_c_pairs=tuple(single_or_aromatic_target_c_pairs),
        alkene_target_c_pairs=tuple(alkene_target_c_pairs),
        azo_target_c_pairs=tuple(azo_target_c_pairs),
    )


def _compress_labels(labels: dict[int, tuple]) -> dict[int, int]:
    """Replace structural labels with dense integer ids."""
    label_to_order = {
        label: index
        for index, label in enumerate(sorted(set(labels.values()), key=repr))
    }
    return {atom: label_to_order[label] for atom, label in labels.items()}


__all__ = [
    "is_symmetric_reaction_rule",
    "is_useful_symmetric_reaction_rule",
    "parse_reaction_rule_smarts",
]
