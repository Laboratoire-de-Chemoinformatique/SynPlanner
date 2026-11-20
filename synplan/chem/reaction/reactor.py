"""Module containing classes and functions for manipulating reactions and reaction
rules."""

import logging
from collections.abc import Iterator
from contextvars import ContextVar
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any

from chython.containers import MoleculeContainer, ReactionContainer, SynthonContainer
from chython.exceptions import InvalidAromaticRing
from chython.periodictable import AnyElement
from chython.reactor import Reactor

from synplan.chem.mapping import (
    MappingBudget,
    MappingBudgetExceeded,
    bounded_query,
    mapping_budget,
)
from synplan.chem.stereo import (
    assert_stereo_preserved,
    assign_stereo,
    has_stereo,
    stereo_requirements,
)
from synplan.chem.utils import safe_canonicalization, validate_and_canonicalize

logger = logging.getLogger(__name__)

ROUTE_ATOM_MAPPING_CONTRACT = "route-origin-v1"
ATOM_ORIGINS_META = "_synplan_atom_origins"
LEGACY_ATOM_MAPS_META = "_synplan_legacy_atom_maps"
ATOM_MAPPING_STATE_META = "_synplan_atom_mapping_state"
_HAS_ISOTOPES_META = "_synplan_has_isotopes"


@dataclass(frozen=True, slots=True)
class AtomMappingState:
    """Route-lineage allocation ceilings carried by an MCTS node."""

    map_ceiling: int
    origin_ceiling: int


@dataclass(frozen=True, slots=True)
class _NativeMappingCall:
    state: AtomMappingState
    origins: dict[int, int]
    legacy_maps: dict[int, int]


@dataclass(frozen=True, slots=True)
class _NativePatch:
    state: AtomMappingState
    origins: dict[int, int]
    legacy_maps: dict[int, int]


_ACTIVE_NATIVE_MAPPING: ContextVar[_NativeMappingCall | None] = ContextVar(
    "synplan_native_mapping_call", default=None
)
_ACTIVE_NATIVE_PATCH: ContextVar[_NativePatch | None] = ContextVar(
    "synplan_native_mapping_patch", default=None
)


def initialize_atom_origins(molecule: MoleculeContainer) -> AtomMappingState:
    """Attach deterministic root provenance and return its initial ceilings."""

    atom_numbers = sorted(molecule._atoms)
    origins = {
        atom_number: origin for origin, atom_number in enumerate(atom_numbers, 1)
    }
    molecule.meta[ATOM_ORIGINS_META] = origins
    molecule.meta.pop(LEGACY_ATOM_MAPS_META, None)
    state = AtomMappingState(max(atom_numbers, default=0), len(atom_numbers))
    molecule.meta[ATOM_MAPPING_STATE_META] = (
        state.map_ceiling,
        state.origin_ceiling,
    )
    molecule.meta[_HAS_ISOTOPES_META] = any(
        atom.isotope is not None for atom in molecule._atoms.values()
    )
    return state


def atom_mapping_state_from_molecules(
    molecules: Iterator[MoleculeContainer] | list[MoleculeContainer] | tuple,
    minimum: AtomMappingState | None = None,
) -> AtomMappingState:
    """Return the highest map/origin ceilings represented by ``molecules``."""

    map_ceiling = minimum.map_ceiling if minimum is not None else 0
    origin_ceiling = minimum.origin_ceiling if minimum is not None else 0
    for molecule in molecules:
        stored = molecule.meta.get(ATOM_MAPPING_STATE_META)
        if stored is not None:
            map_ceiling = max(map_ceiling, int(stored[0]))
            origin_ceiling = max(origin_ceiling, int(stored[1]))
            continue
        map_ceiling = max(map_ceiling, max(molecule._atoms, default=0))
        origins = molecule.meta.get(ATOM_ORIGINS_META, {})
        origin_ceiling = max(origin_ceiling, max(origins.values(), default=0))
    return AtomMappingState(map_ceiling, origin_ceiling)


def copy_atom_mapping_metadata(
    source: MoleculeContainer, target: MoleculeContainer
) -> None:
    """Copy native mapping metadata without sharing mutable nested dictionaries."""

    if ATOM_ORIGINS_META in source.meta:
        target.meta[ATOM_ORIGINS_META] = dict(source.meta[ATOM_ORIGINS_META])
    else:
        target.meta.pop(ATOM_ORIGINS_META, None)
    if LEGACY_ATOM_MAPS_META in source.meta:
        target.meta[LEGACY_ATOM_MAPS_META] = dict(source.meta[LEGACY_ATOM_MAPS_META])
    else:
        target.meta.pop(LEGACY_ATOM_MAPS_META, None)
    if ATOM_MAPPING_STATE_META in source.meta:
        target.meta[ATOM_MAPPING_STATE_META] = tuple(
            source.meta[ATOM_MAPPING_STATE_META]
        )
    else:
        target.meta.pop(ATOM_MAPPING_STATE_META, None)
    if _HAS_ISOTOPES_META in source.meta:
        target.meta[_HAS_ISOTOPES_META] = source.meta[_HAS_ISOTOPES_META]
    else:
        target.meta.pop(_HAS_ISOTOPES_META, None)


class Reaction(ReactionContainer):
    """Reaction class used for a general representation of reaction."""


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

    def __init__(self, *args, reconcile_mapping: bool = False, **kwargs):
        if type(reconcile_mapping) is not bool:
            raise TypeError("reconcile_mapping should be a bool")
        kwargs["fix_tautomers"] = True
        kwargs["fix_aromatic_rings"] = False  # we run all aromatization in _patcher
        if args:
            args = (tuple(bounded_query(q) for q in args[0]), *args[1:])
        elif "patterns" in kwargs:
            kwargs["patterns"] = tuple(bounded_query(q) for q in kwargs["patterns"])
        super().__init__(*args, **kwargs)
        if reconcile_mapping:
            self._atom_identity_plan = self._classify_atom_identities()
            self._preserves_atom_identities = (
                self._atom_identity_plan is not None
                and not any(self._atom_identity_plan)
            )
            self._preserves_nonisotopic_atom_identities = (
                self._atom_identity_plan is not None
                and not any(self._atom_identity_plan[:2])
                and all(
                    product._atoms[position].isotope is None
                    for position in self._atom_identity_plan[2]
                    for product in self._products
                    if product.has_atom(position)
                )
            )

    @property
    def reconcile_mapping(self) -> bool:
        """Whether this reactor carries the opt-in native mapping plan."""

        return hasattr(self, "_atom_identity_plan")

    def _classify_atom_identities(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
        """Compile template positions needing a new atom identity."""

        reactant_atoms = {}
        for pattern in self._patterns:
            for atom_number, atom in pattern.atoms():
                if atom_number in reactant_atoms:
                    return None
                reactant_atoms[atom_number] = atom

        product_only_atoms = set()
        identity_changing_atoms = set()
        identity_sensitive_atoms = set()
        product_atoms = set()
        for product in self._products:
            for atom_number, product_atom in product.atoms():
                if atom_number in product_atoms:
                    return None
                product_atoms.add(atom_number)
                reactant_atom = reactant_atoms.get(atom_number)
                if reactant_atom is None:
                    if isinstance(product_atom, AnyElement):
                        return None
                    product_only_atoms.add(atom_number)
                    continue
                if isinstance(product_atom, AnyElement):
                    continue
                reactant_element = getattr(reactant_atom, "atomic_number", None)
                product_element = getattr(product_atom, "atomic_number", None)
                if not isinstance(reactant_element, int):
                    return None
                if reactant_element != product_element:
                    identity_changing_atoms.add(atom_number)
                    continue
                reactant_isotope = getattr(reactant_atom, "isotope", None)
                product_isotope = getattr(product_atom, "isotope", None)
                if reactant_isotope is not None and product_isotope is not None:
                    if reactant_isotope != product_isotope:
                        identity_changing_atoms.add(atom_number)
                elif reactant_isotope != product_isotope or product_isotope is None:
                    identity_sensitive_atoms.add(atom_number)
        return (
            tuple(sorted(product_only_atoms)),
            tuple(sorted(identity_changing_atoms)),
            tuple(sorted(identity_sensitive_atoms)),
        )

    def __call__(
        self,
        *structures: MoleculeContainer,
        atom_mapping_state: AtomMappingState | None = None,
        _include_reactant_provenance: bool = True,
    ) -> Iterator[ReactionContainer]:
        """Apply the rule, optionally allocating route-lineage atom identities."""

        if not self.reconcile_mapping:
            if atom_mapping_state is not None:
                raise ValueError("atom_mapping_state requires reconcile_mapping=True")
            return super().__call__(*structures)
        return self._iterate_native_mapping(
            structures,
            atom_mapping_state,
            include_reactant_provenance=_include_reactant_provenance,
        )

    def _iterate_native_mapping(
        self,
        structures: tuple[MoleculeContainer, ...],
        atom_mapping_state: AtomMappingState | None,
        *,
        include_reactant_provenance: bool,
    ) -> Iterator[ReactionContainer]:
        if not structures:
            return
        if (
            self._can_preserve_atom_identities(structures)
            and not include_reactant_provenance
            and atom_mapping_state is not None
            and len(structures) == 1
            and ATOM_ORIGINS_META in structures[0].meta
        ):
            # The MCTS hot path already owns validated provenance.  Avoid the
            # native context and allocator entirely, so non-matching Top-50
            # rules differ from baseline only by this constant-time branch.
            yield from self._iterate_identity_preserving(
                structures,
                atom_mapping_state,
                structures[0].meta[ATOM_ORIGINS_META],
                structures[0].meta.get(LEGACY_ATOM_MAPS_META, {}),
                include_reactant_provenance=False,
            )
            return
        # ``apply_reaction_rule`` requests no reactant provenance on the MCTS
        # hot path.  Such inputs are node-owned native molecules whose complete
        # metadata and ceilings were established at root creation or by the
        # preceding reactor patch.  Reuse those immutable dictionaries instead
        # of rebuilding sets and recomputing maxima for every non-matching
        # Top-50 rule (the dominant case in search).
        if (
            not include_reactant_provenance
            and atom_mapping_state is not None
            and len(structures) == 1
            and ATOM_ORIGINS_META in structures[0].meta
        ):
            structure = structures[0]
            context = _NativeMappingCall(
                atom_mapping_state,
                structure.meta[ATOM_ORIGINS_META],
                structure.meta.get(LEGACY_ATOM_MAPS_META, {}),
            )
            yield from self._iterate_with_native_context(
                structures,
                context,
                include_reactant_provenance=False,
            )
            return

        combined_origins: dict[int, int] = {}
        combined_legacy: dict[int, int] = {}
        occupied_maps: set[int] = set()
        metadata_floor = atom_mapping_state_from_molecules(structures)
        allocation_floor = atom_mapping_state or metadata_floor
        next_origin = allocation_floor.origin_ceiling
        for structure in structures:
            overlap = occupied_maps.intersection(structure._atoms)
            if overlap:
                raise ValueError(
                    "native mapping requires disjoint input atom maps; "
                    f"overlap={sorted(overlap)[:10]}"
                )
            occupied_maps.update(structure._atoms)
            origins = structure.meta.get(ATOM_ORIGINS_META)
            if origins is None:
                origins = {}
                for atom_number in sorted(structure._atoms):
                    next_origin += 1
                    origins[atom_number] = next_origin
            elif set(origins) != set(structure._atoms):
                missing = sorted(set(structure._atoms) - set(origins))
                extra = sorted(set(origins) - set(structure._atoms))
                raise ValueError(
                    "incomplete native atom-origin metadata: "
                    f"missing={missing[:10]}, extra={extra[:10]}"
                )
            combined_origins.update(origins)
            combined_legacy.update(structure.meta.get(LEGACY_ATOM_MAPS_META, {}))

        observed = AtomMappingState(
            max(occupied_maps, default=0),
            max(combined_origins.values(), default=0),
        )
        required = AtomMappingState(
            max(metadata_floor.map_ceiling, observed.map_ceiling),
            max(metadata_floor.origin_ceiling, observed.origin_ceiling),
        )
        state = atom_mapping_state or required
        if (
            state.map_ceiling < required.map_ceiling
            or state.origin_ceiling < required.origin_ceiling
        ):
            raise ValueError(
                "atom_mapping_state ceilings are below the input provenance: "
                f"state={state}, required={required}"
            )
        if len(set(combined_origins.values())) != len(combined_origins):
            raise ValueError("native input maps do not have unique atom origins")

        context = _NativeMappingCall(state, combined_origins, combined_legacy)
        if self._can_preserve_atom_identities(structures):
            yield from self._iterate_identity_preserving(
                structures,
                context.state,
                context.origins,
                context.legacy_maps,
                include_reactant_provenance=include_reactant_provenance,
            )
        else:
            yield from self._iterate_with_native_context(
                structures,
                context,
                include_reactant_provenance=include_reactant_provenance,
            )

    def _can_preserve_atom_identities(
        self, structures: tuple[MoleculeContainer, ...]
    ) -> bool:
        if self._preserves_atom_identities:
            return True
        if not self._preserves_nonisotopic_atom_identities:
            return False
        for structure in structures:
            has_isotopes = structure.meta.get(_HAS_ISOTOPES_META)
            if has_isotopes is None:
                has_isotopes = any(
                    atom.isotope is not None for atom in structure._atoms.values()
                )
                structure.meta[_HAS_ISOTOPES_META] = has_isotopes
            if has_isotopes:
                return False
        return True

    @staticmethod
    def _attach_preserved_provenance(
        molecule: MoleculeContainer,
        state: AtomMappingState,
        origins: dict[int, int],
        legacy_maps: dict[int, int],
    ) -> None:
        if molecule._atoms.keys() == origins.keys():
            molecule_origins = origins
        else:
            molecule_origins = {
                atom_number: origins[atom_number] for atom_number in molecule
            }
        molecule.meta[ATOM_ORIGINS_META] = molecule_origins
        molecule.meta[ATOM_MAPPING_STATE_META] = (
            state.map_ceiling,
            state.origin_ceiling,
        )
        molecule.meta[_HAS_ISOTOPES_META] = any(
            atom.isotope is not None for atom in molecule._atoms.values()
        )
        if legacy_maps:
            molecule_legacy = (
                legacy_maps
                if molecule._atoms.keys() == origins.keys()
                else {
                    atom_number: legacy_maps[atom_number]
                    for atom_number in molecule
                    if atom_number in legacy_maps
                }
            )
            if molecule_legacy:
                molecule.meta[LEGACY_ATOM_MAPS_META] = molecule_legacy
            else:
                molecule.meta.pop(LEGACY_ATOM_MAPS_META, None)
        else:
            molecule.meta.pop(LEGACY_ATOM_MAPS_META, None)

    def _iterate_identity_preserving(
        self,
        structures: tuple[MoleculeContainer, ...],
        state: AtomMappingState,
        origins: dict[int, int],
        legacy_maps: dict[int, int],
        *,
        include_reactant_provenance: bool,
    ) -> Iterator[ReactionContainer]:
        """Run an identity-preserving rule on Chython's baseline path."""

        for reaction in super().__call__(*structures):
            if include_reactant_provenance:
                for reactant in reaction.reactants:
                    self._attach_preserved_provenance(
                        reactant, state, origins, legacy_maps
                    )
            for product in reaction.products:
                self._attach_preserved_provenance(product, state, origins, legacy_maps)
            yield reaction

    def _iterate_with_native_context(
        self,
        structures: tuple[MoleculeContainer, ...],
        context: _NativeMappingCall,
        *,
        include_reactant_provenance: bool,
    ) -> Iterator[ReactionContainer]:
        """Run Chython while exposing one validated lineage context to patches."""

        reactions = iter(super().__call__(*structures))
        while True:
            call_token = _ACTIVE_NATIVE_MAPPING.set(context)
            patch_token = _ACTIVE_NATIVE_PATCH.set(None)
            try:
                reaction = next(reactions)
                split_patch = _ACTIVE_NATIVE_PATCH.get()
            except StopIteration:
                return
            finally:
                _ACTIVE_NATIVE_PATCH.reset(patch_token)
                _ACTIVE_NATIVE_MAPPING.reset(call_token)
            if split_patch is not None:
                # Chython can contract split ions after _single_stage yields.
                # A salt inherits the first ion's metadata but contains both graphs.
                for product in reaction.products:
                    if len(product) == len(product.meta.get(ATOM_ORIGINS_META, ())):
                        continue
                    try:
                        self._attach_preserved_provenance(
                            product,
                            split_patch.state,
                            split_patch.origins,
                            split_patch.legacy_maps,
                        )
                    except KeyError as error:
                        raise RuntimeError(
                            f"missing origin for contracted product atom {error.args[0]}"
                        ) from error
                    if len(set(product.meta[ATOM_ORIGINS_META].values())) != len(
                        product
                    ):
                        raise RuntimeError(
                            "conflicting atom origins in contracted product"
                        )
            if include_reactant_provenance:
                for reactant in reaction.reactants:
                    reactant.meta[ATOM_ORIGINS_META] = {
                        atom_number: context.origins[atom_number]
                        for atom_number in reactant
                        if atom_number in context.origins
                    }
                    legacy = {
                        atom_number: context.legacy_maps[atom_number]
                        for atom_number in reactant
                        if atom_number in context.legacy_maps
                    }
                    if legacy:
                        reactant.meta[LEGACY_ATOM_MAPS_META] = legacy
            yield reaction

    def _single_stage(self, chosen, ignored):
        if not self.reconcile_mapping or self._can_preserve_atom_identities(chosen):
            yield from super()._single_stage(chosen, ignored)
            return

        for products in super()._single_stage(chosen, ignored):
            patch = _ACTIVE_NATIVE_PATCH.get()
            _ACTIVE_NATIVE_PATCH.set(None)
            if patch is None:
                raise RuntimeError("native reactor patch provenance was not captured")

            virtual_components = []
            single_product = len(products) == 1
            for product in products:
                product.meta[ATOM_ORIGINS_META] = (
                    patch.origins
                    if single_product
                    else {
                        atom_number: patch.origins[atom_number]
                        for atom_number in product
                    }
                )
                product.meta[ATOM_MAPPING_STATE_META] = (
                    patch.state.map_ceiling,
                    patch.state.origin_ceiling,
                )
                product.meta[_HAS_ISOTOPES_META] = any(
                    atom.isotope is not None for atom in product._atoms.values()
                )
                legacy = (
                    patch.legacy_maps
                    if single_product
                    else {
                        atom_number: patch.legacy_maps[atom_number]
                        for atom_number in product
                        if atom_number in patch.legacy_maps
                    }
                )
                if legacy:
                    product.meta[LEGACY_ATOM_MAPS_META] = legacy
                else:
                    product.meta.pop(LEGACY_ATOM_MAPS_META, None)
                if not single_product:
                    virtual_components.append(
                        {
                            patch.legacy_maps.get(atom_number, atom_number)
                            for atom_number in product
                        }
                    )

            if len(products) > 1:
                component_by_atom = {
                    atom_number: component_index
                    for component_index, component in enumerate(virtual_components)
                    for atom_number in component
                }
                remaining_atoms = set(component_by_atom)
                component_order = []
                while remaining_atoms:
                    component_index = component_by_atom[remaining_atoms.pop()]
                    component_order.append(component_index)
                    remaining_atoms.difference_update(
                        virtual_components[component_index]
                    )
                products = [products[index] for index in component_order]
                _ACTIVE_NATIVE_PATCH.set(patch)
            yield products

    def _patcher(
        self, structure: MoleculeContainer, mapping: dict[int, int]
    ) -> MoleculeContainer:
        from synplan.chem.mapping import backend_preparation

        with backend_preparation():
            return self.canonical_patch(structure, mapping)

    def canonical_patch(self, structure, mapping):
        new = super()._patcher(structure, mapping)
        if self.reconcile_mapping and not self._can_preserve_atom_identities(
            (structure,)
        ):
            self._allocate_native_mapping(structure, new, mapping)
        patched = new.copy() if has_stereo(new) else None

        new.kekule(ignore_pyrrole_hydrogen=self._fix_broken_pyrroles)

        if new.check_valence():
            # ValenceError would escape; InvalidAromaticRing is caught.
            raise InvalidAromaticRing("patched molecule has invalid valence")

        new.standardize(_fix_stereo=False)
        new.implicify_hydrogens(_fix_stereo=False)
        if not new.thiele(fix_tautomers=self._fix_tautomers):
            new.fix_stereo()
        new.standardize_charges(prepare_molecule=False)
        new.standardize_tautomers(prepare_molecule=False)
        if patched is not None:
            assert_stereo_preserved(patched, new)

        return new

    def _allocate_native_mapping(self, structure, product, mapping) -> None:
        """Allocate map/origin IDs after patching and before canonicalization."""

        call = _ACTIVE_NATIVE_MAPPING.get()
        if call is None:
            raise RuntimeError("native mapping context is missing")
        plan = self._atom_identity_plan
        if plan is None:
            reactant_identities = {
                atom_number: (atom.atomic_number, atom.isotope)
                for atom_number, atom in structure.atoms()
            }
            fresh_old_maps = sorted(
                atom_number
                for atom_number, atom in product.atoms()
                if reactant_identities.get(atom_number)
                != (atom.atomic_number, atom.isotope)
            )
        else:
            product_only_positions, always_fresh, identity_sensitive = plan
            fresh_template_positions = [*product_only_positions, *always_fresh]
            for template_position in identity_sensitive:
                atom_number = mapping.get(template_position)
                if atom_number is None or not product.has_atom(atom_number):
                    continue
                source_atom = structure._atoms[atom_number]
                product_atom = product._atoms[atom_number]
                if (source_atom.atomic_number, source_atom.isotope) != (
                    product_atom.atomic_number,
                    product_atom.isotope,
                ):
                    fresh_template_positions.append(template_position)
            fresh_old_maps = []
            for template_position in sorted(fresh_template_positions):
                atom_number = mapping.get(template_position)
                if atom_number is not None and atom_number not in fresh_old_maps:
                    fresh_old_maps.append(atom_number)

        if plan is None:
            product_only = sorted(
                atom_number
                for atom_number in product._atoms
                if atom_number not in structure._atoms
            )
        else:
            product_only = sorted(
                mapping[template_position]
                for template_position in product_only_positions
            )

        if not fresh_old_maps and product._atoms.keys() == structure._atoms.keys():
            # Bond/charge/H/stereo-only rules preserve every atom identity.  The
            # lineage mappings are treated as immutable inside the search, so a
            # single-product outcome can reuse them instead of allocating and
            # populating two per-atom dictionaries for every candidate.  Split
            # products are still given independent exact subsets in
            # ``_single_stage`` below.
            _ACTIVE_NATIVE_PATCH.set(
                _NativePatch(call.state, call.origins, call.legacy_maps)
            )
            return

        if not fresh_old_maps and not call.legacy_maps:
            # The overwhelmingly common rule changes only bonds/charge/H/stereo.
            # Its route maps and origins are already final; construct the one
            # required molecule-local provenance dictionary and stop here.
            origins = {
                atom_number: call.origins[atom_number] for atom_number in product
            }
            _ACTIVE_NATIVE_PATCH.set(_NativePatch(call.state, origins, {}))
            return

        map_ceiling = call.state.map_ceiling
        origin_ceiling = call.state.origin_ceiling
        route_remap = {}
        fresh_origins = {}
        for old_atom_number in fresh_old_maps:
            map_ceiling += 1
            origin_ceiling += 1
            # Chython already allocates above the current molecule maximum.
            # At the start of a lineage that is exactly our desired route ID;
            # avoid an identity ``remap`` (which still flushes every molecular
            # cache) and remap only when a sibling/historical ceiling is higher.
            if old_atom_number != map_ceiling:
                route_remap[old_atom_number] = map_ceiling
            fresh_origins[old_atom_number] = origin_ceiling

        legacy_floor = (
            max(
                (
                    call.legacy_maps.get(atom_number, atom_number)
                    for atom_number in structure
                ),
                default=0,
            )
            if call.legacy_maps
            else max(structure._atoms, default=0)
        )
        product_only_legacy = {
            atom_number: legacy_floor + offset
            for offset, atom_number in enumerate(product_only, 1)
        }
        origins = {}
        legacy_maps = {}
        for old_atom_number in tuple(product._atoms):
            route_atom_number = route_remap.get(old_atom_number, old_atom_number)
            if old_atom_number in fresh_origins:
                origin = fresh_origins[old_atom_number]
            else:
                try:
                    origin = call.origins[old_atom_number]
                except KeyError as error:
                    raise ValueError(
                        f"missing origin for preserved atom map {old_atom_number}"
                    ) from error
            origins[route_atom_number] = origin
            if old_atom_number in structure._atoms:
                legacy_atom_number = call.legacy_maps.get(
                    old_atom_number, old_atom_number
                )
            else:
                legacy_atom_number = product_only_legacy[old_atom_number]
            if legacy_atom_number != route_atom_number:
                legacy_maps[route_atom_number] = legacy_atom_number

        if route_remap:
            product.remap(route_remap)
        state = AtomMappingState(map_ceiling, origin_ceiling)
        _ACTIVE_NATIVE_PATCH.set(_NativePatch(state, origins, legacy_maps))


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


def stereo_free_rule(rule):
    """Project a reactor once per tree, preserving its non-stereo options.

    Chython exposes no reactor projection API; its cached replacement must be
    cleared alongside the input/output templates. Never mutate shared rules.
    """
    if not any(has_stereo(q) for q in (*rule._patterns, *rule._products)):
        return rule
    rule = deepcopy(rule)
    for query in (*rule._patterns, *rule._products, rule._replacement):
        if isinstance(query, MoleculeContainer):
            query.clean_stereo()
            continue
        for _, atom in query.atoms():
            atom._stereo = None
        for *_, bond in query.bonds():
            bond._stereo = None
        query.flush_cache()
    return rule


def apply_reaction_rule(
    molecule: MoleculeContainer,
    reaction_rule: "CanonicalRetroReactor",
    sort_reactions: bool = False,
    top_reactions_num: int = 5,
    rebuild_with_cgr: bool = False,
    multirule: bool = False,
    rm_dup: bool = False,
    co_reactants: tuple[MoleculeContainer, ...] = (),
    max_mapping_work: int = 100_000,
    diagnostics: list[dict] | None = None,
    atom_mapping_state: AtomMappingState | None = None,
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

    work_budget = MappingBudget(max_mapping_work)
    if isinstance(reaction_rule, Reactor) and not isinstance(
        reaction_rule, CanonicalRetroReactor
    ):
        reaction_rule = copy(reaction_rule)
        reaction_rule._patterns = tuple(
            bounded_query(q) for q in reaction_rule._patterns
        )

    def _collect_reactions(
        current_molecule: MoleculeContainer,
        current_mapping_state: AtomMappingState | None,
    ) -> list[ReactionContainer]:
        reactants = add_small_mols(current_molecule, small_molecules=co_reactants)
        reactions = []
        try:
            # The budget applies before any successful match/product is yielded.
            with mapping_budget(max_mapping_work, budget=work_budget):
                call_kwargs = {}
                if getattr(reaction_rule, "reconcile_mapping", False):
                    call_kwargs["atom_mapping_state"] = current_mapping_state
                    call_kwargs["_include_reactant_provenance"] = False
                for reaction in reaction_rule(*reactants, **call_kwargs):
                    reactions.append(reaction)
                    if not sort_reactions and len(reactions) == top_reactions_num:
                        break
        except MappingBudgetExceeded as error:
            if diagnostics is not None:
                diagnostics.append(
                    {"reason": "mapping_budget_exceeded", "detail": str(error)}
                )
            else:
                logger.warning("Incomplete rule application: %s", error)
        except (IndexError, InvalidAromaticRing, ValueError):
            pass
        # Earlier valid outcomes survive exhaustion or a later invalid match.
        if sort_reactions:
            reactions.sort(
                key=lambda react: sum(len(m) > 6 for m in react.products), reverse=True
            )
        return reactions[:top_reactions_num]

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
            if getattr(reaction_rule, "reconcile_mapping", False):
                source_origins = {
                    atom_number: origin
                    for original in reaction.products
                    for atom_number, origin in original.meta.get(
                        ATOM_ORIGINS_META, {}
                    ).items()
                }
                source_legacy = {
                    atom_number: legacy
                    for original in reaction.products
                    for atom_number, legacy in original.meta.get(
                        LEGACY_ATOM_MAPS_META, {}
                    ).items()
                }
                source_state = atom_mapping_state_from_molecules(reaction.products)
                for mol in reactants:
                    mol.meta[ATOM_ORIGINS_META] = {
                        atom_number: source_origins[atom_number] for atom_number in mol
                    }
                    legacy = {
                        atom_number: source_legacy[atom_number]
                        for atom_number in mol
                        if atom_number in source_legacy
                    }
                    if legacy:
                        mol.meta[LEGACY_ATOM_MAPS_META] = legacy
                    mol.meta[ATOM_MAPPING_STATE_META] = (
                        source_state.map_ceiling,
                        source_state.origin_ceiling,
                    )
            canon = []
            for mol in reactants:
                # A CGR contains bond edits, not stereo. Restore the mapped
                # direct output before normalizing or accepting this recovery.
                for original in reaction.products:
                    for req in stereo_requirements(original):
                        if all(mol.has_atom(n) for n in (*req.atoms, *req.environment)):
                            assign_stereo(mol, req)
                c = validate_and_canonicalize(mol)
                if c is None:
                    return None
                c.meta.update(mol.meta)
                copy_atom_mapping_metadata(mol, c)
                canon.append(c)
            return canon

        return [mol for mol in reaction.products if len(mol) > 0]

    def _reactants_key(reactants: list[MoleculeContainer]) -> tuple[str, ...]:
        return tuple(sorted(str(reactant) for reactant in reactants))

    track_keys = rm_dup or multirule
    seen_reactants: set[tuple[str, ...]] = set()
    pending_reactants: list[tuple[list[MoleculeContainer], AtomMappingState | None]] = [
        ([molecule], atom_mapping_state)
    ]
    expanded_keys: set[tuple[str, ...]] = (
        {_reactants_key([molecule])} if multirule else set()
    )
    pending_index = 0

    while pending_index < len(pending_reactants):
        current_reactants, current_mapping_state = pending_reactants[pending_index]
        pending_index += 1

        for mol_index, current_molecule in enumerate(current_reactants):
            for reaction in _collect_reactions(current_molecule, current_mapping_state):
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
                # Only the multirule queue consumes this state.  Normal MCTS
                # expansion carries the candidate state from product metadata
                # when constructing the child, so scanning every successful
                # candidate here was duplicate work.
                next_mapping_state = (
                    atom_mapping_state_from_molecules(
                        merged_reactants, current_mapping_state
                    )
                    if multirule and getattr(reaction_rule, "reconcile_mapping", False)
                    else None
                )

                reactants_key = _reactants_key(merged_reactants) if track_keys else None

                if rm_dup and reactants_key in seen_reactants:
                    continue

                if rm_dup:
                    seen_reactants.add(reactants_key)
                _warn_if_not_canonical(merged_reactants)
                yield merged_reactants

                if multirule and reactants_key not in expanded_keys:
                    expanded_keys.add(reactants_key)
                    pending_reactants.append((merged_reactants, next_mapping_state))

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


def _warn_if_not_canonical(precursors: list[MoleculeContainer]) -> None:
    """Report a precursor this library would not have written itself.

    Off unless this module's logger is set to ``DEBUG``: the only way to tell is to
    canonicalize, and canonicalizing every precursor costs the search about a tenth
    of its time. Turn it on when hunting::

        logging.getLogger("synplan.chem.reaction.reactor").setLevel(logging.DEBUG)

    The reactor canonicalizes the whole patched molecule, but what is yielded are
    its fragments, and a fragment on its own does not always canonicalize the way
    it did inside the whole -- three per celecoxib search, all of them a ring
    hydrogen the canonicalizer puts on the other nitrogen, e.g.
    ``FC(F)(F)c1[nH]nc(-c2ccc(cc2)C)c1``. Aromatic either way, so no aromaticity
    check can see them.

    Nothing is repaired here. A repair hides where the molecule came from, and the
    point of saying so is to be able to go and look -- at a building-block file, a
    target, or a rule's own SMILES.

    Aromaticity is the usual suspect and is usually innocent. None of these is
    reported, because each comes out canonical:

    * ``OCc1cc(=O)c2ccccc2n1C`` with ``[C:1]-[O;h1:2]>>[C:1]=[O:2]`` --
      1-methyl-2-(hydroxymethyl)quinolin-4(1H)-one. The pyridinone ring comes out
      kekulized because RDKit aromatises 4-quinolinones and chython does not, so
      the input arrived claiming an aromaticity this model rejects. This is the
      case the old aromaticity restorer existed for, and restoring it produced a
      molecule ``canonicalize`` disagreed with.
    * ``FC(F)(F)c1[nH]nc(c1Br)C`` with ``[c:1]-[Br:2]>>[c:1]-[O:2]`` -- the rule
      itself turns the pyrazole into a pyrazolone, which chython does not hold
      aromatic.
    * ``FC(F)(F)c1cc(OS(=O)(=O)C(F)(F)F)n(n1)-c2ccc(cc2)S(=O)(=O)N`` losing its
      triflate -- the patcher leaves an aromatic 5-hydroxypyrazole and
      ``standardize_tautomers`` moves it to the pyrazolone.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    for molecule in precursors:
        canonical = str(safe_canonicalization(molecule))
        if str(molecule) != canonical:
            logger.debug(
                "precursor %s is not canonical; the canonicalizer writes it %s",
                molecule,
                canonical,
            )
