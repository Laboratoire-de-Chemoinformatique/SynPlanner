"""Restore target stereochemistry and propagate transferable labels down a route."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from chython import smiles as smiles_parser
from chython.containers import MoleculeContainer, ReactionContainer

from synplan.chem.building_blocks.catalog import BuildingBlockCatalog
from synplan.chem.reaction.reactor import (
    _restore_product_stereo,
    _snapshot_product_stereo,
)
from synplan.chem.reaction.routes.tree_ops import iter_molecule_leaves


class RouteStereoError(ValueError):
    """Raised when stereo cannot be restored without guessing."""


def _molecule(value: object, *, location: str) -> MoleculeContainer:
    if not isinstance(value, str) or not value.strip():
        raise RouteStereoError(f"{location}: expected a non-empty molecule SMILES")
    try:
        parsed = smiles_parser(value)
    except Exception as error:
        raise RouteStereoError(f"{location}: invalid molecule SMILES") from error
    if not isinstance(parsed, MoleculeContainer):
        raise RouteStereoError(f"{location}: expected one molecule")
    return parsed


def _reaction(value: object, *, location: str) -> ReactionContainer:
    if not isinstance(value, str) or not value.strip():
        raise RouteStereoError(f"{location}: expected a non-empty reaction SMILES")
    try:
        parsed = smiles_parser(value)
    except Exception as error:
        raise RouteStereoError(f"{location}: invalid reaction SMILES") from error
    if not isinstance(parsed, ReactionContainer):
        raise RouteStereoError(f"{location}: expected one reaction")
    return parsed


def _copy_stereo(
    source: MoleculeContainer,
    destination: MoleculeContainer,
    *,
    location: str,
) -> None:
    plain_source = source.copy()
    plain_source.clean_stereo()
    plain_destination = destination.copy()
    plain_destination.clean_stereo()
    if len(plain_source) != len(plain_destination) or sum(
        1 for _ in plain_source.bonds()
    ) != sum(1 for _ in plain_destination.bonds()):
        raise RouteStereoError(f"{location}: structures are not isomorphic")
    mappings = list(
        plain_source.get_mapping(plain_destination, automorphism_filter=False)
    )
    if not mappings:
        raise RouteStereoError(f"{location}: structures are not isomorphic")

    destination.clean_stereo()
    candidates: dict[str, MoleculeContainer] = {}
    for mapping in mappings:
        remapped = source.copy()
        remapped.remap(mapping)
        candidate = destination.copy()
        descriptors = _snapshot_product_stereo((remapped,))
        _restore_product_stereo(candidate, *descriptors)
        candidates.setdefault(str(candidate), candidate)
    if len(candidates) != 1:
        raise RouteStereoError(
            f"{location}: ambiguous structural mapping gives "
            f"{len(candidates)} stereochemical assignments"
        )
    selected = next(iter(candidates.values()))
    atom_descriptors, bond_descriptors = _snapshot_product_stereo((selected,))
    _restore_product_stereo(destination, atom_descriptors, bond_descriptors)


def route_root_matches_target_stereo(
    route: Mapping[str, Any],
    target_smiles: str,
) -> bool:
    """Return whether the route root already has the target's stereo state.

    The comparison is graph-isomorphic and therefore independent of atom-map
    numbering and SMILES traversal order. An unlabelled target is authoritative
    too: a labelled route root does not match it.
    """
    if not isinstance(route, Mapping) or route.get("type") != "mol":
        raise TypeError("route must be a JSON-like molecule-root mapping")
    root = _molecule(route.get("smiles"), location="route")
    target = _molecule(target_smiles, location="target_smiles")
    expected = root.copy()
    _copy_stereo(target, expected, location="route target")
    return str(root) == str(expected)


def _propagate_molecule_node(node: dict[str, Any], *, location: str) -> None:
    molecule = _molecule(node.get("smiles"), location=location)
    children = node.get("children", [])
    if children is None:
        children = []
    if not isinstance(children, list):
        raise RouteStereoError(f"{location}.children: expected a list")
    reaction_children = [
        child
        for child in children
        if isinstance(child, Mapping) and child.get("type") == "reaction"
    ]
    if len(reaction_children) > 1:
        raise RouteStereoError(f"{location}: molecule has multiple reaction children")
    if not reaction_children:
        node["smiles"] = str(molecule)
        return

    reaction_node = reaction_children[0]
    reaction_location = f"{location}.children[{children.index(reaction_node)}]"
    reaction = _reaction(reaction_node.get("smiles"), location=reaction_location)
    product_matches = []
    for product in reaction.products:
        try:
            _copy_stereo(molecule, product, location=reaction_location)
        except RouteStereoError:
            continue
        product_matches.append(product)
    if len(product_matches) != 1:
        raise RouteStereoError(
            f"{reaction_location}: expected exactly one product matching its parent molecule"
        )

    descriptors = _snapshot_product_stereo((product_matches[0],))
    for reactant in reaction.reactants:
        _restore_product_stereo(reactant, *descriptors)

    reaction_molecule_children = reaction_node.get("children", [])
    if not isinstance(reaction_molecule_children, list):
        raise RouteStereoError(f"{reaction_location}.children: expected a list")
    if len(reaction_molecule_children) != len(reaction.reactants):
        raise RouteStereoError(
            f"{reaction_location}: reaction has {len(reaction.reactants)} reactants "
            f"but {len(reaction_molecule_children)} molecule children"
        )
    for index, (reactant, child) in enumerate(
        zip(reaction.reactants, reaction_molecule_children, strict=True)
    ):
        child_location = f"{reaction_location}.children[{index}]"
        if not isinstance(child, dict) or child.get("type") != "mol":
            raise RouteStereoError(f"{child_location}: expected a molecule node")
        child_molecule = _molecule(child.get("smiles"), location=child_location)
        _copy_stereo(reactant, child_molecule, location=child_location)
        child["smiles"] = str(child_molecule)
        _propagate_molecule_node(child, location=child_location)

    reaction_node["smiles"] = format(reaction, "m")
    node["smiles"] = str(molecule)


def _validate_building_block_leaves(
    route: dict[str, Any], catalog: BuildingBlockCatalog
) -> bool:
    mismatch_found = False
    for path, leaf in iter_molecule_leaves(route):
        if leaf.get("in_stock") is not True:
            continue
        if not isinstance(leaf, dict):
            raise RouteStereoError(f"route leaf at {path} must be mutable")
        molecule = _molecule(leaf.get("smiles"), location=f"route leaf at {path}")
        exact, candidates = catalog.validate_stereo_for_molecule(molecule)
        mismatch = not exact
        mismatch_found = mismatch_found or mismatch
        bb = leaf.get("bb")
        if bb is None:
            bb = {}
            leaf["bb"] = bb
        if not isinstance(bb, dict):
            raise RouteStereoError(f"route leaf at {path}: bb must be a mapping")
        bb["stereo_mismatch"] = mismatch
        bb["stereo_validation"] = {
            "status": (
                "matched" if exact else ("mismatch" if candidates else "not_found")
            ),
            "propagated_smiles": str(molecule),
            "catalog_smiles": list(candidates),
        }
    route["stereo_mismatch"] = mismatch_found
    return mismatch_found


def restore_route_stereo(
    route: Mapping[str, Any],
    target_smiles: str,
    *,
    catalog: BuildingBlockCatalog | None = None,
) -> dict[str, Any]:
    """Restore target stereo and propagate still-valid labels to route leaves.

    ``route`` is a v1 JSON-like route tree. ``target_smiles`` must describe the
    same root structure and carry the desired tetrahedral and/or cis/trans
    labels. Propagation follows atom mapping and transfers only descriptors that
    remain valid in each precursor.

    This is structural label propagation, not reaction-aware stereochemical
    prediction. When a mapped stereocentre remains structurally valid across a
    step, the method assumes that its target configuration can be transferred
    unchanged. It does not determine whether the reaction causes inversion,
    retention, racemization, epimerization, or stereoselective creation of a new
    stereocentre. Consequently, propagated labels at or near a reaction centre
    require independent chemical validation. Structurally invalid descriptors
    are dropped, but an affected centre that remains stereogenic is not
    automatically flagged.

    When ``catalog`` is supplied, every terminal ``in_stock`` molecule is
    checked against its exact stereo-bearing catalog identity and annotated with
    a mismatch flag. The route root receives the aggregate flag.

    The input mapping is not modified. Ambiguous or structurally inconsistent
    routes raise :class:`RouteStereoError`.
    """
    if not isinstance(route, Mapping):
        raise TypeError("route must be a JSON-like mapping")
    restored = deepcopy(dict(route))
    if restored.get("type") != "mol":
        raise RouteStereoError("route root must be a molecule node")
    root = _molecule(restored.get("smiles"), location="route")
    target = _molecule(target_smiles, location="target_smiles")
    _copy_stereo(target, root, location="route target")
    restored["smiles"] = str(root)
    _propagate_molecule_node(restored, location="route")
    if catalog is not None:
        _validate_building_block_leaves(restored, catalog)
    return restored


__all__ = [
    "RouteStereoError",
    "restore_route_stereo",
    "route_root_matches_target_stereo",
]
