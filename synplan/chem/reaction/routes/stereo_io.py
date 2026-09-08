"""Strict opt-in import for stereo auditing, preserving source reaction evidence.

This adapter does not change the legacy route reader. It rejects inconsistent
molecule/reaction nodes before object linking can erase the inconsistency.
"""

from __future__ import annotations

import re
from copy import deepcopy
from itertools import count

from synplan.chem.reaction.routes.route import Route, Step, StepOrigin
from synplan.chem.reaction.routes.stereo import (
    align_stereo,
    connectivity_key,
    molecule_mappings,
    orientation_key,
    validate_mapping,
)
from synplan.chem.stereo import (
    StereoRequirement,
    UnresolvedStereo,
    assign_stereo,
    stereo_requirements,
)
from synplan.chem.stereo import parse_smiles_preserving_stereo as smiles


def read_stereo_route(
    tree: dict, *, strip_stereo: bool = False, max_mappings: int = 256
) -> tuple[Route, dict[int, str]]:
    """Read a v1 route tree whose reaction SMILES carry explicit atom maps.

    All molecule nodes are checked against the corresponding reaction structure.
    Local reaction maps are rebased into a route-wide map, recording the rebase.
    Symmetry with different possible carbon configurations abstains. The optional
    stereo removal is for measuring recovery from connectivity proposals; source
    reaction strings remain in metadata. ``ValueError`` means unresolved import.
    """
    route = Route(
        tuple(
            read_stereo_steps(
                tree, count(1), strip_stereo=strip_stereo, max_mappings=max_mappings
            )
        )
    )
    return route, {
        i: f"{s.reaction.meta.get('source_id', 'mapped_serialization')}:{s.reaction.meta['stereo_source_path']}"
        for i, s in enumerate(route.steps)
    }


def potential_stereo_requirements(mol):
    # Include every potential centre, not just configurations present in the
    # source, so a connectivity-only serialization receives the same check.
    mol = mol.copy()
    mol.clean_stereo()
    return tuple(
        StereoRequirement(
            (n,),
            "tetrahedron",
            (n,),
            tuple(sorted(mol.stereogenic_tetrahedrons[n])),
            False,
        )
        for n in sorted(mol.chiral_tetrahedrons)
        if mol.atom(n).atomic_number == 6
    )


def read_stereo_steps(
    node,
    atom_numbers,
    *,
    shared=None,
    path="target",
    strip_stereo=False,
    max_mappings=256,
):
    children = node.get("children") or []
    reactions = [c for c in children if c.get("type") == "reaction"]
    if len(reactions) != 1:
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved",
            f"{path}: expected exactly one reaction",
        )
    source = reactions[0]
    raw = source["smiles"]
    sides = raw.split(">")
    if len(sides) != 3:
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved",
            f"{path}: invalid reaction serialization",
        )
    maps = [
        list(map(int, re.findall(r":(\d+)\]", side))) for side in (sides[0], sides[2])
    ]
    if not all(maps):
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved",
            f"{path}: reaction serialization has no explicit atom correspondence",
        )
    if any(len(ids) != len(set(ids)) or 0 in ids for ids in maps):
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved",
            f"{path}: duplicated or zero source atom maps",
        )
    reaction = smiles(raw)
    validate_mapping(reaction)
    expected = smiles(node["smiles"])
    products = [
        p
        for p in reaction.products
        if connectivity_key(p) == connectivity_key(expected)
    ]
    if len(products) != 1:
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved",
            f"{path}: molecule node differs from reaction product",
        )
    product = products[0]
    if not strip_stereo:
        reqs, _, _ = align_stereo(
            expected, product, stereo_requirements(expected), max_mappings
        )
        for req in reqs:
            assign_stereo(product, req)
    mapping = {}
    if shared is not None:
        choices = molecule_mappings(product, shared, max_mappings)
        probes = potential_stereo_requirements(product)
        signatures = {
            orientation_key(
                shared, tuple(r.remap(m) for r in probes), by_requirement=True
            )
            for m in choices
        }
        if len(signatures) != 1:
            raise UnresolvedStereo(
                "mapping_or_representation_unresolved",
                f"{path}: stereo-distinct intermediate alignments",
            )
        mapping.update(choices[0])
    all_atoms = {
        n
        for mol in (*reaction.reactants, *reaction.products, *reaction.reagents)
        for n in mol
    }
    for n in sorted(all_atoms - mapping.keys()):
        mapping[n] = next(atom_numbers)
    for mol in (*reaction.reactants, *reaction.products, *reaction.reagents):
        mol.remap({n: mapping[n] for n in mol})
        if strip_stereo:
            mol.clean_stereo()
    if shared is not None:
        if not strip_stereo:
            for req in stereo_requirements(product):
                assign_stereo(shared, req)
        # The checked structural match becomes the existing parent object.
        reaction._products = tuple(
            shared if p is product else p for p in reaction.products
        )
        product = shared
    reactant_nodes = source.get("children") or []
    available = list(range(len(reaction.reactants)))
    if len(available) != len(reactant_nodes):
        raise UnresolvedStereo(
            "mapping_or_representation_unresolved",
            f"{path}: reactant node count differs from reaction",
        )
    for child_index, child in enumerate(reactant_nodes):
        expected = smiles(child["smiles"])
        slots = [
            i
            for i in available
            if connectivity_key(reaction.reactants[i]) == connectivity_key(expected)
        ]
        if not slots:
            raise UnresolvedStereo(
                "mapping_or_representation_unresolved",
                f"{path}/{child_index}: inconsistent adjacent molecule structures",
            )
        if len(slots) > 1 and child.get("children"):
            raise UnresolvedStereo(
                "mapping_or_representation_unresolved",
                f"{path}/{child_index}: ambiguous identical precursor branches",
            )
        slot = slots[0]
        available.remove(slot)
        if child.get("children"):
            yield from read_stereo_steps(
                child,
                atom_numbers,
                shared=reaction.reactants[slot],
                path=f"{path}/{child_index}",
                strip_stereo=strip_stereo,
                max_mappings=max_mappings,
            )
    reaction.meta.update(deepcopy(source.get("meta", {})))
    reaction.meta.update(
        stereo_source_reaction=raw,
        stereo_source_path=path,
        stereo_atom_rebase=mapping,
    )
    yield Step(
        reaction,
        product,
        StepOrigin(
            rule_key=source.get("rule_key"),
            rule_source=source.get("rule_source"),
            rule_id=source.get("rule_id"),
            tree_node_id=source.get("tree_node_id"),
        ),
    )


__all__ = ["potential_stereo_requirements", "read_stereo_route", "read_stereo_steps"]
