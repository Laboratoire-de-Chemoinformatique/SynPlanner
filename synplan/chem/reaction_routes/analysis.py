"""Analysis helpers for RouteCGR comparison and route-derived BB usage."""

from __future__ import annotations

from typing import Any


def _target_atoms_from_pseudo_products(pseudo_products: Any) -> set[int]:
    products = pseudo_products.split()
    if not products:
        return set()
    target_product = min(products, key=lambda mol: min(mol._atoms))
    return set(target_product._atoms)


def route_cgr_pseudo_reactants_by_role(route_cgr: Any) -> dict[str, list[Any]]:
    """Split decomposed RouteCGR pseudo-reactants into real and supporting parts.

    Real pseudo-reactants contain atoms that survive into the target product
    projection. Supporting pseudo-reactants have no target-product atom overlap
    and usually correspond to PG/FGI/support fragments.
    """

    pseudo_reactants, pseudo_products = route_cgr.decompose()
    target_atoms = _target_atoms_from_pseudo_products(pseudo_products)

    result = {"real_bb": [], "supporting": []}
    if not target_atoms:
        return result

    for mol in pseudo_reactants.split():
        kind = "real_bb" if set(mol._atoms) & target_atoms else "supporting"
        result[kind].append(mol)

    return result
