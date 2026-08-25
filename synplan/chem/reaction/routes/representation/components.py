"""RouteCGR component classification shared by clustering and analysis."""

from __future__ import annotations

from typing import Any

from synplan.chem.utils import safe_canonicalization


def _target_atoms_from_pseudo_products(pseudo_products: Any) -> set[int]:
    products = pseudo_products.split()
    if not products:
        return set()
    target_product = min(products, key=lambda mol: min(mol._atoms))
    return set(target_product._atoms)


def route_cgr_pseudo_reactants_by_role(route_cgr: Any) -> dict[str, list[Any]]:
    """Split decomposed RouteCGR pseudo-reactants into real and supporting parts."""

    pseudo_reactants, pseudo_products = route_cgr.decompose()
    target_atoms = _target_atoms_from_pseudo_products(pseudo_products)

    result = {"real_bb": [], "supporting": []}
    if not target_atoms:
        return result

    for mol in pseudo_reactants.split():
        kind = "real_bb" if set(mol._atoms) & target_atoms else "supporting"
        # decompose() drops the implicit H on an aromatic N, so an azole comes back as
        # `c1cnc2...` and no catalogue lookup on it can ever match.
        result[kind].append(safe_canonicalization(mol))

    return result


__all__ = ["route_cgr_pseudo_reactants_by_role"]
