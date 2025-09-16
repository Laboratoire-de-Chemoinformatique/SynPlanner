"""Tests for synplan.chem.reaction_rules.extraction utilities."""

from __future__ import annotations

from typing import Iterable, Set

import pytest
from CGRtools import smiles
from CGRtools.containers import (
    CGRContainer,
    MoleculeContainer,
    QueryContainer,
    ReactionContainer,
)
from chython import smarts as sq_chy

from synplan.chem.reaction_rules.extraction import (
    add_environment_atoms,
    add_functional_groups,
    add_ring_structures,
    clean_molecules,
)
from synplan.chem.utils import cgrtools_to_chython_molecule
from synplan.utils.config import RuleExtractionConfig

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def default_config() -> RuleExtractionConfig:  # noqa: D401 – simple factory
    """Return the default rule‑extraction configuration."""
    return RuleExtractionConfig()


def _neighbours(mol: MoleculeContainer | CGRContainer, idx: int) -> Set[int]:
    """Return immediate neighbour atom numbers for *idx*.

    Implementation relies on CGRtools' private `_bonds` mapping because the
    public `Atom.neighbors` returns only a **count**.  Falls back to scanning
    `mol.bonds` if the mapping is unavailable.
    """
    neigh: set[int] = set()

    # Preferred: constant‑time lookup from the internal adjacency table.
    if hasattr(mol, "_bonds") and isinstance(mol._bonds, dict):  # type: ignore[attr-defined]
        neigh.update(mol._bonds.get(idx, {}).keys())  # type: ignore[attr-defined]
        if neigh:
            return neigh

    # Fallback: linear scan over bond objects (works for both containers).
    for bond in getattr(mol, "bonds", ()):  # type: ignore[attr-defined]
        a, b = bond.atom1.number, bond.atom2.number  # type: ignore[attr-defined]
        if a == idx:
            neigh.add(b)
        elif b == idx:
            neigh.add(a)
    return neigh


# ---------------------------------------------------------------------------
# `add_*` utilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [0, 1])
def test_add_environment_atoms(simple_cgr: CGRContainer, depth: int) -> None:
    centre = set(simple_cgr.center_atoms)
    expanded = add_environment_atoms(simple_cgr, centre, depth)
    if depth == 0:
        assert expanded == centre, "Depth 0 must echo centre atoms only"
    else:
        assert centre.issubset(expanded), "Centre atoms must be kept"
        expected = centre | {n for idx in centre for n in _neighbours(simple_cgr, idx)}
        # Implementation may include extra context atoms; ensure at least the
        # strict first shell is present.
        assert expected.issubset(expanded)


def test_add_functional_groups(
    simple_esterification_reaction: ReactionContainer,
) -> None:
    centre = {3, 4, 5, 6}
    carbonyl = sq_chy("[C]=[O]")
    r0_ch = cgrtools_to_chython_molecule(simple_esterification_reaction.reactants[0])

    expected = centre.copy()
    for mp in carbonyl.get_mapping(r0_ch):
        carbonyl.remap(mp)
        if set(carbonyl.atoms_numbers) & centre:
            expected.update(carbonyl.atoms_numbers)
        carbonyl.remap({v: k for k, v in mp.items()})

    result = add_functional_groups(simple_esterification_reaction, centre, [carbonyl])

    assert centre.issubset(result)
    assert expected.issubset(result)


def test_add_ring_structures_no_ring(simple_cgr: CGRContainer) -> None:
    centre = set(simple_cgr.center_atoms)
    assert not simple_cgr.sssr, "Fixture unexpectedly contains rings"
    assert add_ring_structures(simple_cgr, centre) == centre


def test_add_ring_structures_ring_formed(diels_alder_cgr: CGRContainer) -> None:
    centre = set(diels_alder_cgr.center_atoms)
    result = add_ring_structures(diels_alder_cgr, centre)
    ring_atoms = {
        a for ring in diels_alder_cgr.sssr if set(ring) & centre for a in ring
    }
    assert centre | ring_atoms == result


@pytest.fixture(scope="session")
def query_ethanol() -> QueryContainer:
    return smiles("CCO").substructure(smiles("CCO"), as_query=True)


def test_clean_molecules(simple_esterification_reaction: ReactionContainer) -> None:
    rxn = simple_esterification_reaction
    centre = {2, 4, 5, 6}
    rule_atoms = centre.copy()

    def _extract(mols):
        out: list[QueryContainer] = []
        for m in mols:
            sel = rule_atoms & set(m.atoms_numbers)
            if sel:
                out.append(m.substructure(atoms=sel, as_query=True))
        return out

    r_queries = _extract(rxn.reactants)
    p_queries = _extract(rxn.products)

    retention = {
        "reaction_center": {
            k: True
            for k in ("neighbors", "hybridization", "implicit_hydrogens", "ring_sizes")
        },
        "environment": {
            k: False
            for k in ("neighbors", "hybridization", "implicit_hydrogens", "ring_sizes")
        },
    }

    cleaned_r = clean_molecules(r_queries, rxn.reactants, centre, retention)
    cleaned_p = clean_molecules(p_queries, rxn.products, centre, retention)

    def _check(orig: Iterable[QueryContainer], clean: Iterable[QueryContainer]):
        for o, c in zip(orig, clean, strict=True):
            for idx in o.atoms_numbers:
                o_atom, c_atom = o.atom(idx), c.atom(idx)
                if idx in centre:
                    assert c_atom.hybridization not in ((), None)
                    assert c_atom.implicit_hydrogens not in ((), None)
                else:
                    assert c_atom.hybridization in ((), set())
                    assert c_atom.implicit_hydrogens in ((), set())

    _check(r_queries, cleaned_r)
    _check(p_queries, cleaned_p)
