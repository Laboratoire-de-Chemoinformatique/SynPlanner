"""Tests for synplan.chem.reaction_rules.extraction utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

import pytest
from chython import smarts as sq_chy
from chython import smiles
from chython.containers import (
    CGRContainer,
    MoleculeContainer,
    QueryContainer,
    ReactionContainer,
)

import synplan.chem.reaction_rules.extraction as extraction
from synplan.chem.data.reaction_result import ExtractedRuleRecord, ExtractionBatchResult
from synplan.chem.reaction_rules.extraction import (
    _process_extraction_result,
    add_environment_atoms,
    add_functional_groups,
    add_ring_structures,
    clean_molecules,
    molecule_substructure_as_query,
)
from synplan.utils.config import RuleExtractionConfig

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def default_config() -> RuleExtractionConfig:
    """Return the default rule-extraction configuration."""
    return RuleExtractionConfig()


def _neighbours(mol: MoleculeContainer | CGRContainer, idx: int) -> set[int]:
    """Return immediate neighbour atom numbers for *idx*.

    Implementation relies on chython's private `_bonds` mapping because the
    public `Atom.neighbors` returns only a **count**.  Falls back to scanning
    `mol.bonds` if the mapping is unavailable.
    """
    neigh: set[int] = set()

    # Preferred: constant-time lookup from the internal adjacency table.
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
        assert expanded == centre, "Depth 0 must echo centre atoms only"
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
    r0_ch = simple_esterification_reaction.reactants[0]

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
    mol = smiles("CCO")
    return molecule_substructure_as_query(mol, mol.atoms_numbers)


def test_clean_molecules(simple_esterification_reaction: ReactionContainer) -> None:
    rxn = simple_esterification_reaction
    centre = {2, 4, 5, 6}
    rule_atoms = centre.copy()

    def _extract(mols):
        out: list[QueryContainer] = []
        for m in mols:
            sel = rule_atoms & set(m.atoms_numbers)
            if sel:
                out.append(molecule_substructure_as_query(m, sel))
        return out

    r_queries = _extract(rxn.reactants)
    p_queries = _extract(rxn.products)

    retention = {
        "reaction_center": {
            k: True for k in ("neighbors", "implicit_hydrogens", "ring_sizes")
        },
        "environment": {
            k: False for k in ("neighbors", "implicit_hydrogens", "ring_sizes")
        },
    }

    cleaned_r = clean_molecules(r_queries, rxn.reactants, centre, retention)
    cleaned_p = clean_molecules(p_queries, rxn.products, centre, retention)

    def _check(orig: Iterable[QueryContainer], clean: Iterable[QueryContainer]):
        for o, c in zip(orig, clean, strict=True):
            for idx in o.atoms_numbers:
                _o_atom, c_atom = o.atom(idx), c.atom(idx)
                if idx in centre:
                    assert c_atom.implicit_hydrogens not in ((), None)
                else:
                    assert c_atom.implicit_hydrogens in ((), set())

    _check(r_queries, cleaned_r)
    _check(p_queries, cleaned_p)


def test_process_extraction_result_uses_worker_serialized_rule(monkeypatch):
    """Parent aggregation must not parse rule SMARTS returned by workers."""

    def fail_if_parent_parses(_smarts):
        raise AssertionError("parent parsed worker rule SMARTS")

    monkeypatch.setattr(
        extraction, "parse_smarts", fail_if_parent_parses, raising=False
    )
    result = ExtractionBatchResult(
        rule_records=[
            (
                7,
                [
                    ExtractedRuleRecord(
                        cgr_key="stable-cgr",
                        rule_smarts="[C:1]>>[O:1]",
                        reactor_validation="passed",
                    )
                ],
                "CO",
            )
        ],
        errors=[],
        n_multi_product=0,
    )
    rules_statistics = defaultdict(list)
    cgr_to_rule = {}

    count = _process_extraction_result(result, rules_statistics, cgr_to_rule)

    assert count == 1
    assert rules_statistics["stable-cgr"] == [7]
    assert cgr_to_rule["stable-cgr"].rule_smarts == "[C:1]>>[O:1]"


def test_print_extraction_summary_includes_error_count(capsys):
    """The processed-reaction summary should always report failed reactions."""
    record = ExtractedRuleRecord(
        cgr_key="stable-cgr",
        rule_smarts="[C:1]>>[O:1]",
        reactor_validation="passed",
    )

    extraction._print_extraction_summary(
        n_processed=10,
        sorted_rules=[(record, [1, 2, 3])],
        filter_stats={},
        error_counts=Counter({("extract_rules", "ValueError"): 2}),
        error_file_path=None,
    )

    output = capsys.readouterr().out
    assert "Finished: processed 10, extracted 1 rules" in output
    assert "failed 2" in output
