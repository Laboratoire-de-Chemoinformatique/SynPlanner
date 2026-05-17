"""Regression test for the ``rebuild_with_cgr=True`` recovery path
inside :func:`synplan.chem.reaction.apply_reaction_rule`.

The CGR rebuild path composes the yielded reaction into a CGR and
decomposes it back to obtain mass-balanced reactants/products. This
roundtrip **bypasses** :meth:`CanonicalRetroReactor._patcher`'s
canonicalize pipeline, so the function must explicitly canonicalize
the CGR-rebuilt fragments via
:func:`synplan.chem.utils.validate_and_canonicalize`.

The contract this test asserts: for a rule whose CGR roundtrip produces
the same products as the direct patcher path,
``apply_reaction_rule(..., rebuild_with_cgr=True)`` yields precursor
sets that are **already canonical** (``str(p)`` is stable under
``safe_canonicalization``) and that match the direct-path output as
canonical-SMILES sets.
"""

from __future__ import annotations

import pytest
from chython import smarts, smiles

from synplan.chem.reaction import CanonicalRetroReactor, apply_reaction_rule
from synplan.chem.utils import safe_canonicalization

# Only bond-rearrangement rules (no element substitutions). chython's
# ``reaction.compose()`` raises ``ValueError: elements should be of the
# same type`` when a mapped atom's element changes across sides
# (e.g. Cl → O), so the CGR-rebuild path skips those reactions. Tests
# here cover only rules whose mapped atoms preserve element identity.
_RULES = [
    (
        # Amide cleavage: bond between N and C broken, no element change.
        "[N;D3:1]-[C;D3:2]=[O:3]>>[N;D2:1].[C;D3:2]=[O:3]",
        "CC(=O)N(C)C",
        "tertiary amide cleavage",
    ),
    (
        # Ester hydrolysis: same atoms, breaks C-O bond and rearranges Hs.
        "[C;D3:1](=[O:2])-[O;D2:3]-[C:4]>>[C;D3:1](=[O:2])-[O;D1:3].[C;D1:4]",
        "CC(=O)OCC",
        "ester hydrolysis on ethyl acetate",
    ),
]


def test_cgr_rebuild_silently_skips_element_substitution():
    """Rules that substitute an atom's element type (e.g. Cl → OH) cannot
    go through ``reaction.compose()`` — chython raises
    ``ValueError: elements should be of the same type``. The CGR path
    must catch that and treat the reaction as "not applicable" (return
    an empty iterator), not propagate the exception."""
    rule = smarts("[c:1]:[c:2]-[Cl;D1:3]>>[c:1]:[c:2]-[O;h1:3]")
    target = smiles("Cc1ccc(Cl)cc1")
    reactor = CanonicalRetroReactor(
        patterns=tuple(rule.reactants),
        products=tuple(rule.products),
        delete_atoms=False,
    )
    # Direct path fires.
    direct = list(apply_reaction_rule(target, reactor))
    assert direct, "sanity: direct path should fire on this substitution rule"
    # CGR-rebuild path silently produces nothing (no crash).
    rebuilt = list(apply_reaction_rule(target, reactor, rebuild_with_cgr=True))
    assert rebuilt == []


@pytest.mark.parametrize("rule_str, target_smi, label", _RULES)
def test_cgr_rebuild_yields_canonical_precursors(rule_str, target_smi, label):
    """Precursors from the CGR-rebuild path must be canonical — running
    ``safe_canonicalization`` on them again must not change ``str()``.
    This is what keeps downstream ``Precursor(canonicalize=False)``
    safe under both paths."""
    rule = smarts(rule_str)
    target = smiles(target_smi)
    reactor = CanonicalRetroReactor(
        patterns=tuple(rule.reactants),
        products=tuple(rule.products),
        delete_atoms=False,
    )
    fired = False
    for products in apply_reaction_rule(target, reactor, rebuild_with_cgr=True):
        fired = True
        for p in products:
            again = safe_canonicalization(p.copy())
            assert str(p) == str(again), (
                f"CGR-rebuilt precursor not canonical on {label!r}:\n"
                f"  yielded:    {p}\n"
                f"  re-canon:   {again}"
            )
    assert fired, f"rule did not fire on {label!r} via CGR-rebuild path"


@pytest.mark.parametrize("rule_str, target_smi, label", _RULES)
def test_cgr_rebuild_matches_direct_path(rule_str, target_smi, label):
    """The set of canonical precursor-tuples must match between the
    direct path (rebuild_with_cgr=False) and the CGR-rebuild path
    (rebuild_with_cgr=True). Both paths feed into MCTS state-dedup; a
    mismatch means the search tree depends on which flag is set."""
    rule = smarts(rule_str)
    target = smiles(target_smi)
    reactor = CanonicalRetroReactor(
        patterns=tuple(rule.reactants),
        products=tuple(rule.products),
        delete_atoms=False,
    )
    direct = sorted(
        tuple(sorted(str(p) for p in products))
        for products in apply_reaction_rule(target, reactor)
    )
    cgr_rebuilt = sorted(
        tuple(sorted(str(p) for p in products))
        for products in apply_reaction_rule(target, reactor, rebuild_with_cgr=True)
    )
    assert direct == cgr_rebuilt, (
        f"CGR-rebuild diverges from direct path on {label!r}:\n"
        f"  direct:       {direct}\n"
        f"  cgr-rebuild:  {cgr_rebuilt}"
    )
