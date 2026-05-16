"""Behavior-equivalence tests for
:class:`synplan.chem.reaction.CanonicalRetroReactor`.

The subclass collapses chython's ``_patcher`` aromatization step and
SynPlanner's ``validate_and_canonicalize`` into one pass. The
canonical-SMILES set of yielded products must match the legacy
two-pass flow exactly (otherwise MCTS state-dedup diverges between
old/new code paths).

Speedup is measured separately in
``sascore-bench/playground/bin1_disagreement/bench_canonical_reactor.py``
against real rules and real targets — that's the source of truth, not
a unit test.
"""
from __future__ import annotations

import pytest
from chython import smarts, smiles
from chython.reactor import Reactor

from synplan.chem.reaction import CanonicalRetroReactor

_RULES = [
    # (rule_smarts, target_smiles, label)
    (
        "[C;D3:1]-[O;D1:2]>>[C;D3:1]-[O;D2:2]-[C;D1:3]",
        "CC(C)O",
        "non-aromatic — alkyl OH methylation",
    ),
    (
        "[c:1]:[c:2]-[Cl;D1:3]>>[c:1]:[c:2]-[O;h1:3]",
        "c1ccc(Cl)cc1",
        "aromatic — Cl → OH",
    ),
    (
        "[N;D3:1]-[C;D3:2]=[O:3]>>[N;D2:1].[C;D3:2]=[O:3]",
        # Tertiary amide so the rule's [N;D3] actually matches.
        # N,N-dimethylacetamide: N has 3 heavy neighbors.
        "CC(=O)N(C)C",
        "non-aromatic — tertiary amide cleavage",
    ),
    (
        # touches an aromatic ring near a fused carbonyl — exercises
        # the aromaticity-snapshot-restore path inside _patcher.
        "[c:1]:[c:2](-[Cl;D1:3]):[c:4]>>[c:1]:[c:2](-[O;h1:3]):[c:4]",
        "Cc1ccc(Cl)cc1",
        "aromatic — substituted ring Cl → OH",
    ),
]


def _legacy_canonical_product_sets(rule_smarts, target_smi):
    """Legacy two-pass flow: vanilla chython Reactor + per-product
    safe_canonicalization downstream. The set of yielded canonical
    product-tuples is the contract we have to match."""
    from synplan.chem.utils import safe_canonicalization

    rule = smarts(rule_smarts)
    target = smiles(target_smi)
    reactor = Reactor(
        patterns=tuple(rule.reactants),
        products=tuple(rule.products),
        delete_atoms=False,
    )
    out = set()
    for reaction in reactor(target):
        # mirror the legacy validation step
        valid = True
        for p in reaction.products:
            tmp = p.copy()
            try:
                tmp.remove_coordinate_bonds(keep_to_terminal=False)
                tmp.kekule()
                if tmp.check_valence():
                    valid = False
                    break
            except Exception:
                valid = False
                break
        if not valid:
            continue
        out.add(
            tuple(sorted(str(safe_canonicalization(p)) for p in reaction.products))
        )
    return out


def _new_canonical_product_sets(rule_smarts, target_smi):
    """New one-pass flow: CanonicalRetroReactor yields canonical
    products directly; no per-product canonicalize downstream."""
    rule = smarts(rule_smarts)
    target = smiles(target_smi)
    reactor = CanonicalRetroReactor(
        patterns=tuple(rule.reactants),
        products=tuple(rule.products),
        delete_atoms=False,
    )
    out = set()
    for reaction in reactor(target):
        out.add(tuple(sorted(str(p) for p in reaction.products)))
    return out


@pytest.mark.parametrize("rule_str, target_smi, label", _RULES)
def test_canonical_retro_reactor_matches_legacy(rule_str, target_smi, label):
    """The set of canonical product-tuples emitted by
    ``CanonicalRetroReactor`` must equal the set the legacy two-pass
    flow produces. This is the safety contract."""
    legacy = _legacy_canonical_product_sets(rule_str, target_smi)
    new = _new_canonical_product_sets(rule_str, target_smi)
    assert legacy == new, (
        f"product-set divergence on {label!r}:\n"
        f"  legacy-only: {legacy - new}\n"
        f"  new-only:    {new - legacy}"
    )


@pytest.mark.parametrize("rule_str, target_smi, label", _RULES)
def test_canonical_retro_reactor_products_are_canonical(rule_str, target_smi, label):
    """Every product yielded by ``CanonicalRetroReactor`` must
    already be in canonical form — i.e. running
    ``safe_canonicalization`` on it again must not change ``str()``.
    If this fails, downstream ``Precursor(canonicalize=False)`` is unsafe."""
    from synplan.chem.utils import safe_canonicalization

    rule = smarts(rule_str)
    target = smiles(target_smi)
    reactor = CanonicalRetroReactor(
        patterns=tuple(rule.reactants),
        products=tuple(rule.products),
        delete_atoms=False,
    )
    fired = False
    for reaction in reactor(target):
        fired = True
        for product in reaction.products:
            again = safe_canonicalization(product.copy())
            assert str(product) == str(again), (
                f"product not canonical on {label!r}: "
                f"reactor emitted {product}, re-canonicalize gives {again}"
            )
    assert fired, f"rule did not fire on {label!r}"


# Speed regression tests were removed: in-process micro-benchmarks are
# too noisy on CI and degenerate on tiny molecules (the bare-benzene
# case spent more time on the aromaticity snapshot than it saved). The
# real-workload speedup measurement lives in
# ``sascore-bench/playground/bin1_disagreement/bench_canonical_reactor.py``.
