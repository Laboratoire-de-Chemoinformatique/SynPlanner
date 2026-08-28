"""``apply_reaction_rule`` says when a precursor is not canonical.

The reactor canonicalizes the whole patched molecule and then yields its
fragments, and a fragment does not always canonicalize the way it did inside the
whole. Such a precursor is not repaired — repairing hides where it came from, and
the point is to be able to go and look.

This replaced an aromaticity restorer that put a ring's old flags back whenever a
``kekule`` -> ``thiele`` round trip dropped them. That could not tell a rule
rewriting a ring from an input claiming an aromaticity chython does not perceive,
and asserting the old flags left molecules ``canonicalize`` disagreed with.
"""

from __future__ import annotations

import logging

from chython import smarts, smiles

from synplan.chem.reaction import CanonicalRetroReactor, apply_reaction_rule
from synplan.chem.reaction.reactor import _warn_if_not_canonical


def _reactor(rule: str) -> CanonicalRetroReactor:
    parsed = smarts(rule)
    return CanonicalRetroReactor(
        patterns=tuple(parsed.reactants),
        products=tuple(parsed.products),
        delete_atoms=False,
    )


def test_a_precursor_the_canonicalizer_rewrites_is_reported(caplog):
    """A ring hydrogen the canonicalizer puts on the other nitrogen. Both forms are
    aromatic, so no aromaticity check could see it."""
    molecule = smiles("FC(F)(F)c1[nH]nc(-c2ccc(cc2)C)c1")

    with caplog.at_level(logging.DEBUG, logger="synplan.chem.reaction.reactor"):
        _warn_if_not_canonical([molecule])

    assert [record for record in caplog.records if "is not canonical" in record.message]
    assert "FC(F)(F)c1n[nH]c(c1)-c2ccc(cc2)C" in caplog.records[0].message, (
        "the warning should name the form the canonicalizer writes"
    )


def test_a_canonical_precursor_is_not_reported(caplog):
    with caplog.at_level(logging.DEBUG, logger="synplan.chem.reaction.reactor"):
        _warn_if_not_canonical([smiles("c1cc(ccc1Cl)C(CCO)N")])

    assert not caplog.records


def test_a_kekulized_ring_is_not_by_itself_a_complaint(caplog):
    """1-methyl-2-(hydroxymethyl)quinolin-4(1H)-one: RDKit aromatises the pyridinone
    ring and chython does not, so the product comes out kekulized — and canonical,
    which is what matters. This is the case the old restorer was written for."""
    with caplog.at_level(logging.DEBUG, logger="synplan.chem.reaction.reactor"):
        products = list(
            apply_reaction_rule(
                smiles("OCc1cc(=O)c2ccccc2n1C"),
                _reactor("[C:1]-[O;h1:2]>>[C:1]=[O:2]"),
            )
        )

    assert products, "the rule did not fire; the test molecule no longer matches"
    assert not [
        record for record in caplog.records if "is not canonical" in record.message
    ]


def test_the_check_costs_nothing_unless_asked_for(monkeypatch, caplog):
    """Canonicalizing every precursor costs the search about a tenth of its time,
    so nothing is canonicalized until this module's logger is set to DEBUG."""
    called = []
    monkeypatch.setattr(
        "synplan.chem.reaction.reactor.safe_canonicalization",
        lambda molecule: called.append(molecule) or molecule,
    )

    with caplog.at_level(logging.INFO, logger="synplan.chem.reaction.reactor"):
        _warn_if_not_canonical([smiles("FC(F)(F)c1[nH]nc(-c2ccc(cc2)C)c1")])
    assert not called, "the canonicalizer ran with the check switched off"

    with caplog.at_level(logging.DEBUG, logger="synplan.chem.reaction.reactor"):
        _warn_if_not_canonical([smiles("FC(F)(F)c1[nH]nc(-c2ccc(cc2)C)c1")])
    assert called, "the check did not run with DEBUG on"
