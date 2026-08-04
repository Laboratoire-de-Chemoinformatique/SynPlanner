"""The hand-written rule sets are importable and usable.

Nothing in synplan imports this package, so it silently stopped importing under
chython 1.100 and only the docs build noticed. These tests are the importer.
"""

import pytest
from chython import smiles
from chython.reactor import Reactor

from synplan.chem.reaction.rules.manual import decompositions, transformations


@pytest.mark.parametrize(
    "module",
    [decompositions, transformations],
    ids=["decompositions", "transformations"],
)
def test_rules_are_built(module):
    assert module.rules, f"{module.__name__} exposes no rules"


@pytest.mark.parametrize(
    "module",
    [decompositions, transformations],
    ids=["decompositions", "transformations"],
)
def test_every_rule_builds_a_reactor(module):
    """A rule that cannot become a Reactor cannot be applied to anything."""
    for i, rule in enumerate(module.rules):
        (
            Reactor([rule.reactants[0]], list(rule.products), delete_atoms=True),
            f"rule {i}",
        )


def test_amide_decomposition_fires():
    """Spot-check that a rule does chemistry, not just construct."""
    rule = decompositions.rules[0]
    reactor = Reactor([rule.reactants[0]], list(rule.products), delete_atoms=True)
    molecule = smiles("CC(=O)NCC")
    molecule.canonicalize()

    products = list(reactor(molecule))
    assert products, "retro-amidation did not fire on N-ethylacetamide"
