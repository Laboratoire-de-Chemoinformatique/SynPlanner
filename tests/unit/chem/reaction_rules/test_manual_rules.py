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
        reactor = Reactor([rule.reactants[0]], list(rule.products), delete_atoms=True)
        assert reactor is not None, f"rule {i}"


def test_amide_decomposition_fires():
    """Retro-amidation of N-ethylacetamide gives acetic acid and ethylamine."""
    rule = decompositions.rules[0]
    reactor = Reactor([rule.reactants[0]], list(rule.products), delete_atoms=True)
    molecule = smiles("CC(=O)NCC")
    molecule.canonicalize()

    reactions = list(reactor(molecule))
    assert len(reactions) == 1, "retro-amidation did not fire on N-ethylacetamide"
    products = set()
    for product in reactions[0].products:
        product.canonicalize()
        products.add(str(product))
    assert products == {"O=C(O)C", "CCN"}
