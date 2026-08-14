"""Regression coverage for typed-stock route parity."""

from __future__ import annotations

from test_tree_stats import (
    FakePolicy,
    FakeReactor,
    FixedEvaluationStrategy,
    make_mol,
)

from synplan.chem.building_blocks import BuildingBlockStock, molecule_to_inchi_key
from synplan.mcts.tree import Tree
from synplan.utils.config import TreeConfig
from synplan.utils.visualisation import extract_routes


def _stock_for(identity_format: str, molecule) -> BuildingBlockStock:
    if identity_format == "inchikey":
        key = molecule_to_inchi_key(molecule)
    else:
        key = str(molecule)
    return BuildingBlockStock(frozenset({key}), identity_format)


def _build_tree(
    target,
    stock: BuildingBlockStock,
    *,
    min_mol_size: int = 0,
    reactor=None,
) -> Tree:
    ranked_rules = [] if reactor is None else [(1.0, reactor, 0)]
    return Tree(
        target=target,
        config=TreeConfig(
            algorithm="breadth_first",
            max_iterations=4,
            max_tree_size=20,
            max_time=5,
            max_depth=2,
            min_mol_size=min_mol_size,
            silent=True,
            enable_pruning=False,
        ),
        reaction_rules=[] if reactor is None else [reactor],
        building_blocks=stock,
        expansion_function=FakePolicy(ranked_rules),
        evaluation_function=FixedEvaluationStrategy(),
    )


def test_smiles_and_inchikey_stocks_produce_the_same_nonzero_route() -> None:
    product = make_mol(8)
    trees = []
    for identity_format in ("smiles", "inchikey"):
        reactor = FakeReactor(lambda: [make_mol(8)])
        tree = _build_tree(
            make_mol(7),
            _stock_for(identity_format, product),
            reactor=reactor,
        )
        tree.stop_at_first = True
        tree.run()
        trees.append(tree)

    smiles_tree, inchikey_tree = trees
    assert smiles_tree.curr_iteration == inchikey_tree.curr_iteration == 1
    assert len(smiles_tree.winning_nodes) == len(inchikey_tree.winning_nodes) == 1
    assert len(smiles_tree.synthesis_route(smiles_tree.winning_nodes[0])) == 1
    assert len(inchikey_tree.synthesis_route(inchikey_tree.winning_nodes[0])) == 1
    assert extract_routes(smiles_tree) == extract_routes(inchikey_tree)
