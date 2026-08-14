from types import SimpleNamespace

from chython import smiles

from synplan.chem.building_blocks import BuildingBlockStock, molecule_to_inchi_key
from synplan.ml.training import reinforcement
from synplan.utils.config import TreeConfig


def test_reinforcement_removes_only_the_full_target_inchikey(monkeypatch):
    target = smiles("N[C@@H](C)C(=O)O")
    other_enantiomer = smiles("N[C@H](C)C(=O)O")
    target_key = molecule_to_inchi_key(target)
    other_key = molecule_to_inchi_key(other_enantiomer)
    assert target_key[:14] == other_key[:14]
    assert target_key != other_key
    source_stock = BuildingBlockStock(
        frozenset({target_key, other_key}),
        "inchikey",
    )
    captured = {}

    class FakeTree:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._tqdm = True

        def run(self):
            return self

    monkeypatch.setattr(
        reinforcement, "load_policy_function", lambda **kwargs: object()
    )
    monkeypatch.setattr(reinforcement, "load_reaction_rules", lambda path: ())
    monkeypatch.setattr(
        reinforcement,
        "load_evaluation_function",
        lambda config: object(),
    )
    monkeypatch.setattr(reinforcement, "Tree", FakeTree)

    tree = reinforcement.run_tree_search(
        target=target,
        tree_config=TreeConfig(),
        policy_config=object(),
        value_config=SimpleNamespace(weights_path="unused.ckpt"),
        reaction_rules_path="unused.rules",
        building_block_stock=source_stock,
    )

    assert tree._tqdm is False
    passed_stock = captured["building_blocks"]
    assert passed_stock.identity_format == "inchikey"
    assert target_key not in passed_stock
    assert other_key in passed_stock
