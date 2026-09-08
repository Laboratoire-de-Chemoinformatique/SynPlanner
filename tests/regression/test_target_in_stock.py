"""A target already in the catalogue must be reported, not planned."""

import csv
import gzip
import json

import pytest
from chython import smiles

from synplan.chem.building_blocks import molecule_to_inchikey
from synplan.chem.utils import mol_from_smiles
from synplan.mcts.search import run_search


def test_purchasable_target_is_skipped_and_flagged(tmp_path, monkeypatch):
    """run_search reports a catalogue hit instead of spending a search on it."""
    calls = []

    class _NoTree:
        def __init__(self, **kwargs):
            calls.append(kwargs["target"])
            raise AssertionError("a purchasable target must not reach the search")

    monkeypatch.setattr("synplan.mcts.search.Tree", _NoTree)
    monkeypatch.setattr(
        "synplan.mcts.search.load_building_blocks", lambda *a, **k: {"CCN"}
    )
    monkeypatch.setattr("synplan.mcts.search.load_reaction_rules", lambda *a, **k: [])
    monkeypatch.setattr(
        "synplan.mcts.search.load_policy_function", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "synplan.mcts.search.load_evaluation_function", lambda *a, **k: None
    )

    targets = tmp_path / "targets.smi"
    targets.write_text("CCN\n")
    run_search(
        targets_path=str(targets),
        search_config={"max_iterations": 1, "silent": True},
        policy_config=None,
        evaluation_config=None,
        reaction_rules_path="unused",
        building_blocks_path="unused",
        results_root=str(tmp_path / "out"),
    )

    assert not calls, "the search was built for a target that is already purchasable"
    stats = list(csv.DictReader((tmp_path / "out" / "tree_search_stats.csv").open()))
    assert stats[0]["target_in_stock"] == "True"
    assert stats[0]["num_routes"] == "0"


@pytest.mark.parametrize("compressed", [False, True])
def test_json_catalogue_skip_still_writes_the_cost_sidecar(
    tmp_path, monkeypatch, compressed
):
    molecule = smiles("CCN")
    key = molecule_to_inchikey(molecule)
    monkeypatch.setattr("synplan.mcts.search.load_reaction_rules", lambda *a, **k: [])
    monkeypatch.setattr(
        "synplan.mcts.search.load_policy_function", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "synplan.mcts.search.load_evaluation_function", lambda *a, **k: None
    )

    targets = tmp_path / "targets.smi"
    targets.write_text("CCN\n")
    catalogue = tmp_path / ("blocks.json.gz" if compressed else "blocks.json")
    data = json.dumps(
        {
            key: {
                "smiles": str(molecule),
                "vendors": {"vendor": 2.0},
                "has_stereo": False,
            }
        }
    ).encode()
    catalogue.write_bytes(gzip.compress(data) if compressed else data)
    output = tmp_path / "out"
    run_search(
        targets_path=str(targets),
        search_config={"max_iterations": 1, "silent": True},
        policy_config=None,
        evaluation_config=None,
        reaction_rules_path="unused",
        building_blocks_path=str(catalogue),
        results_root=str(output),
    )

    assert json.loads((output / "route_costs.json").read_text()) == {"CCN": {}}
    stats = list(csv.DictReader((output / "tree_search_stats.csv").open()))
    assert stats[0]["target_in_stock"] == "True"


@pytest.mark.parametrize("stock_case", ["missing", "grouped", "identity_error"])
def test_searched_target_exports_expansion_statistics(
    tmp_path, monkeypatch, stock_case
):
    from synplan.mcts.evaluation import RandomEvaluationStrategy

    class EmptyPolicy:
        def predict_reaction_rules(self, *args):
            return iter(())

    target, stock = "CCCCCC", set()
    if stock_case == "grouped":
        target = "C[C@H](O)[C@H](N)C(=O)O |&1:1,3|"
        stock = {str(mol_from_smiles(target))}
    elif stock_case == "identity_error":
        stock = {}

        def fail_identity(*args, **kwargs):
            raise ValueError("Cannot generate InChIKey")

        monkeypatch.setattr(
            "synplan.chem.building_blocks.identity.inchi_key", fail_identity
        )

    monkeypatch.setattr(
        "synplan.mcts.search.load_building_blocks", lambda *a, **k: stock
    )
    monkeypatch.setattr("synplan.mcts.search.load_reaction_rules", lambda *a, **k: [])
    monkeypatch.setattr(
        "synplan.mcts.search.load_policy_function", lambda *a, **k: EmptyPolicy()
    )
    monkeypatch.setattr(
        "synplan.mcts.search.load_evaluation_function",
        lambda *a, **k: RandomEvaluationStrategy(),
    )
    targets = tmp_path / "targets.smi"
    targets.write_text(target + "\n")
    run_search(
        targets_path=str(targets),
        search_config={"max_iterations": 1, "silent": True, "min_mol_size": 0},
        policy_config=None,
        evaluation_config=None,
        reaction_rules_path="unused",
        building_blocks_path="unused",
        results_root=str(tmp_path / "out"),
    )
    stats = list(csv.DictReader((tmp_path / "out/tree_search_stats.csv").open()))
    assert stats[0]["target_in_stock"] == "False"
    assert stats[0]["unique_expanded_molecules"] == "1"
    assert stats[0]["unique_expanded_states"] == "1"
    assert stats[0]["root_disconnections"] == "0"
