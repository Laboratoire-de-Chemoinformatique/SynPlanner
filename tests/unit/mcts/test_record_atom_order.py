"""Search-record round trips preserve atom order and stereo evidence."""

from chython import smiles
from test_tree_stats import FakeReactor, build_tree, make_mol

from synplan.mcts.record import read_search_record, write_search_record


def test_record_preserves_symmetric_reactant_order_and_stereo_context(tmp_path):
    from synplan.chem.utils import in_atom_order

    solvent = in_atom_order(smiles("[CH3:27][S:25]([CH3:26])=[O:10]"))
    tree = build_tree(
        rules=[(0.5, FakeReactor(lambda: [solvent, make_mol(3)]), 0)], max_iterations=1
    ).run()
    original = tree.routes()[0]
    assert original.stereo_status == "fulfilled"
    record = read_search_record(write_search_record(tree, tmp_path / "tree.json"))
    restored = record.routes()[0]
    assert restored.stereo_status == original.stereo_status
    assert restored.to_json() == original.to_json()
    # Real edits must still invalidate the original stereo review.
    restored.steps[0].reaction.meta["conditions"] = "changed"
    assert restored.stereo_status == "needs_reassessment"
