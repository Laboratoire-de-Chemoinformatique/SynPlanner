from synplan.chem.reaction import CanonicalRetroReactor, apply_reaction_rule
from synplan.chem.utils import is_useful_symmetric_reaction_rule, mol_from_smiles
from synplan.utils.loading import load_reaction_rules


def test_useful_symmetric_rule_requires_automorphism_filter_off():
    rule_smarts = (
        "[c:1]:[c:2](:[c:3])-[c:5](:[c:4]):[c:6]>>"
        "[c:4]:[c:5](:[c:6])-[B:8](-[O:9])-[O:10]."
        "[c:1]:[c:2](:[c:3])-[I:7]"
    )
    assert is_useful_symmetric_reaction_rule(rule_smarts)

    x_molecule = mol_from_smiles("C1=CC(C2C=CC=NC=2)=CC(C)=C1")
    reactor = CanonicalRetroReactor.from_smarts(
        rule_smarts, delete_atoms=False, automorphism_filter=False
    )

    assert len(list(apply_reaction_rule(x_molecule, reactor))) == 2

    reactor._automorphism_filter = True
    assert len(list(apply_reaction_rule(x_molecule, reactor))) == 1


def test_symmetric_rule_detection_does_not_leak_to_later_rules(tmp_path):
    useful_rule = (
        "[c:1]:[c:2](:[c:3])-[c:5](:[c:4]):[c:6]>>"
        "[c:4]:[c:5](:[c:6])-[B:8](-[O:9])-[O:10]."
        "[c:1]:[c:2](:[c:3])-[I:7]"
    )
    later_non_bmg_rule = (
        "[C:1]-[C:2](-[C:3])-[N:5]-[C:4]>>[C:1]-[C:2](-[C:3])=[O:6].[C:4]-[N:5]"
    )
    rules_path = tmp_path / "rules.tsv"
    rules_path.write_text(
        "rule_smarts\tpopularity\treaction_indices\n"
        f"{useful_rule}\t1\t0\n"
        f"{later_non_bmg_rule}\t1\t1\n",
        encoding="utf-8",
    )

    load_reaction_rules.cache_clear()
    useful_reactor, later_reactor = load_reaction_rules(
        str(rules_path), detect_symmetric_rules=True
    )
    load_reaction_rules.cache_clear()

    assert useful_reactor._automorphism_filter is False
    assert later_reactor._automorphism_filter is not False
