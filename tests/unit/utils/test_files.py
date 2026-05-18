from synplan.utils.files import load_rule_index_mapping_tsv


def test_load_rule_index_mapping_tsv_preserves_multiple_rules_per_reaction(tmp_path):
    rules_path = tmp_path / "rules.tsv"
    rules_path.write_text(
        "rule_smarts\tpopularity\treaction_indices\n"
        "rule-a\t3\t10,11\n"
        "rule-b\t2\t10\n"
        "rule-c\t1\t11,12\n",
        encoding="utf-8",
    )

    assert load_rule_index_mapping_tsv(rules_path) == {
        10: [0, 1],
        11: [0, 2],
        12: [2],
    }
