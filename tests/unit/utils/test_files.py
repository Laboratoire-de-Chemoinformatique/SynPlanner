from synplan.utils.files import (
    count_chemical_records,
    iter_chemical_records,
    load_rule_index_mapping_tsv,
)


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


def test_iter_chemical_records_supports_configured_table_column(tmp_path):
    source = tmp_path / "catalogue.csv"
    source.write_text("vendor,molecule,name\nA,OCC,ethanol\n", encoding="utf-8")

    record = next(
        iter_chemical_records(
            source,
            chemistry_columns={"smiles": "smiles", "cxsmiles": "smiles"},
            chemistry_column="molecule",
        )
    )

    assert record.chemistry == "OCC"
    assert record.chemistry_format == "smiles"
    assert record.input_format == "csv"
    assert record.metadata_value == {"vendor": "A", "name": "ethanol"}
    assert record.fallback_record == ('OCC\t{"vendor":"A","name":"ethanol"}')
    assert record.format_error is None


def test_count_chemical_records_matches_table_framing(tmp_path):
    source = tmp_path / "catalogue.tsv"
    source.write_text(
        'SMILES\tname\nCCO\t"line one\nline two"\n\nCCN\tother\n',
        encoding="utf-8",
    )

    records = list(iter_chemical_records(source))

    assert count_chemical_records(source) == len(records) == 2


def test_count_chemical_records_includes_unterminated_sdf_frame(tmp_path):
    source = tmp_path / "catalogue.sdf"
    source.write_text(
        "valid frame\n$$$$\nunterminated frame\n",
        encoding="utf-8",
    )

    assert count_chemical_records(source) == 2
