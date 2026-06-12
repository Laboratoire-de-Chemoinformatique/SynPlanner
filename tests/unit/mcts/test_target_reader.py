"""run_search target reading across SMILES / CSV / TSV inputs."""

from synplan.mcts.search import _iter_target_smiles


def test_reads_plain_smiles_file(tmp_path):
    p = tmp_path / "targets.smi"
    p.write_text("CCO\n\nCCN\n", encoding="utf-8")
    assert list(_iter_target_smiles(str(p))) == ["CCO", "CCN"]


def test_reads_tsv_smiles_column(tmp_path):
    p = tmp_path / "targets.tsv"
    p.write_text("smiles\tid\tsascore\nCCO\tA\t1.0\nCCN\tB\t2.0\n", encoding="utf-8")
    assert list(_iter_target_smiles(str(p))) == ["CCO", "CCN"]


def test_reads_csv_smiles_column(tmp_path):
    p = tmp_path / "targets.csv"
    p.write_text("SMILES,id\nCCO,A\nCCN,B\n", encoding="utf-8")
    assert list(_iter_target_smiles(str(p))) == ["CCO", "CCN"]
