"""The fragmentation and CLI defects of the Synt-On port, one test per defect."""

import pytest
from chython import smiles

from synplan.chem.synthon import cli
from synplan.chem.synthon.classify import BBClassifier
from synplan.chem.synthon.config import SynthonConfig, load_data
from synplan.chem.synthon.fragment import (
    STUDY_FRAGMENTS_TO_IGNORE,
    Fragmenter,
    _select,
    fragment_smiles,
)
from synplan.chem.synthon.stock import SynthonRecord, write_synthon_stock
from synplan.chem.utils import safe_canonicalization

IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
MACROLACTAM = "O=C1CCCCCCCCCCCCN1"
PARACETAMOL_SYNTHONS = ("C[CH_elec]=O", "c1cc(ccc1[NH2_nuc])O")
RULES = [
    r for r in load_data(SynthonConfig().rules_path)["disconnections"] if not r["macro"]
]


@pytest.fixture
def stock(tmp_path):
    path = tmp_path / "stock.smi"
    write_synthon_stock(
        str(path),
        [SynthonRecord(s, ("bb",), ("X",), 0) for s in PARACETAMOL_SYNTHONS],
    )
    return path


# --- D1: one_by_one is a search, not a synonym for include_only ---------------------------


def test_one_by_one_is_a_strict_non_empty_subset_of_use_all():
    every = fragment_smiles(IMATINIB, SynthonConfig(rule_mode="use_all"))
    stepwise = fragment_smiles(IMATINIB, SynthonConfig(rule_mode="one_by_one"))
    again = fragment_smiles(IMATINIB, SynthonConfig(rule_mode="one_by_one"))
    assert stepwise.pathways
    assert set(stepwise.pathways) < set(every.pathways)
    assert set(stepwise.pathways) == set(again.pathways)
    # level 1 left after the first rule that matched, so every pathway starts with that rule
    assert len({p.rules[0].rsplit("_", 1)[0] for p in stepwise.pathways.values()}) == 1


# --- D2: the macro rules obey rule_mode too -----------------------------------------------


def test_the_macro_rules_honour_the_selection():
    excluded = SynthonConfig(rule_mode="exclude_some", rules_selection="R1-R13")
    assert not fragment_smiles(MACROLACTAM, excluded).pathways
    only = SynthonConfig(rule_mode="include_only", rules_selection="R6")
    dag = fragment_smiles(MACROLACTAM, only)
    assert {r.split(".")[0] for p in dag.roots() for r in p.rules} == {"MR6"}


# --- D3: a salt target is cut as its parent ------------------------------------------------


@pytest.mark.parametrize(
    "salt",
    [f"{PARACETAMOL}.Cl", f"{PARACETAMOL}.O", f"CS(=O)(=O)O.{PARACETAMOL}"],
)
def test_a_salt_target_is_cut_as_its_parent(salt):
    assert set(fragment_smiles(salt).pathways) == set(
        fragment_smiles(PARACETAMOL).pathways
    )


def test_two_comparable_components_are_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        fragment_smiles("NCc1ccccc1.OC(=O)C(F)(F)F")


# --- D4: classification splits components, like synthonisation ----------------------------


def test_classification_splits_components():
    classifier = BBClassifier()
    amine = classifier.classify_smiles("NCc1ccccc1")
    assert amine
    # the counter-ion's exclusion patterns used to destroy every class of the parent
    assert classifier.classify_smiles("NCc1ccccc1.OS(=O)(=O)O") == amine
    trifluoroacetate = classifier.classify_smiles("NCc1ccccc1.OC(=O)C(F)(F)F")
    assert set(amine) < set(trifluoroacetate)
    # the amine of one component and the acid of the other are not an amino acid
    assert not any("Aminoacid" in name for name in trifluoroacetate)


# --- D5: the memo is per target -----------------------------------------------------------


def test_the_memo_does_not_carry_over_between_targets():
    fragmenter = Fragmenter()
    fragmenter.fragment(safe_canonicalization(smiles(PARACETAMOL)))
    alone = len(fragmenter._memo)
    fragmenter.fragment(safe_canonicalization(smiles(IMATINIB)))
    fragmenter.fragment(safe_canonicalization(smiles(PARACETAMOL)))
    assert alone
    assert len(fragmenter._memo) == alone


# --- D6: the CLI entry points --------------------------------------------------------------


def test_no_command_truncates_its_own_input(tmp_path, stock):
    path = tmp_path / "bbs.smi"
    path.write_text("CCO\tid1\nNCc1ccccc1\tid2\n")
    same = str(path)
    before = path.read_text()
    for call in (
        lambda: cli.classify_file(same, same),
        lambda: cli.synthonise_file(same, same),
        lambda: cli.fragment_file(same, same),
        lambda: cli.enumerate_file(same, same, str(stock)),
        lambda: cli.scaffolds_file(same, same),
    ):
        with pytest.raises(ValueError):
            call()
    assert path.read_text() == before


def test_one_bad_row_does_not_abort_the_batch(tmp_path, stock):
    targets = tmp_path / "targets.smi"
    targets.write_text(f"[Xx]\tbroken\n{PARACETAMOL}\tgood\n")
    pathways = tmp_path / "pathways.tsv"
    assert cli.fragment_file(str(targets), str(pathways))
    assert all(
        line.split("\t")[0] == PARACETAMOL for line in pathways.read_text().splitlines()
    )

    rows = tmp_path / "rows.tsv"
    rows.write_text(
        f"broken\tR1.1_0\tnot_a_synthon\t1\t0.0\n"
        f"{PARACETAMOL}\tR2.2_0\t{'.'.join(PARACETAMOL_SYNTHONS)}\t1\t1.0\n"
    )
    library = tmp_path / "library.smi"
    assert cli.enumerate_file(str(rows), str(library), str(stock)) == 1


def test_the_synthon_stock_is_streamed(tmp_path, monkeypatch):
    path = tmp_path / "bbs.smi"
    path.write_text("NCc1ccccc1\tid1\n")
    seen = {}

    def capture(target, records):
        seen["lazy"] = iter(records) is records
        write_synthon_stock(target, records)

    monkeypatch.setattr(cli, "write_synthon_stock", capture)
    written, _ = cli.synthonise_file(str(path), str(tmp_path / "out.smi"))
    assert seen["lazy"]
    assert written


def test_every_input_row_keeps_its_own_name(tmp_path):
    path = tmp_path / "bbs.smi"
    path.write_text("CCO\tid1\nCCO\tid2\n")
    out = tmp_path / "classes.tsv"
    assert cli.classify_file(str(path), str(out)) == 2
    assert {line.split("\t")[1] for line in out.read_text().splitlines()} == {
        "id1",
        "id2",
    }


def test_enumerated_products_carry_their_target(tmp_path, stock):
    rows = tmp_path / "rows.tsv"
    rows.write_text(
        f"{PARACETAMOL}\tR2.2_0\t{'.'.join(PARACETAMOL_SYNTHONS)}\t1\t1.0\n"
    )
    library = tmp_path / "library.smi"
    assert cli.enumerate_file(str(rows), str(library), str(stock)) == 1
    fields = library.read_text().splitlines()[0].split("\t")
    assert len(fields) == 4
    assert fields[1] == PARACETAMOL


# --- D7: the range grammar ------------------------------------------------------------------


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("R1.2-R1.4", ["R1.2", "R1.3", "R1.4"]),
        ("R8-R9", ["R8.1", "R9.1"]),
        ("R12.3", ["R12.3a", "R12.3b"]),
        ("R12.3a", ["R12.3a"]),
    ],
)
def test_the_range_grammar_uses_both_halves(selection, expected):
    assert [r["id"] for r in _select(RULES, "include_only", selection)] == expected


@pytest.mark.parametrize("selection", ["R2.3-R2.1", "R5-R1", "R99", "R1.9", ""])
def test_a_selection_that_resolves_to_nothing_raises(selection):
    with pytest.raises(ValueError):
        _select(RULES, "include_only", selection)


# --- D8: the ignored-fragment list is canonicalised ------------------------------------------


def test_a_fragment_to_ignore_is_canonicalised_before_it_is_compared():
    # the study list is spelled in the reference's dialect; chython writes this one C(=O)C
    assert str(safe_canonicalization(smiles("CC=O"))) != "CC=O"
    plain = fragment_smiles(PARACETAMOL)
    ignored = fragment_smiles(PARACETAMOL, SynthonConfig(fragments_to_ignore=["CC=O"]))
    assert set(ignored.pathways) < set(plain.pathways)
    study = Fragmenter(
        SynthonConfig(fragments_to_ignore=list(STUDY_FRAGMENTS_TO_IGNORE))
    )
    assert study.ignore >= {"C(=O)C", "CCC"}
