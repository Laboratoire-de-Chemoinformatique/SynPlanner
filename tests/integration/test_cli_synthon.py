"""The five synthon subcommands end to end on the published 9-building-block fixture."""

from pathlib import Path

import pytest
from click.testing import CliRunner

import synplan.interfaces.cli as cli

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "synthon"


@pytest.fixture
def blocks(tmp_path):
    path = tmp_path / "bbs.smi"
    path.write_text((FIXTURES / "BBs.cxsmiles").read_text())
    return path


def run(*args):
    result = CliRunner().invoke(cli.synplan, list(args))
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result


def test_bb_classifying(blocks, tmp_path):
    out = tmp_path / "classes.tsv"
    run("bb_classifying", "--input", str(blocks), "--output", str(out))
    rows = [line.split("\t") for line in out.read_text().splitlines()]
    assert len(rows) == 9
    assert rows[0][2] == "Ketones_Ketones"


def test_bb_synthonizing_reproduces_the_published_stock(blocks, tmp_path):
    out = tmp_path / "synthons.smi"
    run("bb_synthonizing", "--input", str(blocks), "--output", str(out))
    rows = [line.split("\t") for line in out.read_text().splitlines()]
    assert len({r[0] for r in rows}) == 18
    assert all(len(r) == 4 for r in rows)


def test_fragment_then_enumerate(blocks, tmp_path):
    stock = tmp_path / "synthons.smi"
    run("bb_synthonizing", "--input", str(blocks), "--output", str(stock))
    targets = tmp_path / "targets.smi"
    targets.write_text("CCCCCC(C)OC(=O)CC\ttarget\n")
    pathways = tmp_path / "pathways.tsv"
    run(
        "synthon_fragment",
        "--input",
        str(targets),
        "--output",
        str(pathways),
        "--stock",
        str(stock),
    )
    rows = [line.split("\t") for line in pathways.read_text().splitlines()]
    assert rows and all(len(r) == 5 for r in rows)
    assert any(float(r[4]) > 0 for r in rows)  # the stock is actually consulted

    library = tmp_path / "library.smi"
    run(
        "synthon_enumerate",
        "--input",
        str(pathways),
        "--output",
        str(library),
        "--stock",
        str(stock),
    )
    assert library.read_text().splitlines()


def test_bb_scaffolds(blocks, tmp_path):
    out = tmp_path / "scaffolds.tsv"
    run("bb_scaffolds", "--input", str(blocks), "--output", str(out))
    rows = dict(line.split("\t") for line in out.read_text().splitlines())
    assert rows["CCO"] == "linearMolecule"
    assert rows["C1=CC=C(C=C1)N"] == "c1ccccc1"
