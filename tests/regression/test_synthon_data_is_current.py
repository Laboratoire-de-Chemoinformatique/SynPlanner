"""The committed JSON must be exactly what the converter produces from the pinned inputs.

A generated file that drifted is a build error, not a runtime warning — and the translated SMARTS
are the one artefact a chemist has to be able to read in the diff.
"""

import json
from pathlib import Path

import pytest

from synplan.chem.synthon.config import SynthonConfig, load_data

# the reference clone lives beside the repo, not inside it, so walk up until it turns up
CONFIG_DIR = next(
    (
        p / "research" / "synthon" / "Synt-On" / "config"
        for p in Path(__file__).resolve().parents
        if (p / "research" / "synthon" / "Synt-On" / "config").is_dir()
    ),
    Path("/nonexistent"),
)

pytestmark = pytest.mark.skipif(
    not CONFIG_DIR.is_dir(),
    reason="the Synt-On reference clone is not checked out here",
)


@pytest.fixture(scope="module")
def built():
    from synplan.chem.synthon.rules._convert import build

    return build(CONFIG_DIR)


@pytest.mark.parametrize(
    "name,attribute",
    [
        ("bb_classes", "classes_path"),
        ("bb_marks", "marks_path"),
        ("rules", "rules_path"),
    ],
)
def test_committed_data_matches_the_converter(built, name, attribute):
    committed = json.loads(Path(getattr(SynthonConfig(), attribute)).read_text())
    assert committed == json.loads(json.dumps(built[name]))


def test_the_converter_self_check_passes(built):
    from synplan.chem.synthon.rules._convert import check

    problems = [p for p in check(built, CONFIG_DIR) if not p.startswith("note:")]
    assert problems == []


def test_the_whole_classifier_corpus_translates():
    from chython import smarts

    from synplan.chem.synthon.rules._dialect import to_chython

    library = json.loads((CONFIG_DIR / "SMARTSLibNew.json").read_text())
    patterns = [
        p
        for big in library.values()
        for record in big.values()
        for key in ("ShouldContainAtLeastOne", "ShouldAlsoContain", "shouldNotContain")
        for p in record.get(key, ())
    ]
    assert len(patterns) == 2401
    for pattern in patterns:
        smarts(to_chython(pattern))


CATALOGUE = next(
    (
        p / "SynPlanner" / "building_blocks" / "building_blocks_em_sa_ln.smi"
        for p in Path(__file__).resolve().parents
        if (
            p / "SynPlanner" / "building_blocks" / "building_blocks_em_sa_ln.smi"
        ).is_file()
    ),
    Path("/nonexistent"),
)


@pytest.mark.skipif(
    not CATALOGUE.is_file(), reason="the building-block catalogue is not here"
)
def test_the_fork_parses_real_building_blocks_identically():
    """The new bracket field is optional and `_` cannot occur in a bracket today, so nothing that
    parses now may parse differently — measured on real catalogue lines, not on hand-picked ones.

    A line the catalogue gets wrong must fail the SAME way on both paths, which is why the
    exception is part of the compared answer.
    """
    import random

    from chython import smiles, synthon_smiles

    def answer(parse, text):
        try:
            molecule = parse(text)
            molecule.canonicalize()
            return str(molecule)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    lines = [
        line.split()[0] for line in CATALOGUE.read_text().splitlines() if line.strip()
    ]
    random.Random(20260808).shuffle(lines)
    differing = [
        text
        for text in lines[:3000]
        if answer(smiles, text) != answer(synthon_smiles, text)
    ]
    assert differing == []


def test_every_disconnection_declares_its_provenance():
    """A chemist reading rules.json must be able to tell curated chemistry from proposed.

    The Enamine rules carry decades of industrial experience; the ring rules were authored here
    and most have not been through a chemist yet.
    """
    rules = load_data(SynthonConfig().rules_path)["disconnections"]
    unknown = {
        r["id"]: r.get("provenance")
        for r in rules
        if r.get("provenance") not in ("human", "llm")
    }
    assert unknown == {}
    # the split is structural: everything converted from Setup.xml is curated, the ring set is not
    assert {r["provenance"] for r in rules if r["ring"]} == {"llm"}
    assert {r["provenance"] for r in rules if not r["ring"]} == {"human"}
