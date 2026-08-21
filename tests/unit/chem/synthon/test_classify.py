"""The classifier against the paper authors' own published fixture."""

from pathlib import Path

import pytest
from chython import smiles

from synplan.chem.utils import safe_canonicalization
from synplan.synthon.classify import (
    BBClassifier,
    SynthonDataError,
    _compile,
)
from synplan.synthon.config import SynthonConfig, load_data

FIXTURES = Path(__file__).resolve().parents[3] / "data" / "synthon"


@pytest.fixture(scope="module")
def classifier():
    return BBClassifier()


def published():
    for line in (FIXTURES / "outSynth_BBmode.smi").read_text().splitlines():
        fields = line.split("\t")
        yield fields[0], fields[2].split("+")


@pytest.mark.parametrize("smi,expected", list(published()))
def test_published_classes(classifier, smi, expected):
    assert classifier.classify_smiles(smi) == expected


def test_the_class_file_is_ordered_and_complete():
    records = load_data(SynthonConfig().classes_path)
    assert len(records) == 147
    assert [r["name"] for r in records[:6]] == [
        "Acetylenes_AlkyneCH",
        "Acid_Aromatic_Acid",
        "Acid_Aliphatic_Acid",
        "Acid_HetAcetic_Acid",
        "Acid_ArAcetic_Acid",
        "Acid_Heteroaromatic_Acid",
    ]
    patterns = sum(len(r[k]) for r in records for k in ("at_least_one", "also", "not"))
    assert patterns == 2401


def test_the_loader_raises_rather_than_skipping():
    # 93% of the corpus is exclusions, and a dropped exclusion OVER-classifies, which a
    # presence-asserting smoke test cannot see
    with pytest.raises(SynthonDataError):
        _compile(["[C;this-is-not-smarts]"], "deliberate")


def test_kekule_input_silently_matches_nothing(classifier):
    # chython does not aromatise on parse: the trap that would zero most of the 147 classes
    kekule = smiles("C1=CC=C(C=C1)NC2=CC=CC=C2")
    assert classifier.classify(kekule) == []
    assert classifier.classify(safe_canonicalization(kekule)) == [
        "SecondaryAmines_diArylAmines"
    ]


def test_substructure_not_proper_substructure(classifier):
    # aniline is exactly the size of its own required pattern, so `<` would return False
    assert "PrimaryAmines_PriAmines_Anilines" in classifier.classify_smiles("Nc1ccccc1")
