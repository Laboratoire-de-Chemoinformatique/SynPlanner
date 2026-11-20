from types import SimpleNamespace

import pytest

from synplan.chem import rdkit_utils


def _node(*smiles):
    return SimpleNamespace(
        precursors_to_expand=[
            SimpleNamespace(molecule=smiles_text) for smiles_text in smiles
        ]
    )


@pytest.mark.parametrize(
    "score_function",
    ["sascore", "heavyAtomCount", "weight", "weightXsascore", "WxWxSAS"],
)
def test_rdkit_score_parses_each_precursor_once(monkeypatch, score_function):
    original = rdkit_utils.Chem.MolFromSmiles
    calls = []

    def parse_once(smiles):
        calls.append(smiles)
        return original(smiles)

    monkeypatch.setattr(rdkit_utils.Chem, "MolFromSmiles", parse_once)

    score = rdkit_utils.RDKitScore(score_function)(_node("CC", "CO"))

    assert score is not None
    assert calls == ["CC", "CO"]


def test_rdkit_score_keeps_invalid_molecule_fallback():
    score = rdkit_utils.RDKitScore("heavyAtomCount")(_node("not a molecule"))

    assert score == 0.0


@pytest.mark.parametrize("text", ["CCO", "[13CH3]O", "[2H]O[2H]", "[NH4+]", "[Cl-]"])
def test_weight_score_keeps_exact_isotope_and_charge_masses(text):
    from rdkit.Chem import Descriptors

    mol = rdkit_utils.Chem.MolFromSmiles(text)
    expected = Descriptors.ExactMolWt(mol)
    assert rdkit_utils.CalcExactMolWt(mol) == expected
    score = rdkit_utils.RDKitScore("weight")(_node(text))
    assert score == pytest.approx(max(0.0, 1.0 - expected / 1000.0))
