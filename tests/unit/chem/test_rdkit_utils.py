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
