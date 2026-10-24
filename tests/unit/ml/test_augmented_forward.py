"""Shared chemistry identity, deterministic augmentation and rank aggregation."""

from types import SimpleNamespace

from chython import inchi_key, smiles
from rdkit import Chem

from synplan.ml.inference.forward import (
    aggregate,
    augment,
    candidate,
    canonical_reactants,
    predict_augmented,
)


def row(*predictions):
    return {
        "predictions": list(predictions),
        "log_likelihoods": [-float(i) for i in range(len(predictions))],
    }


def test_stereo_dedup_invalid_filter_isotopes_and_fragments():
    result = aggregate(
        "C",
        [
            row(
                "bad",
                "C[C@H](O)F",
                "C[C@@H](O)F",
                "CC(O)F",
                "[13CH3]C(O)F",
                "CC(O)F.Cl",
            )
        ],
    )
    assert len(result["predictions"]) == 3
    assert len(set(result["product_inchikeys"])) == 3
    assert "@" in result["predictions"][0]
    assert result["log_likelihoods"] == [-1.0, -4.0, -5.0]
    assert result["ranking_scores"] == [1.0, 0.5 - 1e8, 1 / 3 - 2e8]


def test_best_cleaned_rank_support_and_representative():
    result = aggregate(
        "C", [row("N", "C[C@H](O)F", "O"), row("invalid", "C[C@@H](O)F", "N", "O")]
    )
    # Both best-rank one and support 1.5; N appeared first.
    assert result["product_inchikeys"][0] == inchi_key(smiles("N"), ignore_stereo=True)
    assert result["ranking_scores"][:2] == [1.5, 1.5]
    assert (
        result["predictions"][1]
        == aggregate("C", [row("C[C@@H](O)F")])["predictions"][0]
    )
    assert result["log_likelihoods"][1] == -1.0
    assert result["ranking_scores"][2] == 2 / 3 - 2e8


def test_best_rank_dominates_support_no_padding():
    result = aggregate("C", [row("C", "N"), row("O", "N"), row("F", "N")])
    assert result["predictions"] == ["C", "O", "F", "N"]
    assert aggregate("C", [row("", "invalid")])["predictions"] == []
    assert len(aggregate("C", [row("C", "N", "O")], num_results=2)["predictions"]) == 2


def test_augmentation_determinism_preserves_chemistry_and_stereo():
    value = canonical_reactants("O.[13CH3:8][C@@H:9](F)C(=O)N")
    actual = augment(value)
    assert actual == augment(value) and 1 < len(actual) <= 10
    assert len(actual) == len(set(actual))
    assert actual[0] == ".".join(sorted(value.split(".")))
    assert all(
        Chem.MolToSmiles(Chem.MolFromSmiles(candidate)) == value for candidate in actual
    )
    assert all(canonical_reactants(candidate) == value for candidate in actual)
    assert all("@" in candidate and "13C" in candidate for candidate in actual)
    assert augment("C.C") == ["C.C"]


def test_augmentation_passes_directly_to_tokenizer_and_is_batch_independent():
    received = []

    def raw(values, width):
        received.extend(values)
        return [row("C", "N") for _ in values]

    model = SimpleNamespace(augmentation_size=10, seed=0, _predict_batch=raw)
    inputs = ["CCO.N", "CC(=O)O", "C"]
    expected = predict_augmented(model, inputs, batch_size=1)
    assert predict_augmented(model, inputs, batch_size=2) == expected
    assert predict_augmented(model, list(reversed(inputs)), batch_size=3) == list(
        reversed(expected)
    )
    assert any(value != canonical_reactants(value) for value in received)


def test_chiral_root_in_later_component_keeps_stereo_and_isotope():
    for value in ("O.[C@H](C)(F)C(N)=O", "O.[C@H]([13CH3])(F)C(N)=O"):
        expected = Chem.MolToInchiKey(Chem.MolFromSmiles(value))
        normalized = canonical_reactants(value)
        product, key = candidate(value)
        assert Chem.MolToInchiKey(Chem.MolFromSmiles(normalized)) == expected
        assert Chem.MolToInchiKey(Chem.MolFromSmiles(product)) == expected
        assert key == candidate(normalized)[1]
