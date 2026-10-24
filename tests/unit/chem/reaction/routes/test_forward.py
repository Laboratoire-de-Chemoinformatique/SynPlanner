"""Product recovery, persistent cache, and metadata-preserving collection updates."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from chython import inchi_key, smiles

from scripts import assess_routes
from synplan.chem.reaction.routes.quality import forward
from synplan.chem.reaction.routes.quality.forward import ForwardReactionAssessor
from synplan.chem.reaction.routes.route import Route, Step
from synplan.ml.inference.forward import aggregate, candidate, canonical_reactants
from synplan.utils import json as json_io

RUNTIME_FILES = ("encoder.onnx", "decoder.onnx", "vocabulary.json", "model.json")


def route(product="CCO", reactants="CC.N", *, product_index=0):
    reaction = smiles(f"{reactants}>>{product}")
    return Route((Step(reaction, reaction.products[product_index]),))


@pytest.fixture
def model(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    for name in RUNTIME_FILES:
        (directory / name).write_text(name)
    session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

    class Model:
        backend = "onnx"
        runtime_files = RUNTIME_FILES
        augmentation_size = 10
        seed = 0
        max_length = 512
        decoder_batch_size = 64

        def __init__(self):
            self.directory = directory
            self.encoder = self.decoder = session
            self.providers = {
                "encoder": session.get_providers(),
                "decoder": session.get_providers(),
            }
            self.calls = []
            self.predictions = ["CCO"] + ["N"] * 9

        def predict(self, reactants, **kwargs):
            self.calls.append((list(reactants), kwargs))
            return [
                aggregate(
                    value,
                    [
                        {
                            "predictions": list(self.predictions),
                            "log_likelihoods": [
                                -float(i) for i in range(len(self.predictions))
                            ],
                        }
                    ],
                )
                for value in reactants
            ]

    return Model()


def assessor(model, tmp_path, **kwargs):
    return ForwardReactionAssessor(model, tmp_path / "cache.json", **kwargs)


@pytest.mark.parametrize(
    ("predictions", "rank"),
    [
        (["CCO"] + ["N"] * 9, 1),
        (["C", "CC", "CCC", "CCCC", "N", "O", "F", "Cl", "Br", "CCO"], 10),
        (["N"] * 10, None),
        (["C1CC", "", "N", "N", "CCO", "CCO"] + ["N"] * 4, 2),
    ],
)
def test_ranks_invalid_and_duplicate_positions(model, tmp_path, predictions, rank):
    model.predictions = predictions
    service = assessor(model, tmp_path)
    result = service.assess([route()])[0].steps[0].forward_prediction
    assert result == {
        "product_inchikey": inchi_key(smiles("CCO"), ignore_stereo=True),
        "rank": rank,
        "score": 1 / rank if rank else 0.0,
        "alert": rank is None,
    }
    saved = service.entries["CC.N"]
    assert len(saved["product_inchikeys"]) == len(set(saved["product_inchikeys"]))
    assert saved["ranking_scores"]
    assert all(saved["canonical_products"])
    assert "reason" not in result


@pytest.mark.parametrize(
    ("candidate", "matches"),
    [
        ("C[C@@H](O)F", True),
        ("CC(O)F", True),
        ("[13CH3]C(O)F", False),
        ("CC(O)Cl", False),
    ],
)
def test_matching_ignores_only_stereo(model, tmp_path, candidate, matches):
    model.predictions = [candidate] + ["N"] * 9
    source = route("C[C@H](O)F", "N[C@@H:2](C)C(=O)O.O")
    before = str(source.steps[0].reaction)
    result = assessor(model, tmp_path).assess([source])[0]
    assert result.steps[0].forward_prediction["alert"] is not matches
    assert result is not source and result.steps[0] is not source.steps[0]
    assert result.steps[0].reaction is source.steps[0].reaction
    assert result.steps[0].product is source.steps[0].product
    assert source.steps[0].forward_prediction is None
    assert "forward_prediction" not in source.steps[0].reaction.meta
    assert str(source.steps[0].reaction) == before
    assert model.calls[0][0] == ["C[C@H](N)C(=O)O.O"]


def test_designated_product_and_shared_reactants(model, tmp_path):
    model.predictions = ["CCO", "CN"] + ["N"] * 8
    source = [route(), route("CCO.CN", product_index=1), route()]
    service = assessor(model, tmp_path)
    result = service.assess(source)
    assert [r.steps[0].forward_prediction["rank"] for r in result] == [1, 2, 1]
    assert len(model.calls) == 1 and model.calls[0][0] == ["CC.N"]
    assert model.calls[0][1]["beam_width"] == 20
    assert service.last_stats["reaction_occurrences"] == 3
    assert service.last_stats["unique_inputs"] == 1
    assert service.last_stats["predictions_saved"] == 2


def test_candidate_canonicalization_keeps_tautomers():
    canonical, key = candidate("C=CO")
    assert "=" in canonical and key == inchi_key(smiles("C=CO"), ignore_stereo=True)
    assert candidate("CC>>CO") == (None, None)


def test_reopen_reuses_predictions_for_different_products(model, tmp_path):
    first = assessor(model, tmp_path)
    expected = first.assess([route()])
    model.calls.clear()
    second = assessor(model, tmp_path)
    actual = second.assess([route(), route("N")])
    assert (
        actual[0].steps[0].forward_prediction == expected[0].steps[0].forward_prediction
    )
    assert actual[1].steps[0].forward_prediction["rank"] == 2
    assert model.calls == []
    assert second.last_stats["cache_hits"] == 1
    assert second.last_stats["cache_misses"] == 0
    assert second.cache_path.is_file()


@pytest.mark.parametrize(
    "change",
    [
        "file",
        "runtime",
        "provider",
        "matching",
        "beam",
        "seed",
        "augmentation",
        "results",
        "pipeline",
    ],
)
def test_cache_fingerprint_invalidation(model, tmp_path, monkeypatch, change):
    old = assessor(model, tmp_path)
    old.assess([route()])
    if change == "file":
        (model.directory / "encoder.onnx").write_text("new graph")
    elif change == "runtime":
        monkeypatch.setattr(forward, "version", lambda name: "new version")
    elif change == "provider":
        model.providers = {"encoder": ["CUDAExecutionProvider"]}
    elif change == "matching":
        monkeypatch.setattr(forward, "ASSESSMENT_VERSION", 3)
    elif change == "seed":
        model.seed = 42
    elif change == "augmentation":
        model.augmentation_size = 1
    elif change == "results":
        monkeypatch.setattr(forward, "NUM_RESULTS", 5)
    elif change == "pipeline":
        monkeypatch.setattr(forward, "PIPELINE_VERSION", 3)
    else:
        model.decoder_batch_size = 32
    new = assessor(model, tmp_path)
    new.assess([route()])
    assert old.fingerprint != new.fingerprint
    assert len(model.calls) == 2
    assert len(json_io.read_json(new.cache_path)["models"]) == 2


def test_interrupted_atomic_write_keeps_completed_batches(model, tmp_path, monkeypatch):
    service = assessor(model, tmp_path, batch_size=1)
    service.assess([route(reactants="CC.N")])
    old = service.cache_path.read_bytes()
    with monkeypatch.context() as patch:

        def interrupt(*args):
            raise OSError("interrupted before replacement")

        patch.setattr(json_io.os, "replace", interrupt)
        with pytest.raises(OSError, match="interrupted"):
            service.assess([route(reactants="CCC.N")])
    assert service.cache_path.read_bytes() == old
    # An orphan from a killed process is never mistaken for the committed cache.
    (tmp_path / ".cache.json.orphan").write_text('{"incomplete":')
    resumed = assessor(model, tmp_path)
    model.calls.clear()
    resumed.assess([route(reactants="CC.N"), route(reactants="CCC.N")])
    assert model.calls[0][0] == ["CCC.N"]
    assert resumed.last_stats["cache_hits"] == 1


def test_inference_failure_preserves_previous_batch(model, tmp_path):
    predict = model.predict

    def interrupted(values, **kwargs):
        if values == ["CCC.N"]:
            raise RuntimeError("inference interrupted")
        return predict(values, **kwargs)

    model.predict = interrupted
    with pytest.raises(RuntimeError, match="inference interrupted"):
        assessor(model, tmp_path, batch_size=1).assess(
            [route(reactants="CC.N"), route(reactants="CCC.N")]
        )
    resumed = assessor(model, tmp_path)
    assert list(resumed.entries) == ["CC.N"]


@pytest.mark.parametrize("reactants", ["CC", "CC.N"])
def test_original_product_identification_errors_propagate(
    model, tmp_path, monkeypatch, reactants
):
    def fail(*args, **kwargs):
        raise ValueError("product identification failed")

    monkeypatch.setattr(forward, "inchi_key", fail)
    with pytest.raises(ValueError, match="product identification failed"):
        assessor(model, tmp_path).assess([route(reactants=reactants)])
    assert model.calls == []


@pytest.mark.parametrize("silent", [True, False])
def test_progress_totals_batches_and_order(
    model, tmp_path, monkeypatch, silent, capsys
):
    bars = []

    class Progress:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.updates = []
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, count=1):
            self.updates.append(count)

        write = staticmethod(print)

    monkeypatch.setattr(forward, "tqdm", Progress)
    source = [route(reactants=value) for value in ("CC.N", "CCC.N", "CO.N", "CC.N")]
    source.append(route(reactants="CC"))
    service = assessor(model, tmp_path, batch_size=2)
    result = service.assess(source, silent=silent)
    assert [r.steps[0].reaction for r in result] == [
        r.steps[0].reaction for r in source
    ]
    assert [bar.options["total"] for bar in bars] == [5, 3]
    assert [bar.options["disable"] for bar in bars] == [silent, silent]
    assert bars[0].updates == [1] * 5 and bars[1].updates == [2, 1]
    assert [call[0] for call in model.calls] == [["CC.N", "CCC.N"], ["CO.N"]]
    summary = capsys.readouterr().out
    assert bool(summary) is not silent
    assert "assumed" not in summary
    if not silent:
        assert "5 positive, 0 zero" in summary


def test_annotation_round_trip_does_not_touch_reaction_metadata(model, tmp_path):
    source = route()
    source.steps[0].reaction.meta["existing"] = {"confidence": 0.2}
    assessed = assessor(model, tmp_path).assess([source])[0]
    document = assessed.to_json()
    prediction = assessed.steps[0].forward_prediction
    assert document["children"][0]["meta"]["forward_prediction"] == prediction
    restored = Route.from_json(document)
    assert restored.steps[0].forward_prediction == prediction
    assert (
        restored.steps[0].reaction.meta["existing"]
        == source.steps[0].reaction.meta["existing"]
    )
    assert "forward_prediction" not in restored.steps[0].reaction.meta
    assert restored.to_json()["children"][0]["meta"]["forward_prediction"] == prediction
    assert "reason" not in json.dumps(document)
    cleared = replace(
        assessed, steps=(replace(assessed.steps[0], forward_prediction=None),)
    )
    assert "forward_prediction" not in cleared.to_json()["children"][0]["meta"]


def test_collection_preserves_all_metadata_and_branch_order(model, tmp_path):
    left, right = route("CCO"), route("CN", "N")
    root = route("CCOCN", "CCO.CN")
    root.steps[0].reaction._reactants = (left.target, right.target)
    tree = Route((*left.steps, *right.steps, *root.steps)).to_json()
    tree["user_metadata"] = {"search_score": 7, "notes": ["keep me"]}
    tree["children"][0]["children"].reverse()
    tree["children"][0]["meta"] = {"existing": 42}
    tree["children"][0]["children"][0]["supplier_note"] = "keep this too"
    collection = {"target-b": {"route-9": tree}, "target-a": [route().to_json()]}
    original = deepcopy(collection)
    output = assess_routes.assess_collection(collection, assessor(model, tmp_path))
    assert collection == original
    assert list(output) == list(collection)

    def remove_annotations(node):
        if node["type"] == "reaction":
            annotation = node["meta"].pop("forward_prediction")
            assert set(annotation) == {"product_inchikey", "rank", "score", "alert"}
            if not node["meta"]:
                del node["meta"]
        for child in node.get("children", []):
            remove_annotations(child)

    remove_annotations(output["target-b"]["route-9"])
    remove_annotations(output["target-a"][0])
    assert output == original


def test_script_gzip_input_output_and_default_cache(model, tmp_path, monkeypatch):
    source, output = tmp_path / "input.json.gz", tmp_path / "results/output.json.gz"
    collection = {"target": [route().to_json()]}
    json_io.write_json_atomic(source, collection)
    monkeypatch.setattr(assess_routes, "OnnxChemformer", lambda *a, **k: model)
    monkeypatch.setattr(
        "sys.argv",
        [
            "assess_routes",
            "--input",
            str(source),
            "--output",
            str(output),
            "--model",
            str(model.directory),
            "--quiet",
        ],
    )
    assess_routes.main()
    assert json_io.read_json(source) == collection
    restored = json_io.read_json(output)
    assert (
        restored["target"][0]["children"][0]["meta"]["forward_prediction"]["rank"] == 1
    )
    assert (output.parent / "forward_cache.json").exists()


def test_mapping_equivalence_deduplicates(model, tmp_path):
    service = assessor(model, tmp_path)
    service.assess([route(reactants="[CH3:7][CH3:8].N"), route(reactants="CC.N")])
    assert model.calls[0][0] == [canonical_reactants("CC.N")]


def test_one_reactant_assumption_bypasses_preprocessing_and_inference(
    model, tmp_path, monkeypatch
):
    def unexpected(*args, **kwargs):
        raise RuntimeError("one-reactant steps must bypass prediction")

    monkeypatch.setattr(forward, "canonical_reactants", unexpected)
    model.predict = unexpected
    source = route("CCO.CN", "N[C@@H:2](C)C(=O)O", product_index=1)
    chemistry = str(source.steps[0].reaction)
    service = assessor(model, tmp_path)
    result = service.assess([source])[0]
    annotation = result.steps[0].forward_prediction
    assert annotation == {
        "product_inchikey": inchi_key(source.steps[0].product, ignore_stereo=True),
        "rank": None,
        "score": 1.0,
        "alert": False,
    }
    assert Route.from_json(result.to_json()).steps[0].forward_prediction == annotation
    assert str(result.steps[0].reaction) == chemistry
    assert source.steps[0].forward_prediction is None
    assert service.last_stats["single_reactant_steps"] == 1
    assert service.last_stats["model_assessed_occurrences"] == 0
    assert service.last_stats["matched"] == service.last_stats["unmatched"] == 0
    assert service.last_stats["nonzero_scored"] == 1
    assert service.last_stats["predictions_saved"] == 0
    assert not service.cache_path.exists()


def test_old_single_reactant_cache_ignored_and_other_predictions_reused(
    model, tmp_path
):
    service = assessor(model, tmp_path)
    service.assess([route()])
    service.entries["CC"] = {"product_inchikeys": [None] * 10}
    json_io.write_json_atomic(service.cache_path, service.cache)
    previous = service.cache_path.read_bytes()
    model.calls.clear()
    reopened = assessor(model, tmp_path)
    result = reopened.assess([route(reactants="CC"), route(), route()])
    assert [r.steps[0].forward_prediction["score"] for r in result] == [1.0] * 3
    assert model.calls == []
    assert service.fingerprint == reopened.fingerprint
    assert reopened.cache_path.read_bytes() == previous
    assert reopened.last_stats["cache_hits"] == 1
    assert reopened.last_stats["predictions_saved"] == 2
    assert reopened.last_stats["single_reactant_steps"] == 1
    assert reopened.last_stats["matched"] == 2


def test_two_identical_reactants_still_require_prediction(model, tmp_path):
    model.predictions = ["N"] * 10
    service = assessor(model, tmp_path)
    result = service.assess([route(reactants="CC.CC")])[0]
    assert model.calls[0][0] == ["CC.CC"]
    assert result.steps[0].forward_prediction["score"] == 0.0
    assert service.last_stats["single_reactant_steps"] == 0


def test_reaction_reassessment_clears_stale_route_scores(model, tmp_path):
    source = replace(route(), assessment={"forward_score": 0.1, "combined_score": 0.01})
    service = assessor(model, tmp_path)
    result = service.assess([source])[0]
    assert result.assessment is None
    assert source.assessment["forward_score"] == 0.1
    collection = {"target": {"route": source.to_json()}}
    updated = assess_routes.assess_collection(collection, service)
    assert "assessment" not in updated["target"]["route"]
    assert "assessment" in collection["target"]["route"]
