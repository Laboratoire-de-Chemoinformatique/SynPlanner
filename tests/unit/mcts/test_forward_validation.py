"""Forward checks gate real search edges and preserve planning/export semantics."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from chython import smiles
from test_tree_stats import FakePriorityReactor, FakeReactor, build_tree, make_mol

from synplan.chem.reaction.routes.quality.forward import ForwardReactionAssessor
from synplan.mcts.config import ForwardValidationConfig
from synplan.mcts.forward import ForwardValidationError, load_forward_validation
from synplan.mcts.record import read_search_record, write_search_record
from synplan.mcts.search import build_target_routes
from synplan.ml.inference.forward import aggregate
from synplan.utils.visualisation import extract_routes


@pytest.fixture
def model(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "encoder.onnx").write_text("fixture")
    model = SimpleNamespace(
        directory=directory,
        backend="onnx",
        runtime_files=("encoder.onnx",),
        augmentation_size=10,
        seed=0,
        max_length=512,
        decoder_batch_size=64,
        providers={"encoder": ["CPUExecutionProvider"]},
        calls=[],
        predictions=["N", "CCCCCCC"],
    )

    def predict(values, **kwargs):
        model.calls.append((list(values), kwargs))
        return [
            aggregate(
                value,
                [
                    {
                        "predictions": model.predictions,
                        "log_likelihoods": [
                            -float(i) for i in range(len(model.predictions))
                        ],
                    }
                ],
                num_results=10,
            )
            for value in values
        ]

    model.predict = predict
    return model


def service(model, tmp_path, **kwargs):
    return ForwardReactionAssessor(model, tmp_path / "cache.json", **kwargs)


def two_reactant_tree(**kwargs):
    return build_tree(
        rules=[(0.7, FakeReactor(lambda: [make_mol(3), make_mol(4)]), 0)],
        max_iterations=1,
        **kwargs,
    )


def test_disabled_configs_load_nothing(tmp_path, monkeypatch):
    import synplan.ml.inference.chemformer as module

    def fail(*args, **kwargs):
        pytest.fail("Disabled validation loaded a model")

    monkeypatch.setattr(module, "OnnxChemformer", fail)
    path = tmp_path / "forward.yaml"
    path.write_text("forward_val_type: null\nforward_val_path: missing\n")
    for config in (None, path, ForwardValidationConfig()):
        assert load_forward_validation(config, tmp_path) == (None, None)
    assert not (tmp_path / "forward_cache.json").exists()


def test_paths_and_config_validation(tmp_path):
    path = tmp_path / "config/forward.yaml"
    path.parent.mkdir()
    path.write_text(
        "forward_val_type: on-the-fly\nforward_val_path: ../model\ncache_path: cache.json\ntop_k: 3\n"
    )
    config = ForwardValidationConfig.from_yaml(path)
    assert config.forward_val_path == tmp_path / "model"
    assert config.cache_path == path.parent / "cache.json"
    assert config.top_k == 3
    for cutoff in (0, 11):
        with pytest.raises(ValueError):
            ForwardValidationConfig(top_k=cutoff)
    with pytest.raises(ValueError, match="forward_val_path"):
        load_forward_validation(ForwardValidationConfig(forward_val_type="on-the-fly"))


@pytest.mark.parametrize("top_k,rank", [(1, None), (2, 2), (10, 2)])
def test_cutoff_uses_same_prediction_identity(model, tmp_path, top_k, rank):
    original = service(model, tmp_path)
    reactants, expected = [[smiles("CCC"), smiles("CCCC")]], [smiles("CCCCCCC")]
    original.assess_reactions(reactants, expected)
    reopened = service(model, tmp_path, top_k=top_k)
    result = reopened.assess_reactions(reactants, expected)[0]
    assert result["rank"] == rank
    assert result["alert"] is (rank is None)
    assert reopened.fingerprint == original.fingerprint
    assert len(model.calls) == 1
    assert model.calls[0][1]["num_results"] == 10


def test_different_expected_products_share_predictions(model, tmp_path):
    assessor = service(model, tmp_path)
    inputs = [[smiles("CCC"), smiles("CCCC")]] * 2
    result = assessor.assess_reactions(inputs, [smiles("N"), smiles("CCCCCCC")])
    assert [r["rank"] for r in result] == [1, 2]
    assert len(model.calls) == 1 and len(model.calls[0][0]) == 1


def test_filter_precedes_node_pruning_and_evaluation(model, tmp_path, monkeypatch):
    tree = two_reactant_tree()
    tree.config.enable_pruning = True
    tree.forward_assessor = service(model, tmp_path, top_k=1)
    initial = deepcopy(
        tree.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks
    )

    def fail(*args, **kwargs):
        pytest.fail("Rejected candidate reached child creation or evaluation")

    monkeypatch.setattr(tree, "_add_node", fail)
    monkeypatch.setattr(tree, "_get_node_value", fail)
    with pytest.raises(StopIteration):
        tree._expand_node(1)
    assert len(tree.nodes) == 1 and tree.curr_tree_size == 2
    assert tree.children[1] == tree.redundant_children[1] == set()
    assert (
        tree.big_dict_of_all_tuples_of_precursors_to_expand_but_not_building_blocks
        == initial
    )
    assert not tree.winning_nodes
    assert tree.forward_stats["rejections"] == 1
    assert len(model.calls[0][0][0].split(".")) == 2


@pytest.mark.parametrize("strategy", ["expansion_first", "evaluation_first"])
def test_passes_preserve_scoring_and_exports(model, tmp_path, strategy):
    original = two_reactant_tree(search_strategy=strategy).run()
    tree = two_reactant_tree(search_strategy=strategy)
    tree.forward_assessor = service(model, tmp_path)
    tree.run()
    for node_id, node in tree.nodes.items():
        before = original.nodes[node_id]
        assert (
            node.prob,
            node.policy_probability,
            node.init_value,
            node.total_value,
        ) == (
            before.prob,
            before.policy_probability,
            before.init_value,
            before.total_value,
        )
    routes = tree.finalize_forward(tree.forward_assessor, "on-the-fly")
    assert len(model.calls) == 1
    assert routes[0].assessment["forward_score"] == 0.5
    assert routes[0].steps[0].forward_prediction["rank"] == 2
    assert "forward_prediction" not in routes[0].steps[0].reaction.meta
    assert [str(s.reaction) for s in routes[0]] == [
        str(s.reaction) for s in original.routes()[0]
    ]
    restored = read_search_record(write_search_record(tree, tmp_path / "tree.json.gz"))
    assert [r.to_json() for r in restored.routes()] == [r.to_json() for r in routes]
    assert (
        restored.forward_metadata["model_fingerprint"]
        == tree.forward_assessor.fingerprint
    )
    for doc in (build_target_routes(tree)[0], extract_routes(tree)[0]):
        assert doc["assessment"]["forward_score"] == 0.5
        assert doc["children"][0]["meta"]["forward_prediction"]["rank"] == 2
    assert len(model.calls) == 1


def test_one_reactant_pass_bypasses_model(model, tmp_path):
    model.predict = lambda *a, **k: pytest.fail("Assumed step invoked model")
    tree = build_tree(max_iterations=1)
    tree.forward_assessor = service(model, tmp_path)
    tree.run()
    routes = tree.finalize_forward(tree.forward_assessor, "on-the-fly")
    assert routes[0].assessment["forward_score"] == 1
    assert tree.forward_stats["assumed_passes"] == 1


def test_post_processing_retains_failures_and_order(model, tmp_path):
    model.predictions = ["N"]
    tree = two_reactant_tree().run()
    source = [r.to_json() for r in tree.routes()]
    assessor = service(model, tmp_path)
    result = tree.finalize_forward(assessor, "post-processing")
    assert len(result) == len(source) == 1
    assert result[0].assessment["forward_score"] == 0
    assert result[0].provenance.search_score == source[0]["provenance"]["search_score"]
    assert not tree.forward_assessor
    assert [r.to_json() for r in tree.routes()] == [r.to_json() for r in result]


def test_priority_outcomes_are_filtered(model, tmp_path):
    tree = build_tree(
        rules=[],
        priority_rules={
            "custom": [FakePriorityReactor(lambda: [make_mol(3), make_mol(4)])]
        },
        use_priority=True,
    )
    tree.forward_assessor = service(model, tmp_path, top_k=1)
    tree.run()
    assert not tree.winning_nodes and len(tree.nodes) == 1
    assert tree.forward_stats["rejections"] == 1


def test_errors_propagate_instead_of_rejecting(model, tmp_path):
    tree = two_reactant_tree()

    def fail(*args, **kwargs):
        raise RuntimeError("broken session")

    model.predict = fail
    tree.forward_assessor = service(model, tmp_path)
    with pytest.raises(ForwardValidationError, match="broken session"):
        tree.run()
    assert tree.forward_stats.get("rejections", 0) == 0


def test_finite_limit_starts_no_second_prediction(model, tmp_path, monkeypatch):
    import synplan.mcts.tree as module

    clock = [100.0]
    monkeypatch.setattr(module, "time", lambda: clock[0])
    tree = build_tree(
        rules=[
            (0.5, FakeReactor(lambda: [make_mol(3), make_mol(4)]), 0),
            (0.4, FakeReactor(lambda: [make_mol(2), make_mol(5)]), 1),
        ]
    )
    original = model.predict

    def slow(*args, **kwargs):
        clock[0] += 20
        return original(*args, **kwargs)

    model.predict = slow
    tree.forward_assessor = service(model, tmp_path)
    tree.run()
    assert len(model.calls) == 1
    assert tree.stop_reason == "max_time"
    assert tree.curr_time == 20


@pytest.mark.parametrize("mode", [None, "post-processing", "on-the-fly"])
def test_planning_runner_modes_and_shared_sessions(model, tmp_path, monkeypatch, mode):
    import synplan.mcts.search as search
    import synplan.ml.inference.chemformer as chemformer
    from synplan.utils.json import read_json

    template = two_reactant_tree()
    monkeypatch.setattr(
        search, "load_reaction_rules", lambda *a, **k: template.reaction_rules
    )
    monkeypatch.setattr(
        search, "load_building_blocks", lambda *a, **k: template.building_blocks
    )
    monkeypatch.setattr(
        search, "load_policy_function", lambda *a, **k: template.expansion_function
    )
    monkeypatch.setattr(
        search, "load_evaluation_function", lambda *a, **k: template.evaluator
    )
    loaded = []

    def load(*args, **kwargs):
        loaded.append(args)
        return model

    monkeypatch.setattr(chemformer, "OnnxChemformer", load)
    targets = tmp_path / "targets.smi"
    targets.write_text("CCCCCCC\nCCCCCCC\n")
    output = tmp_path / "results"
    config = (
        ForwardValidationConfig(forward_val_type=mode, forward_val_path=model.directory)
        if mode
        else None
    )
    search.run_search(
        str(targets),
        template.config.to_dict(),
        None,
        None,
        "rules",
        "stock",
        results_root=output,
        export_routes=True,
        forward_val_config=config,
    )
    public = next(iter(read_json(output / "results.json.gz").values()))[0]
    if mode is None:
        assert not loaded and not model.calls
        assert "assessment" not in public
        assert not (output / "forward_cache.json").exists()
    else:
        assert len(loaded) == len(model.calls) == 1
        assert public["assessment"]["forward_score"] == 0.5
        assert public["children"][0]["meta"]["forward_prediction"]["rank"] == 2
        raw = read_json(output / "extracted_routes_html/routes_0_unassessed.json.gz")
        if mode == "post-processing":
            assert (
                "forward_prediction"
                not in next(iter(raw["CCCCCCC"].values()))["children"][0]["meta"]
            )
        saved = read_json(output / "extracted_routes_html/mapped_routes_0.json")
        assert next(iter(saved.values()))["assessment"]["forward_score"] == 0.5
        manifest = read_json(output / "forward_validation.json")
        assert len(manifest["targets"]) == 2
        assert manifest["configuration"]["cache_path"] == str(
            output / "forward_cache.json"
        )
        assert (output / "extracted_routes_html/search_0_unassessed.json.gz").exists()


def test_post_failure_keeps_completed_search(model, tmp_path, monkeypatch):
    import synplan.mcts.search as search
    import synplan.ml.inference.chemformer as chemformer

    template = two_reactant_tree()
    for name, value in [
        ("load_reaction_rules", template.reaction_rules),
        ("load_building_blocks", template.building_blocks),
        ("load_policy_function", template.expansion_function),
        ("load_evaluation_function", template.evaluator),
    ]:
        monkeypatch.setattr(search, name, lambda *a, _value=value, **k: _value)
    monkeypatch.setattr(chemformer, "OnnxChemformer", lambda *a, **k: model)

    def fail(*args, **kwargs):
        raise RuntimeError("inference unavailable")

    model.predict = fail
    targets = tmp_path / "targets.smi"
    targets.write_text("CCCCCCC\n")
    output = tmp_path / "results"
    with pytest.raises(RuntimeError, match="inference unavailable"):
        search.run_search(
            str(targets),
            template.config.to_dict(),
            None,
            None,
            "rules",
            "stock",
            results_root=output,
            forward_val_config=ForwardValidationConfig(
                forward_val_type="post-processing", forward_val_path=model.directory
            ),
        )
    record = read_search_record(
        output / "extracted_routes_html/search_0_unassessed.json.gz"
    )
    assert len(record.routes()) == 1
    assert record.routes()[0].assessment is None


@pytest.mark.parametrize("reference", [None, "forward_predictor.yaml"])
def test_cli_resolves_top_level_reference(tmp_path, monkeypatch, reference):
    import yaml

    import synplan.interfaces.cli as cli

    folder = tmp_path / "configuration"
    folder.mkdir()
    config_path = folder / "planning.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "tree": {},
                "node_expansion": {},
                "node_evaluation": {"evaluation_type": "random"},
                "forward_val_config": reference,
            }
        )
    )
    calls = []
    monkeypatch.setattr(cli, "run_search", lambda **kwargs: calls.append(kwargs))
    monkeypatch.chdir(tmp_path)
    cli.planning_cli.callback(
        config_path=str(config_path),
        targets="targets.smi",
        reaction_rules="rules",
        building_blocks="stock",
        policy_network="policy.onnx",
        value_network=None,
        results_dir="results",
        reconcile_mapping=False,
        export_routes=False,
    )
    assert len(calls) == 1
    assert "forward_val_config" not in calls[0]["search_config"]
    assert calls[0]["forward_val_config"] == (folder / reference if reference else None)
