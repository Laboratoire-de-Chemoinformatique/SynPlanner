"""Forward checks gate real search edges and preserve planning/export semantics."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from chython import inchi_key, smiles
from test_tree_stats import (
    FakePolicy,
    FakePriorityReactor,
    FakeReactor,
    FixedEvaluationStrategy,
    build_tree,
    make_mol,
)

from synplan.chem.precursor import Precursor
from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.reaction.routes.quality.forward import ForwardReactionAssessor
from synplan.chem.reaction.routes.route import Route
from synplan.mcts.config import ForwardValidationConfig, TreeConfig
from synplan.mcts.forward import ForwardValidationError, load_forward_validation
from synplan.mcts.node import Node
from synplan.mcts.record import read_search_record, write_search_record
from synplan.mcts.search import build_target_routes
from synplan.mcts.tree import Tree
from synplan.ml.inference.forward import aggregate, prediction_identity
from synplan.utils.files import read_json, write_json_atomic
from synplan.utils.visualisation import extract_routes, routes_report_html


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

    model.prediction_identity = lambda **kwargs: prediction_identity(model, **kwargs)
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
    import synplan.ml.inference as module

    def fail(*args, **kwargs):
        pytest.fail("Disabled validation loaded a model")

    monkeypatch.setattr(module, "ForwardPredictor", fail)
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


@pytest.mark.parametrize("legacy", [None, 4, 5])
def test_record_stores_aggregates_on_routes_only(model, tmp_path, legacy):
    tree = two_reactant_tree().run()
    assessed = tree.finalize_forward(
        service(model, tmp_path), "post-processing", combine_search=True
    )
    expected = [route.to_json() for route in assessed]
    path = write_search_record(tree, tmp_path / "record.json")
    raw = read_json(path)
    assert raw["schema"] == "synplan-tree/6"
    assert all("route_assessment" not in entry for entry in raw["nodes"])
    assert raw["route_assessments"] == {
        str(route.provenance.tree_node_id): route.assessment for route in assessed
    }
    if legacy:
        raw["schema"] = f"synplan-tree/{legacy}"
        raw.pop("bonds_state")
        for entry in raw["nodes"]:
            for field in (
                "new_provenance",
                "expand_provenance",
                "remaining_required_bonds",
            ):
                entry.pop(field)
        if legacy == 4:
            assessments = raw.pop("route_assessments")
            for entry in raw["nodes"]:
                entry["route_assessment"] = assessments.get(str(entry["id"]))
        write_json_atomic(path, raw)

    model.predict = lambda *a, **k: pytest.fail("Reading scores invoked inference")
    record = read_search_record(path)
    assert [route.to_json() for route in record.routes()] == expected
    html = routes_report_html(record.routes(), None)
    assert "Forward route score: 0.5" in html
    assert "<th>Combined score</th>" in html
    assert all(not hasattr(node, "route_assessment") for node in tree.nodes.values())
    assert all(not hasattr(node, "route_assessment") for node in record.nodes.values())
    rewritten = write_search_record(record, tmp_path / "rewritten.json.gz")
    assert read_json(rewritten)["schema"] == "synplan-tree/6"
    assert [r.to_json() for r in read_search_record(rewritten).routes()] == expected
    # Returned objects must not mutate the retained route assessment.
    assessed[0].assessment["forward_score"] = 99
    record.routes()[0].assessment["forward_score"] = 99
    assert [r.to_json() for r in tree.routes()] == expected
    assert [r.to_json() for r in record.routes()] == expected


def test_shared_prefix_keeps_route_aggregates_independent(model, tmp_path):
    class BranchingPolicy(FakePolicy):
        def predict_reaction_rules(self, precursor, reaction_rules):
            yield from (
                self.rules[:1] if len(precursor.molecule) == 7 else self.rules[1:]
            )

    tree = build_tree(
        rules=[
            (0.8, FakeReactor(lambda: [make_mol(8)]), 0),
            (0.6, FakeReactor(lambda: [make_mol(3), make_mol(5)]), 1),
            (0.4, FakeReactor(lambda: [make_mol(2), make_mol(6)]), 2),
        ],
        policy_cls=BranchingPolicy,
        expand_deeper=True,
    ).run()
    # Include both terminal siblings in the exported collection.
    tree.winning_nodes = [i for i, node in tree.nodes.items() if node.is_solved()]
    left, right = tree.winning_nodes
    shared = tree.parents[left]
    assert shared == tree.parents[right] and shared != 1
    for node_id, rank, score in (
        (shared, None, 1.0),
        (left, 4, 0.25),
        (right, None, 0.0),
    ):
        product = tree.nodes[tree.parents[node_id]].curr_precursor.molecule
        tree.nodes[node_id].forward_prediction = {
            "product_inchikey": inchi_key(product, ignore_stereo=True),
            "rank": rank,
            "score": score,
            "alert": score == 0,
        }
    model.predict = lambda *a, **k: pytest.fail("Aggregation invoked inference")
    finalized = tree.finalize_forward(service(model, tmp_path), "on-the-fly")
    expected = {left: 0.5, right: 0.0}
    assert {
        r.provenance.tree_node_id: r.assessment["forward_score"] for r in finalized
    } == expected
    record = read_search_record(write_search_record(tree, tmp_path / "branches.json"))
    for source in (tree, record):
        assert Route.from_tree(source, shared).assessment is None
        for node_id, score in expected.items():
            assert Route.from_tree(source, node_id).assessment["forward_score"] == score
        assert all(
            not hasattr(node, "route_assessment") for node in source.nodes.values()
        )


@pytest.mark.parametrize("mode", ["on-the-fly", "post-processing"])
def test_reconstructed_branches_keep_scores_and_origins(model, tmp_path, mode):
    target, left, right, carbon_left, oxygen, carbon_right, nitrogen = (
        Precursor(smiles(value), canonicalize=False)
        for value in (
            "[CH3:1][CH2:2][O:3][CH2:4][NH2:5]",
            "[CH3:1][CH2:2][OH:3]",
            "[CH3:4][NH2:5]",
            "[CH3:1][CH3:2]",
            "[OH2:3]",
            "[CH4:4]",
            "[NH3:5]",
        )
    )
    tree = build_tree(rules=[])
    tree.original_target = target.molecule
    tree.nodes = {
        1: Node((target,), (target,)),
        2: Node((left, right), (left, right)),
        3: Node((right,), (carbon_left, oxygen)),
        4: Node((), (carbon_right, nitrogen)),
    }
    tree.parents = {1: 0, 2: 1, 3: 2, 4: 3}
    tree.children = {1: {2}, 2: {3}, 3: {4}, 4: set()}
    tree.winning_nodes = [4]
    tree.curr_tree_size = 5
    expected = {}
    for node_id, precursor, rank in ((2, target, 1), (3, left, 2), (4, right, 4)):
        node = tree.nodes[node_id]
        node.rule_id, node.rule_source, node.rule_key = (
            node_id,
            "policy",
            f"policy:{node_id}",
        )
        expected[node_id] = {
            "product_inchikey": inchi_key(precursor.molecule, ignore_stereo=True),
            "rank": rank,
            "score": 1 / rank,
            "alert": False,
        }
    original = Route.from_tree(tree, 4)
    assert [step.origin.tree_node_id for step in original] == [3, 4, 2]
    # Stereo reconstruction retains origins in branch order; node annotations
    # may arrive later and must not be read from this saved copy.
    tree.nodes[4].reconstructed_route = deepcopy(original)
    if mode == "on-the-fly":
        for node_id, annotation in expected.items():
            tree.nodes[node_id].forward_prediction = deepcopy(annotation)
    model.predictions = ["CCOCN", "CCO", "O", "CN"]
    assessed = tree.finalize_forward(service(model, tmp_path), mode)[0]
    assert bool(model.calls) is (mode == "post-processing")
    record = read_search_record(write_search_record(tree, tmp_path / "record.json.gz"))
    restored = Route.from_json(assessed.to_json())
    for route in (assessed, tree.routes()[0], record.routes()[0], restored):
        assert route.assessment["forward_score"] == pytest.approx(0.5)
        assert [step.origin for step in route] == [step.origin for step in original]
        assert [str(step.reaction) for step in route] == [
            str(step.reaction) for step in original
        ]
        for step in route:
            assert step.forward_prediction == expected[step.origin.tree_node_id]
            assert step.forward_prediction["product_inchikey"] == inchi_key(
                step.product, ignore_stereo=True
            )
    for source in (tree, record):
        for node_id, annotation in expected.items():
            assert source.nodes[node_id].forward_prediction == annotation


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
    import synplan.ml.inference as chemformer
    from synplan.utils.files import read_json

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

    monkeypatch.setattr(chemformer, "ForwardPredictor", load)
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
    import synplan.ml.inference as chemformer

    template = two_reactant_tree()
    for name, value in [
        ("load_reaction_rules", template.reaction_rules),
        ("load_building_blocks", template.building_blocks),
        ("load_policy_function", template.expansion_function),
        ("load_evaluation_function", template.evaluator),
    ]:
        monkeypatch.setattr(search, name, lambda *a, _value=value, **k: _value)
    monkeypatch.setattr(chemformer, "ForwardPredictor", lambda *a, **k: model)

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


@pytest.mark.parametrize("stereo_mode", ["proposal", "off"])
@pytest.mark.parametrize(
    "bonds_state,top_k,solved,checks",
    [
        ({(3, 4): 1, (1, 2): 2}, 2, True, 1),
        ({(3, 4): 1, (1, 2): 2}, 1, False, 1),
        ({(3, 4): 2}, 2, False, 0),
        ({(1, 2): 1}, 2, False, 1),
    ],
)
def test_bond_constraints_with_forward_filter_and_record(
    model, tmp_path, stereo_mode, bonds_state, top_k, solved, checks
):
    """Real cleavage must pass both gates, with both kinds of state surviving export."""
    target = smiles("[CH3:1][CH2:2][CH2:3][OH:4]")
    rule = CanonicalRetroReactor.from_smarts("[C;D2:1]-[O;D1:2]>>[C;D1:1].[O;D0:2]")
    model.predictions = ["N", str(target)]
    assessor = service(model, tmp_path, top_k=top_k)
    tree = Tree(
        target=target,
        config=TreeConfig(
            algorithm="breadth_first",
            max_iterations=1,
            max_time=None,
            min_mol_size=0,
            silent=True,
            stereo_mode=stereo_mode,
        ),
        reaction_rules=[rule],
        building_blocks={"CCC", "O"},
        expansion_function=FakePolicy([(0.7, rule, 0)]),
        evaluation_function=FixedEvaluationStrategy(),
        forward_assessor=assessor,
        bonds_state=bonds_state,
    ).run()
    assert bool(tree.winning_nodes) is solved
    assert tree.forward_stats.get("candidate_checks", 0) == checks
    assert len(model.calls) == checks
    if solved:
        node = tree.nodes[tree.winning_nodes[0]]
        assert not node.remaining_required_bonds
        assert node.forward_prediction["rank"] == 2
        assert node.policy_probability == 0.7
        routes = tree.finalize_forward(assessor, "on-the-fly")
        assert routes[0].assessment["forward_score"] == 0.5
        assert len(model.calls) == 1
    else:
        assert len(tree.nodes) == 1

    restored = read_search_record(write_search_record(tree, tmp_path / "bonds.json.gz"))
    assert restored.bonds_state == bonds_state
    assert restored.config.stereo_mode == stereo_mode
    assert [r.to_json() for r in restored.routes()] == [
        r.to_json() for r in tree.routes()
    ]
    for node_id, node in tree.nodes.items():
        saved = restored.nodes[node_id]
        assert saved.remaining_required_bonds == node.remaining_required_bonds
        assert saved.forward_prediction == node.forward_prediction
        assert [p.target_atom_provenance for p in saved.new_precursors] == [
            p.target_atom_provenance for p in node.new_precursors
        ]
