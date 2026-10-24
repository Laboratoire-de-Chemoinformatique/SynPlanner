"""Route aggregation, persistence, collection scoring and offline reporting."""

import json
from copy import deepcopy
from dataclasses import replace
from io import StringIO

import pytest
from chython import inchi_key, smiles
from tqdm import tqdm

from scripts import score_routes
from synplan.chem.reaction.routes.quality import ForwardRouteScorer
from synplan.chem.reaction.routes.route import Route, RouteProvenance, Step
from synplan.utils.json import read_json, write_json_atomic
from synplan.utils.visualisation import routes_report_html


def annotated_step(product, reactants, score, *, assumed=False):
    reaction = smiles(f"{reactants}>>{product}")
    return Step(
        reaction,
        reaction.products[0],
        forward_prediction={
            "product_inchikey": inchi_key(reaction.products[0], ignore_stereo=True),
            "rank": None if not score or assumed else round(1 / score),
            "score": score,
            "alert": score == 0,
        },
    )


def linear(scores, *, assumed=(), search=0.8):
    steps = []
    for i, score in enumerate(scores):
        product = "C" * (i + 2)
        reactants = "C" if i in assumed else "C.N"
        step = annotated_step(product, reactants, score, assumed=i in assumed)
        if steps:
            step.reaction._reactants = (
                steps[-1].product,
                *step.reaction.reactants[1:],
            )
        steps.append(step)
    return Route(tuple(steps), provenance=RouteProvenance(search, 42, 0, -2.5))


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.25], 0.25),
        ([0.25, 1.0], 0.5),
        ([0.1, 0.1], 0.1),
        ([0.25, 0.5, 1.0], 0.5),
        ([0.0, 1.0], 0.0),
        ([1.0], 1.0),
        ([1e-200, 1e-200], 1e-200),
    ],
)
def test_geometric_mean_and_numerical_stability(scores, expected):
    actual = ForwardRouteScorer().score(linear(scores))
    assert actual == pytest.approx(expected, abs=0, rel=1e-12)


def test_convergent_branches_count_final_step_once():
    left = annotated_step("CC", "C.N", 0.25)
    right = annotated_step("CO", "C.O", 0.5)
    root = annotated_step("CCO", "CC.CO", 1.0)
    root.reaction._reactants = (left.product, right.product)
    route = Route((root, right, left))
    score = ForwardRouteScorer().assess([route])[0].assessment
    assert score["total_steps"] == 3
    assert score["forward_score"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("scores", "assumed", "expected"),
    [
        ([0.25, 1.0], (1,), 0.5),
        ([1.0, 1.0], (0, 1), 1.0),
    ],
)
def test_assumed_steps_included(scores, assumed, expected):
    result = ForwardRouteScorer().assess([linear(scores, assumed=assumed)])[0]
    assert result.assessment["assumed_one_reactant_steps"] == len(assumed)
    assert result.assessment["forward_score"] == pytest.approx(expected)


def test_assessment_copies_preserve_order_chemistry_and_provenance():
    source = [linear([0.25, 1.0], assumed=(1,)), linear([0.0])]
    before = [route.to_json() for route in source]
    result = ForwardRouteScorer().assess(iter(source))
    for original, assessed in zip(source, result, strict=True):
        assert assessed is not original and original.assessment is None
        assert assessed.provenance is original.provenance
        assert all(a is b for a, b in zip(original.steps, assessed.steps))
    assert [route.to_json() for route in source] == before
    assert result[0].assessment == {
        "aggregation": "geometric_mean_all_steps",
        "forward_score": 0.5,
        "search_score": 0.8,
        "total_steps": 2,
        "zero_score_steps": 0,
        "assumed_one_reactant_steps": 1,
    }
    assert result[1].assessment["zero_score_steps"] == 1


def test_combination_is_optional_and_uses_original_scale():
    route = linear([0.25, 1.0], search=3.0)
    scorer = ForwardRouteScorer(combine_search=True)
    result = scorer.assess([route])[0]
    assert scorer.score(route) == result.assessment["combined_score"] == 1.5
    assert result.assessment["search_score"] == 3.0
    assert result.provenance.policy_log_likelihood == -2.5
    # Reassessment with the default removes a previously requested combination.
    assert "combined_score" not in ForwardRouteScorer().assess([result])[0].assessment


@pytest.mark.parametrize("search", [None, -0.1, float("nan"), float("inf")])
def test_combination_requires_finite_nonnegative_search(search):
    route = linear([0.0], search=search)
    assert ForwardRouteScorer().score(route) == 0.0
    with pytest.raises(ValueError, match="finite, nonnegative search"):
        ForwardRouteScorer(combine_search=True).score(route)


def test_no_provenance_and_missing_annotation():
    route = replace(linear([0.25]), provenance=None)
    result = ForwardRouteScorer().assess([route])[0]
    assert result.assessment["forward_score"] == 0.25
    assert result.assessment["search_score"] is None
    with pytest.raises(ValueError, match="search"):
        ForwardRouteScorer(combine_search=True).score(route)
    partial = linear([0.0, 1.0])
    partial = replace(
        partial,
        steps=(partial.steps[0], replace(partial.steps[1], forward_prediction=None)),
    )
    with pytest.raises(ValueError, match=r"Step 2.*no forward"):
        ForwardRouteScorer().score(partial)


def test_rank_is_explicit_and_stable_for_both_modes():
    a, b, c = (
        linear([0.25], search=0.8),
        linear([1.0], search=0.1),
        linear([0.25], search=0.8),
    )
    assert ForwardRouteScorer().rank([a, b, c]) == [b, a, c]
    assert ForwardRouteScorer(combine_search=True).rank([a, b, c]) == [a, c, b]


def test_json_round_trip_and_legacy_without_metadata():
    assessed = ForwardRouteScorer(combine_search=True).assess([linear([0.25])])[0]
    tree = assessed.to_json()
    restored = Route.from_json(tree)
    assert restored.provenance == assessed.provenance
    assert restored.assessment == assessed.assessment
    exported = restored.to_json()
    assert exported["assessment"] == tree["assessment"]
    assert exported["provenance"] == tree["provenance"]
    assert restored.steps[0].forward_prediction == assessed.steps[0].forward_prediction
    restored.assessment["forward_score"] = 0.7
    assert tree["assessment"]["forward_score"] == 0.25
    legacy = deepcopy(tree)
    legacy.pop("assessment")
    legacy.pop("provenance")
    restored = Route.from_json(legacy)
    assert restored.assessment is None
    assert restored.provenance.search_score is None
    assert ForwardRouteScorer().score(restored) == 0.25


@pytest.mark.parametrize("silent", [True, False])
def test_assess_progress_and_empty_collection(monkeypatch, silent):
    bars = []
    stream = StringIO()

    def progress(**kwargs):
        bar = tqdm(**kwargs, file=stream, mininterval=0)
        bars.append(bar)
        return bar

    monkeypatch.setattr("synplan.chem.reaction.routes.quality.scorer.tqdm", progress)
    scorer = ForwardRouteScorer()
    assert len(scorer.assess([linear([1.0]), linear([0.0])], silent=silent)) == 2
    assert bars[0].total == 2
    if not silent:
        assert bars[0].n == 2
    assert bool(stream.getvalue()) is not silent
    assert scorer.assess([], silent=silent) == []


def test_collection_preserves_metadata_order_and_chemistry():
    first, second = linear([0.25]).to_json(), linear([0.0]).to_json()
    first["custom"] = {"keep": [1, 2]}
    collection = {
        "target-z": {"route-b": first, "route-a": second},
        "target-a": [deepcopy(first)],
    }
    previous = deepcopy(collection)
    result = score_routes.score_collection(collection, ForwardRouteScorer())
    assert collection == previous
    assert list(result) == list(collection)
    assert list(result["target-z"]) == ["route-b", "route-a"]
    for _, tree in score_routes.collection_routes(result):
        tree.pop("assessment")
    assert result == collection


@pytest.mark.parametrize("combine", [False, True])
def test_script_gzip_and_html_without_loading_model(tmp_path, monkeypatch, combine):
    from synplan.ml.inference.chemformer import OnnxChemformer

    def unexpected(*args, **kwargs):
        raise AssertionError("route scoring must not initialize the model")

    monkeypatch.setattr(OnnxChemformer, "__init__", unexpected)
    source, output, html = (
        tmp_path / "input.json.gz",
        tmp_path / "out.json.gz",
        tmp_path / "routes.html",
    )
    collection = {
        "target": {"custom-route": linear([0.25, 1.0], assumed=(1,)).to_json()}
    }
    write_json_atomic(source, collection)
    monkeypatch.setattr(
        "sys.argv",
        [
            "score_routes",
            "--input",
            str(source),
            "--output",
            str(output),
            "--html",
            str(html),
            "--quiet",
            *(["--combine-search"] if combine else []),
        ],
    )
    score_routes.main()
    result = read_json(output)["target"]["custom-route"]["assessment"]
    assert ("combined_score" in result) is combine
    assert read_json(source) == collection
    page = html.read_text(encoding="utf-8")
    assert ("<th>Combined score</th>" in page) is combine
    assert "custom-route" in page
    assert "Step scores · 1 → n" in page
    assert "Assumed" not in page
    assert '<span class="forward-match" title="—">1.000</span>' in page


def test_report_filters_step_tables_and_escaping():
    routes = ForwardRouteScorer().assess(
        [linear([1.0]), linear([0.0, 1.0], assumed=(1,))]
    )
    page = routes_report_html(
        routes, None, route_labels=["<script>bad</script>", "route-2"]
    )
    assert "<script>bad</script>" not in page
    assert "&lt;script&gt;bad&lt;/script&gt;" in page
    assert "No zero-score steps · 1" in page
    assert "At least one zero-score step · 1" in page
    assert page.count('class="route card"') == 2
    assert page.count("data-step-color=") == 3
    assert page.count('data-step-color="#267154"') == 2
    assert 'data-step-color="#bd3f3f"' in page
    assert "forward-assumed" not in page
    assert "Assumed" not in page
    assert "<th>Combined score</th>" not in page
    assert "No top-10 match" in page
    assert "Policy log likelihood" in page
    assert "reason" not in json.dumps(routes[0].assessment)
