"""Compact scientific reporting stays independent of forward validation."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from bs4 import BeautifulSoup

from synplan.chem.reaction.routes import RouteProvenance
from synplan.chem.reaction.routes.quality import (
    ForwardReactionAssessor,
    ForwardRouteScorer,
)
from synplan.chem.reaction.routes.thermochemistry_report import (
    display_status,
    lightweight_log,
)
from synplan.chem.reaction.score import AIMNetReactionAssessor
from synplan.utils.visualisation import routes_report_html, thermochemistry_report_html

from .test_aimnet import Calculator, route
from .test_persistence import ForwardModel


def annotated():
    source = replace(
        route("CC(O)F>>CC(O)F"),
        provenance=RouteProvenance(
            search_score=0.75, policy_log_likelihood=-9.87654321
        ),
    )
    result = AIMNetReactionAssessor(cache_path=None, calculator=Calculator()).assess(
        [source]
    )[0]
    result.steps[0].thermochemistry["molecules"]["private"] = {
        "coordinates_angstrom": ["DO_NOT_EMBED"]
    }
    return result


def test_compact_report_layout_independence_and_manifest():
    result = annotated()
    # Malformed unrelated model data must not be read by a thermo renderer.
    result = replace(
        result,
        assessment={**result.assessment, "forward_score": object()},
        steps=tuple(
            replace(s, forward_prediction={"unexpected": object()}) for s in result
        ),
    )
    page = thermochemistry_report_html(
        [result], None, manifest={"source": {"name": "<script>x</script>"}}
    )
    soup = BeautifulSoup(page, "html.parser")
    card = soup.select_one(".route")
    assert card.select_one(".draw").find_next_sibling()["class"] == ["thermo-scroll"]
    assert len(soup.select(".thermo-step")) == 1
    assert soup.select_one(".thermo-step")["data-status"] == "processed"
    assert "0.750" in page and "min-max" in page
    for forbidden in (
        "Forward route assessment",
        "Policy log likelihood",
        "-9.87654321",
        "DO_NOT_EMBED",
        "coordinates_angstrom",
        "Molecular results and calculation provenance",
    ):
        assert forbidden not in page
    dialog = soup.select_one("#thermo-manifest")
    assert not dialog.has_attr("open")
    assert len(soup.select("#open-manifest")) == 1
    assert "config" in dialog.get_text() and "config" not in card.get_text()
    assert "<script>x</script>" not in page
    assert "showModal()" in page and "dialog.addEventListener('close'" in page
    assert "Policy log likelihood" in thermochemistry_report_html(
        [result], None, show_policy_log_likelihood=True
    )


def test_ranges_exclude_failed_variants_and_log_is_small():
    result = annotated()
    annotation = result.steps[0].thermochemistry
    for i, v in enumerate(annotation["variants"]):
        v.update(delta_e=float(i), delta_h=float(i + 10), delta_g=float(i + 20))
    annotation["status"] = "partial"
    bad = annotation["variants"][-1]
    bad.update(
        status="failed",
        delta_e=99999,
        delta_h=99999,
        delta_g=99999,
        failures=[
            {
                "stage": "conformers",
                "reason": "no_valid_minimum",
                "detail": "<unsafe>",
                "smiles": "CC(O)F",
                "conformer_failures": [
                    {
                        "stage": "optimization",
                        "reason": "not_converged",
                        "coordinates_angstrom": [1],
                    }
                ]
                * 8,
            }
        ],
    )
    page = thermochemistry_report_html([result], None)
    soup = BeautifulSoup(page, "html.parser")
    row = soup.select_one(".thermo-step")
    assert row["data-status"] == "errored"
    assert "3 / 4" in row.get_text() and "99999" not in page
    assert len(soup.select(".variants tbody tr")) == 4
    log = json.loads(soup.select_one(".thermo-log pre").get_text())
    assert log["failures"][0]["conformer_failures"] == [
        {"stage": "optimization", "reason": "not_converged", "count": 8}
    ]
    assert "coordinates_angstrom" not in json.dumps(log)
    assert soup.select_one("[data-copy-log]") and soup.select_one("[data-download-log]")
    assert "<unsafe>" not in page


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            {"category": "filtered_out", "reason": "confidence_below_threshold"},
            "filtered_out",
        ),
        ({"reason": "unsupported_element"}, "filtered_out"),
        (
            {"reason": "balancing_rejected", "detail": "confidence below threshold"},
            "errored",
        ),
        ({"category": "future_category", "reason": "unknown"}, "errored"),
        ({"category": "errored", "reason": "not_conserved"}, "errored"),
    ],
)
def test_explicit_filter_and_legacy_error_categories(failure, expected):
    assert (
        display_status({"status": "skipped", "variants": [], "failure": failure})
        == expected
    )


def test_missing_and_aborted_steps_and_manifest_references():
    source = route()
    assert display_status(None) == "errored"
    log = lightweight_log(None, "<route>", 1)
    assert log["failures"][0]["reason"] == "missing_assessment"
    annotation = {
        "status": "failed",
        "variants": [],
        "failure": {
            "category": "errored",
            "stage": "run",
            "reason": "run_aborted",
            "detail": "failure",
        },
    }
    source = replace(
        source, steps=(replace(source.steps[0], thermochemistry=annotation),)
    )
    page = thermochemistry_report_html(
        [source], None, manifest={"run_status": "aborted"}
    )
    assert "Errored — run aborted" in page
    assert "run_aborted" in page


def test_forward_report_does_not_include_thermochemistry():
    result = annotated()
    before = deepcopy(result.to_json())
    page = routes_report_html([result], None)
    assert "Route thermochemistry" not in page and "Molecular results" not in page
    assert before == result.to_json()


@pytest.mark.parametrize("mode", ["thermo", "forward", "both", "mixed"])
def test_report_modes_and_escaping(tmp_path, mode):
    source = route()
    thermo = AIMNetReactionAssessor(cache_path=None, calculator=Calculator()).assess(
        [source]
    )[0]
    forward = ForwardRouteScorer().assess(
        ForwardReactionAssessor(ForwardModel(), tmp_path / "forward.json").assess(
            [source]
        )
    )[0]
    both = AIMNetReactionAssessor(cache_path=None, calculator=Calculator()).assess(
        [forward]
    )[0]
    selected = {
        "thermo": [thermo],
        "forward": [forward],
        "both": [both],
        "mixed": [thermo, forward, source],
    }[mode]
    for item in selected:
        if item.steps[0].thermochemistry:
            item.steps[0].thermochemistry["original_reaction"] = (
                '<script>alert("x")</script>'
            )
    renderer = routes_report_html if mode == "forward" else thermochemistry_report_html
    page = renderer(
        selected, None, route_labels=[f"route<{i}>" for i in range(len(selected))]
    )
    assert '<script>alert("x")</script>' not in page
    assert "route&lt;0&gt;" in page
    if mode != "forward":
        assert "Route thermochemistry" in page
        assert "&lt;script&gt;" in page
        assert "kcal/mol" in page
    assert ("Forward route assessment" in page) == (mode == "forward")
