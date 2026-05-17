"""Route-quality scoring invariants.

The S(T) protection score has the form ``1 - penalty/N``. Today
``N = max(len(route), 1)``, i.e. the full route length, even when the
scanner silently skipped steps whose CGR could not be composed (e.g.
stereo-bearing reactions throwing inside ``~reaction``). Routes whose every
step fails CGR composition contribute 0 to ``step_worst`` and therefore
receive ``S(T) = 1.0``, a perfect score that disguises total scan failure.

Invariants asserted here:

Score correctness: a route the scanner cannot process must not receive the
same score as one it can. The scorer must surface this distinction, either
via an explicit ``n_processed`` counter, a typed return, or a deviation in
the score value.

Static guard: the scorer's denominator must not be plain ``len(route)``
without tracking the processed-step count. A clean fix introduces a counter
like ``n_processed = len(step_worst)`` or returns the processed count;
the static check looks for either pattern in the scorer.

Test 1 is the *behavioural* test (uses mocking to force scan failures).
Test 2 is the *static* fallback for when behavioural testing isn't viable.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORER_PATH = REPO_ROOT / "synplan" / "route_quality" / "protection" / "scorer.py"


def test_failed_scan_does_not_yield_perfect_score():
    """If the scanner reports no interactions because every step failed
    its CGR composition, S(T) must NOT be 1.0.

    Strategy: mock ``ProtectionScanner.scan_route`` to return
    ``(interactions=[], halogen_count=0)``, what the scanner emits when it
    silently skips every step. The scorer is then called with a multi-step
    route. The current implementation returns ``1.0 - 0/N = 1.0``, which is
    indistinguishable from a real "no problems found" run. A correct
    implementation either reflects the scan failure in the score or
    surfaces it through the return type.
    """
    try:
        from synplan.route_quality.protection.scorer import CompetingSitesScore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"CompetingSitesScore unavailable: {e}")

    # Mock scanner with a controlled empty scan return. Use MagicMock so we
    # don't have to construct the real RouteScanner (which requires fg
    # detector + incompatibility matrix); the only API the scorer touches
    # is ``scan_route(...)`` and ``last_processed_steps``.
    scanner = MagicMock()
    scanner.scan_route.return_value = ([], 0)
    scanner.last_processed_steps = set()  # zero steps scanned successfully

    scorer = CompetingSitesScore(scanner)
    fake_route = {i: object() for i in range(5)}  # 5 steps, opaque
    result = scorer.score_route(fake_route)

    # The current return shape is (score, interactions); we accept either
    # that or any extended shape, but the score must be < 1.0 for an
    # empty-scan input that *was* a non-empty route; otherwise the
    # contract collapses real scan failures into apparent perfection.
    score = result[0] if isinstance(result, tuple) else result
    assert score < 1.0, (
        f"ProtectionScorer.score returned S(T) = {score} for a 5-step "
        "route where the scanner reported zero interactions. This is "
        "indistinguishable from a route where every step's CGR "
        "composition failed silently (the scanner skips and logs a "
        "warning, but contributes nothing to step_worst). The denominator "
        "n_reactions = max(len(route), 1) does not account for skipped "
        "steps. Either track processed steps in the scanner's return and "
        "use that as the denominator, or surface the processed count in "
        "the score's return shape."
    )


def test_scorer_denominator_tracks_processed_steps():
    """Static fallback: the scorer's denominator must NOT be simply
    ``max(len(route), 1)`` without a processed-step counter nearby."""
    assert SCORER_PATH.exists(), f"missing {SCORER_PATH}"
    tree = ast.parse(SCORER_PATH.read_text(encoding="utf-8"))

    # Find every assignment of the form `n_reactions = max(len(...), 1)`
    # inside the scorer's score() method.
    offenders: list[int] = []

    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef) or func.name not in {
            "score",
            "score_route",
        }:
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            call = node.value
            # Look for max(len(<x>), 1) shape.
            callee = call.func
            name = (
                callee.id
                if isinstance(callee, ast.Name)
                else callee.attr
                if isinstance(callee, ast.Attribute)
                else None
            )
            if name != "max" or len(call.args) != 2:
                continue
            first = call.args[0]
            if not isinstance(first, ast.Call):
                continue
            inner = first.func
            inner_name = (
                inner.id
                if isinstance(inner, ast.Name)
                else inner.attr
                if isinstance(inner, ast.Attribute)
                else None
            )
            if inner_name != "len":
                continue
            if not first.args:
                continue
            # Check the arg of len(); bug pattern is `len(route)`. A fix
            # uses `len(step_worst)` or `len(processed_steps)` or similar.
            len_arg = first.args[0]
            if isinstance(len_arg, ast.Name) and len_arg.id == "route":
                offenders.append(node.lineno)

    assert not offenders, (
        f"scorer.py uses len(route) as the score denominator at line(s) "
        f"{offenders}. When the scanner silently skips steps whose CGR "
        "composition fails, those steps still count in the denominator, "
        "making total scan failure look like a perfect score (S(T) = 1.0). "
        "The denominator should be the number of *successfully processed* "
        "steps (e.g. len(step_worst) if step_worst tracks every "
        "scanned step, or a dedicated counter)."
    )
