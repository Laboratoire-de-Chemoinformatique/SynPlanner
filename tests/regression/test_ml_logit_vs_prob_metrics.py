"""ML-network metric correctness invariant.

For multilabel and binary-classification heads, torchmetrics' ``recall``,
``specificity``, ``f1_score``, ``binary_recall``, and ``binary_specificity``
apply a fixed 0.5 threshold to the *first* tensor argument. If the caller
passes raw logits (centred around 0, range unbounded), the threshold is
almost never crossed and the reported metrics are systematically wrong —
the *loss* trains correctly but the user sees near-zero recall throughout.

The contract this test enforces is structural: for every metric call in
``policy.py`` / ``value.py`` that uses one of the threshold-applying
metrics with ``task="multilabel"`` or any ``binary_*`` variant, the first
positional argument must be a ``sigmoid(...)`` expression (or a name
assigned from one).

This is a static AST check; it does not run a training loop and does not
care about specific architecture choices. It survives any rearrangement of
the network that keeps the same metric APIs.

Caveat: a future implementation could compute metrics on a logits-domain
threshold deliberately (e.g. ``threshold=0.0`` argument). If that's the
intent, the test can be adjusted; today the chython/torchmetrics defaults
make the bug live.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

THRESHOLD_METRICS = {
    "recall",
    "specificity",
    "f1_score",
    "binary_recall",
    "binary_specificity",
    "binary_f1_score",
}


def _is_sigmoid_call(node: ast.expr) -> bool:
    """True if the node is a call like sigmoid(...) or torch.sigmoid(...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = (
        func.id
        if isinstance(func, ast.Name)
        else func.attr
        if isinstance(func, ast.Attribute)
        else None
    )
    return name == "sigmoid"


def _is_problike_name(name: str) -> bool:
    """A name that suggests probability-domain (post-sigmoid) values."""
    lower = name.lower()
    return any(tag in lower for tag in ("prob", "_p", "sigmoid"))


def _resolve_assignment_origin(
    arg: ast.expr, assignments: dict[str, ast.expr]
) -> ast.expr:
    """Walk back through simple name→expr assignments until we hit a non-Name."""
    seen = set()
    while isinstance(arg, ast.Name) and arg.id in assignments and arg.id not in seen:
        seen.add(arg.id)
        arg = assignments[arg.id]
    return arg


def _check_module(path: Path) -> list[tuple[int, str]]:
    """Return a list of (lineno, metric_call_source) for offending calls."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    offenders: list[tuple[int, str]] = []
    src_lines = src.splitlines()

    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        # Build a map: name → most-recent Assign value seen in this function.
        assignments: dict[str, ast.expr] = {}
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        assignments[tgt.id] = node.value
            elif isinstance(node, ast.Call):
                callee = node.func
                name = (
                    callee.id
                    if isinstance(callee, ast.Name)
                    else callee.attr
                    if isinstance(callee, ast.Attribute)
                    else None
                )
                if name not in THRESHOLD_METRICS:
                    continue
                # Inspect task kwarg — multiclass is fine (argmax-based).
                task_kw = None
                for kw in node.keywords:
                    if kw.arg == "task" and isinstance(kw.value, ast.Constant):
                        task_kw = kw.value.value
                if task_kw == "multiclass":
                    continue
                # First positional arg is the predictions tensor.
                if not node.args:
                    continue
                first_arg = node.args[0]
                resolved = _resolve_assignment_origin(first_arg, assignments)
                ok = (
                    _is_sigmoid_call(resolved)
                    or (
                        isinstance(resolved, ast.Name)
                        and _is_problike_name(resolved.id)
                    )
                    or (
                        isinstance(first_arg, ast.Name)
                        and _is_problike_name(first_arg.id)
                    )
                )
                if not ok:
                    src_line = src_lines[node.lineno - 1].strip()
                    offenders.append((node.lineno, src_line))
    return offenders


@pytest.mark.parametrize(
    "module_relpath",
    [
        "synplan/ml/networks/policy.py",
        "synplan/ml/networks/value.py",
    ],
)
def test_threshold_metrics_called_on_probabilities(module_relpath: str):
    """Every threshold-using metric call must pass a sigmoid'd argument."""
    p = REPO_ROOT / module_relpath
    assert p.exists(), f"missing {p}"
    offenders = _check_module(p)
    assert not offenders, (
        f"{module_relpath}: metric call(s) take raw logits as their first "
        "argument. torchmetrics applies a 0.5 threshold on that tensor; "
        "for logits (centred around 0) the threshold is almost never "
        "crossed, so the logged recall/specificity/f1 are systematically "
        "wrong throughout training. Offending lines:\n"
        + "\n".join(f"  {ln}: {src}" for ln, src in offenders)
    )
