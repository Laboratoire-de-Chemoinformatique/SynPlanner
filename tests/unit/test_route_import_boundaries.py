import subprocess
import sys


def _run_fresh_process(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_mcts_imports_do_not_depend_on_route_import_order():
    snippets = [
        "from synplan import Tree; assert Tree.__name__ == 'Tree'",
        "from synplan.mcts import Tree; assert Tree.__name__ == 'Tree'",
        "import synplan.mcts.tree",
        "import synplan.mcts.search",
        "from synplan.chem.reaction.routes.quality.scorer import RouteScorer",
        "import synplan.chem.reaction.routes",
    ]

    for snippet in snippets:
        _run_fresh_process(snippet)


def test_route_quality_imports_stay_lightweight():
    for import_line in (
        "from synplan.chem.reaction.routes.quality.scorer import RouteScorer",
        "from synplan.route_quality.scorer import RouteScorer",
    ):
        code = f"""
import sys
{import_line}

assert RouteScorer.__name__ == 'RouteScorer'
unexpected = [
    name
    for name in (
        'matplotlib',
        'synplan.utils.visualisation',
        'synplan.chem.reaction.routes.clustering.core',
        'synplan.chem.reaction.routes.clustering.subclustering',
        'synplan.mcts.tree',
    )
    if name in sys.modules
]
assert unexpected == [], unexpected
"""
        _run_fresh_process(code)


def test_route_package_roots_stay_lightweight():
    for snippet in (
        "import synplan.chem.reaction.routes",
        "import synplan.route_quality",
    ):
        code = f"""
import sys
{snippet}

unexpected = [
    name
    for name in (
        'matplotlib',
        'synplan.utils.visualisation',
        'synplan.chem.reaction.routes.clustering.core',
        'synplan.chem.reaction.routes.clustering.subclustering',
        'synplan.mcts.tree',
    )
    if name in sys.modules
]
assert unexpected == [], unexpected
"""
        _run_fresh_process(code)


def test_synplan_routes_lazy_root_exports_still_work():
    code = """
from synplan.chem.reaction.routes import RouteScorer, compose_route_cgr

assert RouteScorer.__name__ == 'RouteScorer'
assert compose_route_cgr.__name__ == 'compose_route_cgr'
"""
    _run_fresh_process(code)
