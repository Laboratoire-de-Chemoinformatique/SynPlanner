import subprocess
import sys


def _run_fresh_process(code: str) -> str:
    wrapped_code = f"""
import os
import sys

exec({code!r})
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", wrapped_code],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
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
    code = """
import sys
from synplan.chem.reaction.routes.quality.scorer import RouteScorer

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


def test_route_package_root_stays_lightweight():
    code = """
import sys
import synplan.chem.reaction.routes

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


def test_route_package_lazy_root_exports_still_work():
    code = """
from synplan.chem.reaction.routes import RouteScorer, compose_route_cgr

assert RouteScorer.__name__ == 'RouteScorer'
assert compose_route_cgr.__name__ == 'compose_route_cgr'
"""
    _run_fresh_process(code)


def test_representation_import_does_not_load_route_visualisation():
    code = """
import sys
import synplan.chem.reaction.routes.representation

assert 'synplan.chem.reaction.routes.visualisation' not in sys.modules
"""
    _run_fresh_process(code)


def test_representation_depiction_is_canonical_and_visualisation_stays_compatible():
    code = """
from synplan.chem.reaction.routes.representation.depiction import depict_route_cgr
from synplan.chem.reaction.routes.visualisation import (
    depict_route_cgr as facade_depict_route_cgr,
)

assert callable(depict_route_cgr)
assert facade_depict_route_cgr is depict_route_cgr
"""
    _run_fresh_process(code)


def test_synthon_package_root_stays_lightweight():
    code = """
import sys
import synplan.enumeration.synthon

unexpected = [
    name
    for name in (
        'torch',
        'matplotlib',
        'rdkit',
        'synplan.mcts.tree',
        'synplan.enumeration.synthon.classify',
        'synplan.enumeration.synthon.synthonise',
        'synplan.enumeration.synthon.fragment',
    )
    if name in sys.modules
]
assert unexpected == [], unexpected
assert synplan.enumeration.synthon.SynthonConfig.__name__ == 'SynthonConfig'
"""
    _run_fresh_process(code)
