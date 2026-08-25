import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


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

    # Each snippet still gets its own fresh interpreter; they just no longer
    # queue behind each other's torch import, which cost ~2s apiece serially.
    with ThreadPoolExecutor(max_workers=len(snippets)) as pool:
        for snippet, future in [
            (s, pool.submit(_run_fresh_process, s)) for s in snippets
        ]:
            try:
                future.result()
            except AssertionError as exc:
                raise AssertionError(f"{snippet!r}\n{exc}") from None


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
import synplan.chem.synthon

unexpected = [
    name
    for name in (
        'torch',
        'matplotlib',
        'rdkit',
        'synplan.mcts.tree',
        'synplan.chem.synthon.classify',
        'synplan.chem.synthon.coverage',
        'synplan.chem.synthon.synthonise',
        'synplan.chem.synthon.fragment',
    )
    if name in sys.modules
]
assert unexpected == [], unexpected
assert synplan.chem.synthon.SynthonConfig.__name__ == 'SynthonConfig'
"""
    _run_fresh_process(code)


def test_synthon_never_imports_torch():
    """`select_device` is the only torch consumer in `utils.parallel`, and synthon never calls it.

    Before the import moved into that function, `SynthonConfig` pulled the whole framework for the
    sake of `default_num_workers`, which is `min(os.cpu_count() or 4, cap)`.
    """
    code = """
import sys
import synplan.chem.synthon.config
import synplan.chem.synthon.coverage
import synplan.chem.synthon.fragment
import synplan.interfaces.synthon_commands

assert 'torch' not in sys.modules, 'synthon pulled torch'

from synplan.utils.parallel import default_num_workers
assert default_num_workers() >= 1
assert 'torch' not in sys.modules, 'default_num_workers pulled torch'

from synplan.utils.parallel import select_device
assert select_device() is not None
assert 'torch' in sys.modules, 'select_device must load torch when actually called'
"""
    _run_fresh_process(code)
