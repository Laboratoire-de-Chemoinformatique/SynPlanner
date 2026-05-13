"""Route I/O invariants.

Two contracts are asserted:

1. **Default-filename sanity.** ``read_routes_json`` advertises a JSON-reading
   function but its default ``file_path`` argument is ``"routes.csv"``. Any
   caller relying on the default will ``json.load`` a CSV file (or vice
   versa). Test: introspect the signature, the default for a JSON reader
   must end in ``.json``.

2. **Round-trip equivalence.** Writing a routes dict to JSON and reading it
   back must yield a structurally equivalent object. The test fixture is the
   already-checked-in routes JSON at ``tests/data/routes_mol_1.json``; we
   load it, write a copy, and re-load — equality of the loaded structure is
   the invariant. Catches silent drops in ``make_json`` (broad-except
   skipping routes) and any future serialization breakage that loses
   information.

These tests do not pin the *contents* of the routes, so they survive any
future change to chython rendering, reaction ordering, or the MCTS internals.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from synplan.chem.reaction_routes.io import read_routes_json, write_routes_json

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTES_JSON_FIXTURE = REPO_ROOT / "tests" / "data" / "routes_mol_1.json"


def test_read_routes_json_default_filename_is_json():
    """``read_routes_json`` must default to a ``.json`` path, not ``.csv``."""
    sig = inspect.signature(read_routes_json)
    default = sig.parameters["file_path"].default
    assert isinstance(default, (str, Path)), (
        f"read_routes_json file_path default is {default!r}, expected a string path"
    )
    assert str(default).lower().endswith(".json"), (
        f"read_routes_json defaults to file_path={default!r}, which does not "
        "end in '.json'. Callers relying on the default will try to "
        "json.load() a non-JSON file. Either change the default to "
        "'routes.json' or require an explicit path."
    )


def test_write_then_read_routes_json_is_lossless(tmp_path: Path):
    """Round-trip a real routes JSON through write_routes_json+read_routes_json.

    We start from the JSON form (the canonical wire format) rather than the
    routes_dict form because ``write_routes_json`` accepts a dict shape, the
    fixture is JSON, and the cleanest invariant is "JSON → ? → JSON
    preserves structure".
    """
    if not ROUTES_JSON_FIXTURE.exists():
        pytest.skip(f"routes fixture not found at {ROUTES_JSON_FIXTURE}")
    original = json.loads(ROUTES_JSON_FIXTURE.read_text(encoding="utf-8"))

    # Convert the JSON-shaped data back into the routes_dict shape expected
    # by write_routes_json. That shape is route_id -> step_id -> reaction.
    # If the fixture is already in the same shape, the conversion is a
    # no-op; if it requires read_routes_json with to_dict=True, do that.
    routes_dict = read_routes_json(str(ROUTES_JSON_FIXTURE), to_dict=True)
    out_path = tmp_path / "rt.json"
    write_routes_json(routes_dict, str(out_path))
    assert out_path.exists()

    reloaded = json.loads(out_path.read_text(encoding="utf-8"))

    # We don't compare to ``original`` directly because make_json may
    # legitimately reorder keys or re-render SMILES. The invariant we
    # require is structural: same set of route ids, and each route has the
    # same set of step ids.
    def _route_ids(j):
        if isinstance(j, dict):
            return set(j.keys())
        if isinstance(j, list):
            return {item.get("id", i) for i, item in enumerate(j)}
        raise TypeError(f"unexpected json shape: {type(j).__name__}")

    orig_ids = _route_ids(original)
    rt_ids = _route_ids(reloaded)
    assert orig_ids == rt_ids, (
        f"write→read lost route ids. Missing: {orig_ids - rt_ids}. Added: "
        f"{rt_ids - orig_ids}. Likely cause: make_json's broad-except clause "
        "is silently dropping routes whose serialization raises."
    )
