"""Tests for the public route-export contract in ``synplan.mcts.search``."""

import gzip
import json

from synplan import __version__
from synplan.mcts.search import (
    ROUTE_EXPORT_SCHEMA_VERSION,
    _canonical_target_key,
    export_routes_artifact,
)


def _sample_results() -> dict:
    """A synthetic envelope using the documented make_json node keys."""
    return {
        "CCO": [
            {
                "type": "mol",
                "smiles": "CCO",
                "in_stock": False,
                "children": [
                    {
                        "type": "reaction",
                        "smiles": "[CH3:1][CH2:2][Cl:3]>>[CH3:1][CH2:2][OH:4]",
                        "children": [
                            {"type": "mol", "smiles": "CCCl", "in_stock": True},
                        ],
                    }
                ],
            }
        ],
        "c1ccccc1": [],
    }


def test_results_gzip_roundtrip(tmp_path):
    results = _sample_results()
    out = export_routes_artifact(results, tmp_path)

    assert out == tmp_path / "results.json.gz"
    assert out.exists()

    # Confirm it is gzip (magic bytes) and round-trips byte-equal.
    with open(out, "rb") as fh:
        assert fh.read(2) == b"\x1f\x8b"
    with gzip.open(out, "rt", encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded == results
    # Unsolved targets are an empty list, node shape preserved exactly.
    assert loaded["c1ccccc1"] == []
    assert loaded["CCO"] == results["CCO"]


def test_manifest_contract(tmp_path):
    export_routes_artifact(_sample_results(), tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["directives"]["adapter"] == "synplanner"
    assert manifest["directives"]["raw_results_filename"] == "results.json.gz"
    assert manifest["schema_version"] == ROUTE_EXPORT_SCHEMA_VERSION
    assert manifest["synplan_version"] == __version__


def test_manifest_filename_tracks_results_filename(tmp_path):
    export_routes_artifact(_sample_results(), tmp_path, filename="routes.json.gz")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["directives"]["raw_results_filename"] == "routes.json.gz"


def test_canonical_target_key_uses_chython():
    from synplan.chem.utils import mol_from_smiles

    assert _canonical_target_key("OCC") == str(mol_from_smiles("CCO"))
    assert _canonical_target_key("OCC") == _canonical_target_key("CCO")


def test_canonical_target_key_preserves_stereo():
    assert _canonical_target_key("C/C=C/C") != _canonical_target_key("C/C=C\\C")
    assert _canonical_target_key("N[C@@H](C)C(=O)O") != _canonical_target_key(
        "N[C@H](C)C(=O)O"
    )
    assert _canonical_target_key("CC=[C@]=CC") != _canonical_target_key("CC=[C@@]=CC")


def test_canonical_target_key_unparseable_falls_back_to_raw():
    # Invalid input remains traceable without a silently repaired export key.
    assert _canonical_target_key("not a smiles") == "not a smiles"
