"""Tests for the public route-export contract in ``synplan.mcts.search``."""

import gzip
import json

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
    export_routes_artifact(
        _sample_results(), tmp_path, synplan_version="9.9.9"
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["directives"]["adapter"] == "synplanner"
    assert manifest["directives"]["raw_results_filename"] == "results.json.gz"
    assert manifest["schema_version"] == ROUTE_EXPORT_SCHEMA_VERSION
    assert manifest["synplan_version"] == "9.9.9"


def test_custom_filename_and_no_manifest(tmp_path):
    out = export_routes_artifact(
        _sample_results(),
        tmp_path,
        filename="routes.json.gz",
        write_manifest=False,
    )
    assert out == tmp_path / "routes.json.gz"
    assert out.exists()
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_filename_tracks_results_filename(tmp_path):
    export_routes_artifact(
        _sample_results(), tmp_path, filename="routes.json.gz"
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["directives"]["raw_results_filename"] == "routes.json.gz"


def test_canonical_target_key_matches_retrocast():
    # Non-canonical input is normalized to the exact string retrocast stores in
    # Target.smiles (canonicalize_smiles defaults: isomericSmiles=True, mapping
    # preserved). "OCC" -> "CCO".
    assert _canonical_target_key("OCC") == "CCO"
    # Already canonical input is a no-op.
    assert _canonical_target_key("CCO") == "CCO"


def test_canonical_target_key_preserves_stereo():
    # retrocast keeps stereochemistry (isomericSmiles=True), so it must survive.
    assert _canonical_target_key("C/C=C/C") == "C/C=C/C"
    assert _canonical_target_key("N[C@@H](C)C(=O)O") == "C[C@H](N)C(=O)O"


def test_canonical_target_key_unparseable_falls_back_to_raw():
    # RDKit cannot parse this; helper keys by the raw string instead of crashing.
    assert _canonical_target_key("not a smiles") == "not a smiles"
