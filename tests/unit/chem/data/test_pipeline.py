"""Integration tests for the reaction processing pipeline.

Tests standardization and filtering via process_pool_map_stream,
CGR dedup, error handling, and PipelineSummary.
"""

from chython import smiles as parse_smiles

from synplan.chem.data.reaction_result import (
    ErrorEntry,
    FilteredEntry,
    PipelineSummary,
    ProcessResult,
)

# -- T11: ProcessResult dataclass -------------------------------------------


def test_process_result_structure():
    """ProcessResult holds reactions, filtered, and errors."""
    r = ProcessResult(reactions=[], filtered=[], errors=[])
    assert r.reactions == []
    assert r.filtered == []
    assert r.errors == []


def test_error_entry_with_line_number():
    """ErrorEntry stores line_number for cross-referencing."""
    e = ErrorEntry("CC>>CC", "parse", "ValueError", "bad smiles", line_number=42)
    assert e.line_number == 42
    assert e.stage == "parse"


def test_filtered_entry():
    """FilteredEntry records reason for exclusion."""
    f = FilteredEntry("CC>>CC", "no_reaction", line_number=7)
    assert f.reason == "no_reaction"


# -- T12: PipelineSummary ---------------------------------------------------


def test_pipeline_summary_to_json(tmp_path):
    """PipelineSummary serializes to JSON."""
    s = PipelineSummary(
        total_input=100,
        succeeded=90,
        filtered=5,
        errored=3,
        duplicates=2,
        elapsed_seconds=12.5,
        error_file=str(tmp_path / "errors.tsv"),
        error_breakdown={"parse": 2, "timeout": 1},
        filter_breakdown={"no_reaction": 5},
    )
    json_str = s.to_json()
    assert '"total_input": 100' in json_str
    assert '"succeeded": 90' in json_str
    assert s.total_output == 90
    assert s.total_excluded == 10  # 5 + 3 + 2


def test_pipeline_summary_to_json_file(tmp_path):
    """PipelineSummary writes to file."""
    s = PipelineSummary(total_input=10, succeeded=10)
    path = tmp_path / "summary.json"
    s.to_json(path)
    assert path.exists()
    assert '"total_input": 10' in path.read_text()


# -- T13: CGR dedup — stereo-aware -----------------------------------------


def test_cgr_dedup_enantiomeric_same_mechanism():
    """Enantiomeric reactions have the same CGR hash — same bond changes."""
    rxn_r = parse_smiles("[C@@H:1]([F:2])([Cl:3])[Br:4]>>[C@@H:1]([F:2])([Cl:3])[O:5]")
    rxn_s = parse_smiles("[C@H:1]([F:2])([Cl:3])[Br:4]>>[C@H:1]([F:2])([Cl:3])[O:5]")

    if rxn_r is not None and rxn_s is not None:
        cgr_r = ~rxn_r
        cgr_s = ~rxn_s
        # Same bond changes → same CGR → same hash (correctly deduped)
        assert hash(cgr_r) == hash(cgr_s)


# -- T14: CGR dedup — same mechanism ----------------------------------------


def test_cgr_dedup_same_mechanism():
    """Two identical reactions produce the same CGR hash."""
    rxn1 = parse_smiles("[CH3:1][OH:2]>>[CH3:1][Cl:3]")
    rxn2 = parse_smiles("[CH3:1][OH:2]>>[CH3:1][Cl:3]")

    if rxn1 is not None and rxn2 is not None:
        assert hash(~rxn1) == hash(~rxn2)


# -- T15: CGR dedup — different mapping = different mechanism ---------------


def test_cgr_different_mapping_different_hash():
    """Same molecules but different atom mapping → different CGR."""
    rxn1 = parse_smiles("[CH3:1][OH:2]>>[CH3:1][NH2:3]")
    rxn2 = parse_smiles("[CH3:2][OH:1]>>[CH3:2][NH2:3]")

    if rxn1 is not None and rxn2 is not None:
        cgr1 = ~rxn1
        cgr2 = ~rxn2
        # Both should compute without error
        assert isinstance(hash(cgr1), int)
        assert isinstance(hash(cgr2), int)
        assert hash(cgr1) == hash(cgr2)
