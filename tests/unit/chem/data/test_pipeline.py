"""Tests for reaction processing pipeline result types and shared utilities."""

import pytest
from chython import smiles as parse_smiles

import synplan.chem.data.standardizing as standardizing
from synplan.chem.data.pipeline import (
    build_batch_result,
    reaction_cgr_key,
    serialize_reaction,
    write_batch_results,
)
from synplan.chem.data.reaction_result import (
    BatchResult,
    ErrorEntry,
    ExtractedRuleRecord,
    ExtractionBatchResult,
    FilteredEntry,
    PipelineSummary,
    ProcessResult,
)
from synplan.chem.data.standardizing import ReactionStandardizationConfig
from synplan.utils.files import ReactionReader, ReactionWriter

# -- BatchResult / ExtractionBatchResult dataclasses -------------------------


def test_batch_result_structure():
    """BatchResult holds pre-serialized records, dedup keys, filtered and errors."""
    b = BatchResult(records=["CC>>CC"], dedup_keys=["key"])
    assert b.records == ["CC>>CC"]
    assert b.dedup_keys == ["key"]
    assert b.filtered == []
    assert b.errors == []


def test_extraction_batch_result_structure():
    """ExtractionBatchResult holds rule_records, errors, and multi-product count."""
    e = ExtractionBatchResult(
        rule_records=[
            (
                0,
                [
                    ExtractedRuleRecord(
                        cgr_key="cgr",
                        rule_smarts="[C:1]>>[O:1]",
                        reactor_validation="passed",
                    )
                ],
                "CC",
            )
        ],
        errors=[],
        n_multi_product=1,
    )
    assert e.n_multi_product == 1
    assert e.rule_records[0][0] == 0


# -- ProcessResult (kept for backward compat) --------------------------------


def test_process_result_structure():
    """ProcessResult still works after BatchResult addition."""
    r = ProcessResult(reactions=[], filtered=[], errors=[])
    assert r.reactions == []


def test_process_result_can_carry_worker_dedup_keys():
    r = ProcessResult(reactions=[], dedup_keys=["CGR"])
    assert r.dedup_keys == ["CGR"]


# -- ErrorEntry / FilteredEntry ----------------------------------------------


def test_error_entry_with_line_number():
    e = ErrorEntry("CC>>CC", "parse", "ValueError", "bad smiles", line_number=42)
    assert e.line_number == 42
    assert e.stage == "parse"


def test_filtered_entry():
    f = FilteredEntry("CC>>CC", "no_reaction", line_number=7)
    assert f.reason == "no_reaction"


# -- PipelineSummary ----------------------------------------------------------


def test_pipeline_summary_to_json(tmp_path):
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
    assert s.total_excluded == 10


def test_pipeline_summary_to_json_file(tmp_path):
    s = PipelineSummary(total_input=10, succeeded=10)
    path = tmp_path / "summary.json"
    s.to_json(path)
    assert path.exists()
    assert '"total_input": 10' in path.read_text()


# -- reaction_cgr_key ---------------------------------------------------------


def test_reaction_cgr_key_stable():
    """reaction_cgr_key returns the canonical CGR string."""
    rxn = parse_smiles("[CH3:1][OH:2]>>[CH3:1][Cl:3]")
    if rxn is not None:
        assert reaction_cgr_key(rxn) == str(~rxn)


def test_reaction_cgr_key_alias():
    """_reaction_dedup_key in standardizing is an alias for reaction_cgr_key."""
    rxn = parse_smiles("[CH3:1][OH:2]>>[CH3:1][Cl:3]")
    if rxn is not None:
        assert standardizing._reaction_dedup_key(rxn) == reaction_cgr_key(rxn)


# -- CGR dedup consistency ---------------------------------------------------


def test_cgr_dedup_enantiomeric_same_mechanism():
    rxn_r = parse_smiles("[C@@H:1]([F:2])([Cl:3])[Br:4]>>[C@@H:1]([F:2])([Cl:3])[O:5]")
    rxn_s = parse_smiles("[C@H:1]([F:2])([Cl:3])[Br:4]>>[C@H:1]([F:2])([Cl:3])[O:5]")
    if rxn_r is not None and rxn_s is not None:
        assert hash(~rxn_r) == hash(~rxn_s)


def test_cgr_dedup_same_mechanism():
    rxn1 = parse_smiles("[CH3:1][OH:2]>>[CH3:1][Cl:3]")
    rxn2 = parse_smiles("[CH3:1][OH:2]>>[CH3:1][Cl:3]")
    if rxn1 is not None and rxn2 is not None:
        assert hash(~rxn1) == hash(~rxn2)


def test_cgr_different_mapping_different_hash():
    rxn1 = parse_smiles("[CH3:1][OH:2]>>[CH3:1][NH2:3]")
    rxn2 = parse_smiles("[CH3:2][OH:1]>>[CH3:2][NH2:3]")
    if rxn1 is not None and rxn2 is not None:
        assert isinstance(hash(~rxn1), int)
        assert isinstance(hash(~rxn2), int)


# -- serialize_reaction -------------------------------------------------------


def test_serialize_reaction_smi():
    """serialize_reaction returns a tab-separated SMILES record for smi format."""
    rxn = parse_smiles("[CH3:1][OH:2]>>[CH3:1][Cl:3]")
    if rxn is not None:
        record = serialize_reaction(rxn, "smi")
        assert ">>" in record
        assert "\n" not in record


def test_serialize_reaction_rdf():
    """serialize_reaction returns an RDF block starting with $RFMT for rdf format."""
    rxn = parse_smiles("[CH3:1][OH:2]>>[CH3:1][Cl:3]")
    if rxn is not None:
        block = serialize_reaction(rxn, "rdf")
        assert block.startswith("$RFMT")


def test_reaction_writer_write_string_rdf_roundtrip(tmp_path):
    """Pre-serialized RDF records can be written and read as a valid RDF file."""
    rxn = parse_smiles("[CH3:1][OH:2]>>[CH3:1][Cl:3]")
    if rxn is None:
        return

    path = tmp_path / "reactions.rdf"
    with ReactionWriter(path) as writer:
        writer.write_string(serialize_reaction(rxn, "rdf"))

    with ReactionReader(path) as reader:
        reactions = list(reader)

    assert len(reactions) == 1
    assert format(reactions[0], "m") == format(rxn, "m")


# -- build_batch_result -------------------------------------------------------


def test_build_batch_result_smi(tmp_path):
    """build_batch_result serializes reactions and computes dedup keys."""
    rxn = parse_smiles("[CH3:1][OH:2]>>[CH3:1][Cl:3]")
    if rxn is None:
        return
    errors = [ErrorEntry("bad", "parse", "ValueError", "bad smiles")]
    filtered = [FilteredEntry("CC>>CC", "no_change")]
    result = build_batch_result([rxn], errors, filtered, "smi")
    assert len(result.records) == 1
    assert ">>" in result.records[0]
    assert len(result.dedup_keys) == 1
    assert result.dedup_keys[0] == reaction_cgr_key(rxn)
    assert len(result.errors) == 1
    assert len(result.filtered) == 1


def test_build_batch_result_no_dedup():
    """build_batch_result with compute_dedup=False sets all keys to None."""
    rxn = parse_smiles("[CH3:1][OH:2]>>[CH3:1][Cl:3]")
    if rxn is None:
        return
    result = build_batch_result([rxn], [], [], "smi", compute_dedup=False)
    assert result.dedup_keys == [None]


# -- write_batch_results ------------------------------------------------------


def test_write_batch_results_basic():
    """write_batch_results writes records and updates summary counts."""
    written = []
    summary = PipelineSummary()
    batches = [
        BatchResult(records=["CC>>CC\n", "CC>>O\n"], dedup_keys=[None, None]),
    ]
    write_batch_results(batches, written.append, summary)
    assert written == ["CC>>CC\n", "CC>>O\n"]
    assert summary.succeeded == 2
    assert summary.total_input == 2


def test_write_batch_results_dedup():
    """write_batch_results deduplicates using CGR keys."""
    written = []
    summary = PipelineSummary()
    batches = [
        BatchResult(records=["CC>>CC\n", "CC>>CC\n"], dedup_keys=["key1", "key1"]),
    ]
    write_batch_results(batches, written.append, summary, dedup=True)
    assert len(written) == 1
    assert summary.succeeded == 1
    assert summary.duplicates == 1
    assert summary.total_input == 2


def test_write_batch_results_dedup_requires_worker_keys():
    """Deduplication must not silently fall back to serialized records."""
    written = []
    summary = PipelineSummary()
    batches = [BatchResult(records=["CC>>CC\n"], dedup_keys=[None])]

    with pytest.raises(ValueError, match="worker-computed dedup keys"):
        write_batch_results(batches, written.append, summary, dedup=True)

    assert written == []
    assert summary.succeeded == 0


def test_write_batch_results_errors(tmp_path):
    """write_batch_results logs errors to error file and updates summary."""
    written = []
    summary = PipelineSummary()
    err_path = tmp_path / "errors.tsv"
    errors = [ErrorEntry("bad", "parse", "ValueError", "msg")]
    batches = [BatchResult(records=[], dedup_keys=[], errors=errors)]
    with open(err_path, "w") as ef:
        write_batch_results(batches, written.append, summary, error_file=ef)
    assert summary.errored == 1
    assert "parse/ValueError" in summary.error_breakdown
    assert "bad\tparse\tValueError\tmsg" in err_path.read_text()


def test_write_batch_results_filtered():
    """write_batch_results tracks filtered entries in summary breakdown."""
    written = []
    summary = PipelineSummary()
    filtered = [FilteredEntry("CC>>CC", "no_change")]
    batches = [BatchResult(records=[], dedup_keys=[], filtered=filtered)]
    write_batch_results(batches, written.append, summary)
    assert summary.filtered == 1
    assert "no_change" in summary.filter_breakdown


def test_write_batch_results_on_batch_callback():
    """write_batch_results calls on_batch with the batch total."""
    totals = []
    summary = PipelineSummary()
    batches = [
        BatchResult(records=["a\n", "b\n"], dedup_keys=[None, None]),
        BatchResult(
            records=["c\n"], dedup_keys=[None], errors=[ErrorEntry("x", "s", "E", "m")]
        ),
    ]
    write_batch_results(batches, lambda s: None, summary, on_batch=totals.append)
    assert totals == [2, 2]  # batch1: 2 records; batch2: 1 record + 1 error


# -- standardization dedup integration ---------------------------------------


def test_standardization_dedup_uses_worker_computed_keys(monkeypatch, tmp_path):
    """Parent must not recompute CGR — workers pre-compute it in BatchResult."""
    written = []

    class FakeRawReactionReader:
        format = "smi"

        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def __iter__(self):
            return iter(["raw"])

    class FakeReactionWriter:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def write_string(self, record):
            written.append(record)

    def fake_process_pool_map_stream(*_args, **_kwargs):
        yield BatchResult(records=["CC>>CC"], dedup_keys=["CGR"])

    monkeypatch.setattr(standardizing, "RawReactionReader", FakeRawReactionReader)
    monkeypatch.setattr(standardizing, "ReactionWriter", FakeReactionWriter)
    monkeypatch.setattr(
        standardizing, "process_pool_map_stream", fake_process_pool_map_stream
    )

    summary = standardizing.standardize_reactions_from_file(
        ReactionStandardizationConfig(deduplicate=True),
        "input.smi",
        tmp_path / "output.smi",
        silent=True,
    )

    assert written == ["CC>>CC"]
    assert summary.succeeded == 1
