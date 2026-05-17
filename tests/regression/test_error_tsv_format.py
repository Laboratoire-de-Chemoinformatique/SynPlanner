"""Invariants for the pipeline-wide error TSV format.

Every pipeline stage writes "failed reactions" to a tab-separated error file.
The reference format (set by ``standardize_reactions_from_file``) is::

    # original_smiles\tsource_info\tstage\terror_type\terror_message

Five columns, one header line beginning with ``#``. The pipeline stays
introspectable only if every stage agrees on this format. These tests assert
the agreement; they do **not** care which exact reaction failed or why.

A 4-column extraction header or a 3-column mapping error file both break here.
Embedded tabs/newlines in any field would also push fields out of alignment
and trip the column-count check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synplan.chem.data.filtering import filter_reactions_from_file
from synplan.chem.data.standardizing import standardize_reactions_from_file
from synplan.chem.reaction_rules.extraction import extract_rules_from_reactions

from .conftest import ERROR_TSV_COLUMNS, ERROR_TSV_HEADER, parse_error_tsv


def _run_standardize(std_config, input_path: Path, out_dir: Path) -> Path:
    err_path = out_dir / "std.errors.tsv"
    out_path = out_dir / "std.smi"
    standardize_reactions_from_file(
        config=std_config,
        input_reaction_data_path=str(input_path),
        standardized_reaction_data_path=str(out_path),
        num_cpus=1,
        ignore_errors=True,
        error_file_path=str(err_path),
    )
    return err_path


def _run_filter(filt_config, input_path: Path, out_dir: Path) -> Path:
    err_path = out_dir / "filt.errors.tsv"
    out_path = out_dir / "filt.smi"
    filter_reactions_from_file(
        config=filt_config,
        input_reaction_data_path=str(input_path),
        filtered_reaction_data_path=str(out_path),
        num_cpus=1,
        ignore_errors=True,
        error_file_path=str(err_path),
    )
    return err_path


def _run_extract(rule_cfg_factory, input_path: Path, out_dir: Path) -> Path:
    err_path = out_dir / "rules.errors.tsv"
    rules_path = out_dir / "rules.tsv"
    extract_rules_from_reactions(
        config=rule_cfg_factory(),
        reaction_data_path=str(input_path),
        reaction_rules_path=str(rules_path),
        num_cpus=1,
        batch_size=4,
        ignore_errors=True,
        error_file_path=str(err_path),
    )
    return err_path


STAGE_RUNNERS = {
    "standardize": ("std_config", _run_standardize),
    "filter": ("filt_config", _run_filter),
    "extract": ("rule_cfg_factory", _run_extract),
}


@pytest.fixture
def stage_error_tsv(request, tmp_path, smi_with_one_unparseable):
    """Run a pipeline stage and return its error-TSV path.

    ``request.param`` is the stage name in ``STAGE_RUNNERS``.
    """
    fixture_name, runner = STAGE_RUNNERS[request.param]
    cfg = request.getfixturevalue(fixture_name)
    return runner(cfg, smi_with_one_unparseable, tmp_path)


@pytest.mark.parametrize(
    "stage_error_tsv", list(STAGE_RUNNERS), indirect=True, ids=list(STAGE_RUNNERS)
)
def test_error_tsv_header_matches_reference(stage_error_tsv: Path):
    """Header line is the 5-column reference, byte-for-byte."""
    header, _ = parse_error_tsv(stage_error_tsv)
    assert header == ERROR_TSV_HEADER, (
        f"{stage_error_tsv.name} header diverges from the pipeline-wide "
        f"reference.\n  expected: {ERROR_TSV_HEADER!r}\n       got: {header!r}"
    )


@pytest.mark.parametrize(
    "stage_error_tsv", list(STAGE_RUNNERS), indirect=True, ids=list(STAGE_RUNNERS)
)
def test_error_tsv_column_count_invariant(stage_error_tsv: Path):
    """Every row has exactly N columns (no tab injection, no missing fields)."""
    _, rows = parse_error_tsv(stage_error_tsv)
    if not rows:
        pytest.skip(f"{stage_error_tsv.name} produced no error rows for this input")
    bad = [(i, len(row)) for i, row in enumerate(rows) if len(row) != ERROR_TSV_COLUMNS]
    assert not bad, (
        f"{stage_error_tsv.name}: {len(bad)} row(s) do not have "
        f"{ERROR_TSV_COLUMNS} tab-separated fields. First offenders: {bad[:5]}. "
        "Typical causes: embedded tabs in original_smiles or error_message, "
        "header column-count mismatch, multi-line RDF blocks written verbatim."
    )


@pytest.mark.parametrize(
    "stage_error_tsv", list(STAGE_RUNNERS), indirect=True, ids=list(STAGE_RUNNERS)
)
def test_error_tsv_has_no_blank_rows(stage_error_tsv: Path):
    """No blank lines between rows. A blank line indicates an unescaped newline
    inside some field has split one logical record into two."""
    text = stage_error_tsv.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 2:
        pytest.skip("error TSV has no data rows")
    body = lines[1:]
    blanks_in_middle = [i for i, line in enumerate(body[:-1]) if line == ""]
    assert not blanks_in_middle, (
        f"{stage_error_tsv.name} contains blank lines in the middle of its "
        "rows; likely a record body was split by an unescaped newline in "
        f"original_smiles or error_message. Indices: {blanks_in_middle[:5]}."
    )
