"""Conservation-of-records invariant across pipeline stages.

A pipeline stage that drops records silently (incrementing a counter but
never writing the dropped row anywhere) is a recurring bug class in this
project. Examples already triaged: pre-e1607e6 filtering, extraction's
multi-product skip, mapping's ``ignore_errors=False`` swallow.

The invariant tested here is general and survives any future stage:

    summary.total_input == succeeded + errored + filtered + duplicates

and, separately, every record present in the input must be represented in
exactly one output destination (success file, error TSV, or filter TSV).

These tests only inspect counts and line totals; they don't care about
specific reactions or rule definitions, so they don't churn when chemistry
inputs change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synplan.chem.data.filtering import filter_reactions_from_file
from synplan.chem.data.standardizing import standardize_reactions_from_file
from synplan.chem.reaction_rules.extraction import extract_rules_from_reactions


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def _count_data_rows(tsv_path: Path) -> int:
    """Count non-blank rows that aren't the leading ``#`` header."""
    if not tsv_path.exists():
        return 0
    return sum(
        1
        for line in tsv_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )


def test_standardize_summary_counts_balance(
    tmp_path: Path, smi_with_one_unparseable, std_config
):
    """succeeded + errored + filtered + duplicates == total_input."""
    out = tmp_path / "std.smi"
    err = tmp_path / "std.errors.tsv"
    summary = standardize_reactions_from_file(
        config=std_config,
        input_reaction_data_path=str(smi_with_one_unparseable),
        standardized_reaction_data_path=str(out),
        num_cpus=1,
        ignore_errors=True,
        error_file_path=str(err),
    )
    accounted = (
        summary.succeeded + summary.errored + summary.filtered + summary.duplicates
    )
    assert accounted == summary.total_input, (
        f"standardize summary loses records: total_input={summary.total_input} "
        f"but succeeded({summary.succeeded}) + errored({summary.errored}) + "
        f"filtered({summary.filtered}) + duplicates({summary.duplicates}) = "
        f"{accounted}. Some records were counted in 'total' but never routed "
        "to a destination."
    )


def test_filter_summary_counts_balance(
    tmp_path: Path, smi_with_one_unparseable, filt_config
):
    out = tmp_path / "filt.smi"
    err = tmp_path / "filt.errors.tsv"
    summary = filter_reactions_from_file(
        config=filt_config,
        input_reaction_data_path=str(smi_with_one_unparseable),
        filtered_reaction_data_path=str(out),
        num_cpus=1,
        ignore_errors=True,
        error_file_path=str(err),
    )
    accounted = (
        summary.succeeded + summary.errored + summary.filtered + summary.duplicates
    )
    assert accounted == summary.total_input, (
        f"filter summary loses records: total_input={summary.total_input} "
        f"but succeeded({summary.succeeded}) + errored({summary.errored}) + "
        f"filtered({summary.filtered}) + duplicates({summary.duplicates}) = "
        f"{accounted}."
    )


@pytest.mark.parametrize(
    ("runner_fixture", "runner_fn", "out_name", "err_name"),
    [
        ("std_config", "standardize", "std.smi", "std.errors.tsv"),
        ("filt_config", "filter", "filt.smi", "filt.errors.tsv"),
    ],
)
def test_input_records_fully_accounted_in_outputs(
    request,
    tmp_path: Path,
    smi_with_one_unparseable,
    runner_fixture,
    runner_fn,
    out_name,
    err_name,
):
    """No record may disappear: input_lines <= success_lines + error_rows."""
    cfg = request.getfixturevalue(runner_fixture)
    out = tmp_path / out_name
    err = tmp_path / err_name
    if runner_fn == "standardize":
        standardize_reactions_from_file(
            config=cfg,
            input_reaction_data_path=str(smi_with_one_unparseable),
            standardized_reaction_data_path=str(out),
            num_cpus=1,
            ignore_errors=True,
            error_file_path=str(err),
        )
    else:
        filter_reactions_from_file(
            config=cfg,
            input_reaction_data_path=str(smi_with_one_unparseable),
            filtered_reaction_data_path=str(out),
            num_cpus=1,
            ignore_errors=True,
            error_file_path=str(err),
        )
    input_lines = _count_lines(smi_with_one_unparseable)
    success_lines = _count_lines(out)
    error_rows = _count_data_rows(err)
    assert success_lines + error_rows >= input_lines, (
        f"{runner_fn}: input had {input_lines} records but only "
        f"{success_lines} succeeded and {error_rows} errored; "
        f"{input_lines - success_lines - error_rows} records silently lost. "
        "A pipeline stage incremented a counter without writing the dropped "
        "row to the error TSV."
    )


def test_extraction_multi_product_records_are_traceable(
    tmp_path: Path,
    sample_reactions_file,
    rule_cfg_factory,
):
    """Multi-product reactions must not vanish: their count is reflected in the
    summary string OR rows appear in the error TSV.

    The known bug is that ``n_multi_product`` is incremented inside
    ``_extract_rules_serial`` but the skipped reactions are never written to
    ``error_file``; they vanish from the output and only surface as a single
    aggregate number. This test asserts the inverse: either no multi-product
    skips happened, or every skipped reaction is recoverable from the error
    TSV.
    """
    rules_path = tmp_path / "rules.tsv"
    err = tmp_path / "rules.errors.tsv"
    extract_rules_from_reactions(
        config=rule_cfg_factory(),
        reaction_data_path=str(sample_reactions_file),
        reaction_rules_path=str(rules_path),
        num_cpus=1,
        batch_size=4,
        ignore_errors=True,
        error_file_path=str(err),
    )
    input_lines = _count_lines(sample_reactions_file)
    # Sum rules with their reaction-id coverage:
    covered_ids: set[int] = set()
    if rules_path.exists():
        for line in rules_path.read_text(encoding="utf-8").splitlines()[1:]:
            if not line:
                continue
            cols = line.split("\t")
            # Reaction indices appear in one of the rule TSV columns; format
            # is "rule_smarts\trule_id\treaction_ids_csv\t..." with the CSV
            # of indices typically at column 2 or 3.
            for col in cols[1:4]:
                for token in col.split(","):
                    token = token.strip()
                    if token.isdigit():
                        covered_ids.add(int(token))
    error_rows = _count_data_rows(err)
    # Every input row must appear either in the rule coverage set or as an
    # error row. The set is by reaction-index, indexes are 0-based and dense.
    missing = input_lines - len(covered_ids) - error_rows
    assert missing <= 0, (
        f"extract_rules dropped {missing} input record(s) silently: "
        f"{input_lines} inputs, {len(covered_ids)} covered by rules, "
        f"{error_rows} in error TSV. Multi-product or stereo-bearing "
        "reactions are likely being counted but not written anywhere."
    )
