"""Shared fixtures for regression tests built from the multi-agent bug review.

These tests assert *invariants* the pipeline must hold for any future input, so
they're designed to surface behavioural regressions without needing to update
the assertions when rule definitions, models, or chython internals change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REGRESSION_DATA_DIR = Path(__file__).parent.parent / "data" / "regression"
SUSPICIOUS_SAMPLE = REGRESSION_DATA_DIR / "suspicious_sample.smi"


_STEREO_TOKENS = ("[C@", "[N@", "/C", "\\C")


@pytest.fixture(scope="session")
def suspicious_sample_path() -> Path:
    """80 real reactions (stereo-biased) lifted from local/suspicious_reactions.smi.

    The sample is checked in and must keep its stereo-biased character; tests
    that exercise stereo handling silently pass on a stereo-free corpus, so we
    validate the fixture itself before handing it to a test.
    """
    assert SUSPICIOUS_SAMPLE.exists(), (
        f"missing fixture {SUSPICIOUS_SAMPLE}; regenerate from "
        "local/suspicious_reactions.smi (see commit history for the seeding "
        "script)"
    )
    text = SUSPICIOUS_SAMPLE.read_text(encoding="utf-8")
    line_count = sum(1 for line in text.splitlines() if line.strip())
    assert line_count >= 50, (
        f"suspicious_sample.smi has only {line_count} lines — too small to "
        "meaningfully exercise stereo / multi-reaction code paths. Regenerate."
    )
    stereo_count = sum(
        1 for line in text.splitlines() if any(tok in line for tok in _STEREO_TOKENS)
    )
    assert stereo_count >= 20, (
        f"suspicious_sample.smi has only {stereo_count} stereo-bearing rows "
        "(expected >=20). The stereo handling tests will silently pass on a "
        "stereo-free corpus and stop catching the ReactorValidation "
        "regression. Regenerate with a stereo-biased seed."
    )
    return SUSPICIOUS_SAMPLE


@pytest.fixture
def smi_with_tabs_and_source(tmp_path: Path, sample_reactions) -> Path:
    """A .smi file whose records carry tab-separated source columns.

    Real USPTO-style records look like:
        <reaction_smiles>\\t<reaction_id>\\t<patent_ids>

    Several bugs (mapping TSV column shift, extraction error orig field) only
    manifest when the input has these source columns. Tests that feed clean
    reactions miss the bug entirely.
    """
    p = tmp_path / "reactions_with_sources.smi"
    lines = []
    for i, rxn in enumerate(sample_reactions):
        # Mix in source columns so tab-injection bugs become observable.
        lines.append(f"{rxn}\t{i}\tUS{1000000 + i:07d},US{2000000 + i:07d}")
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.fixture
def smi_with_one_unparseable(tmp_path: Path, sample_reactions) -> Path:
    """A .smi file with mostly-valid reactions plus one syntactically broken row.

    Used to assert that pipeline stages either record the failure in the error
    TSV or surface it via exception — they must not silently drop it.
    """
    p = tmp_path / "reactions_with_broken.smi"
    lines = list(sample_reactions)
    # Insert a junk line that won't parse as SMILES.
    lines.insert(len(lines) // 2, "this_is_not_a_reaction>>>nope")
    p.write_text("\n".join(lines) + "\n")
    return p


# ----- TSV invariants -------------------------------------------------------- #

#: The pipeline-wide reference error-file header. Every stage's error file must
#: write this exact header and emit rows with this many tab-separated columns.
ERROR_TSV_HEADER = "# original_smiles\tsource_info\tstage\terror_type\terror_message"
ERROR_TSV_COLUMNS = 5


def parse_error_tsv(path: Path) -> tuple[str, list[list[str]]]:
    """Return (header_line, rows) for an error TSV.

    Rows are split on raw '\\t'. Returning the raw split lets the caller detect
    column-count drift (the canonical symptom of tab-injection bugs).
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return "", []
    header = lines[0]
    rows = [line.split("\t") for line in lines[1:] if line]
    return header, rows
