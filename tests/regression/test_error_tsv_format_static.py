"""Static-source contract: every pipeline stage must reference the canonical
5-column error TSV header literal in its source.

Mapping (``synplan/chem/data/mapping.py``) cannot be driven through a real
end-to-end pipeline run from a unit test because it requires the atom-mapping
model. Its error-file bugs (3-column rows, missing header) are real but
invisible to ``test_error_tsv_format.py``.

This test cheaply closes that gap by asserting a *static* contract: the
exact reference header line must appear verbatim in every pipeline module
that writes an error TSV. Catches:

* mapping.py never writing a header at all,
* mapping.py writing a 3-column row format,
* extraction.py writing a 4-column header (no ``source_info``).

It does not catch runtime tab-injection (the runtime test handles that for
standardize/filter/extract). It does catch the *spec drift* bug class where a
new stage author copies the wrong format.

Fragile dimensions: if the canonical header is intentionally evolved (a new
column added), this test breaks in one place; update the constant in
``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import ERROR_TSV_HEADER

# Modules that write per-record error TSVs at end-of-pipeline. New stages
# must opt in by appending here.
PIPELINE_MODULES_WITH_ERROR_TSV = [
    "synplan/chem/data/standardizing.py",
    "synplan/chem/data/filtering.py",
    "synplan/chem/data/mapping.py",
    "synplan/chem/reaction_rules/extraction.py",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module_path", PIPELINE_MODULES_WITH_ERROR_TSV)
def test_module_references_canonical_header(module_path: str):
    """The 5-column reference header literal must appear in the source.

    ``ERROR_TSV_HEADER`` is the source-of-truth constant; we don't pin the
    surrounding code (any wrapping/escaping is fine), only that the literal
    line is somewhere in the file.
    """
    p = _repo_root() / module_path
    assert p.exists(), f"module {module_path} not found at {p}"
    src = p.read_text(encoding="utf-8")
    # The source writes the header through an f-string or .write call; the
    # literal contains real tab characters and a trailing newline. We accept
    # either the bare literal or the escaped form ("\\t").
    needles = [
        ERROR_TSV_HEADER,
        ERROR_TSV_HEADER.replace("\t", "\\t"),
    ]
    hit = any(n in src for n in needles)
    assert hit, (
        f"{module_path} does not contain the canonical 5-column error-TSV "
        f"header literal {ERROR_TSV_HEADER!r}. Pipelines that write a "
        "different header (4-column extraction-style or 3-column mapping-"
        "style) break the cross-stage error-TSV invariant tested in "
        "test_error_tsv_format.py."
    )
